"""Finite development calibration of full-window HR and beyond-trend residuals."""
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from audit_ftu_structure import spectrum,scalar_metrics,FREQ

ROOT=Path('experiment_runs/ftu_residual_calibration')
SRC=Path('experiment_runs/ftu_reference_support_v2')


def phase_boundaries(f):
    valid=np.isfinite(f);count=int(valid.sum())
    assert valid[:count].all() and not valid[count:].any()
    phi=np.full(len(f)+1,np.nan);phi[:count+1]=np.r_[0,np.cumsum(2*np.pi*f[:count]/(60*20))]
    return phi


def residual_metrics(pred,baseline,target,trend):
    n=len(target)
    keys=['mae','rmse','trend_mae','mae_gain','known_trend_mae_gain','target_residual_mean','target_residual_rms',
          'target_centered_rms','residual_bias','residual_rmse','centered_residual_rmse','residual_r2','residual_r','residual_gain']
    if not n:return {k:np.nan for k in keys}
    r=target-trend;d=pred-baseline;rc=r-r.mean();dc=d-d.mean();den=np.dot(rc,rc)
    mae=float(abs(pred-target).mean());base=float(abs(baseline-target).mean())
    evaluate=n>=5 and np.sqrt(np.mean(rc**2))>1e-6
    return dict(mae=mae,rmse=float(np.sqrt(np.mean((pred-target)**2))),trend_mae=base,mae_gain=base-mae,
        known_trend_mae_gain=float(abs(trend-target).mean()-mae),target_residual_mean=float(r.mean()),
        target_residual_rms=float(np.sqrt(np.mean(r**2))),target_centered_rms=float(np.sqrt(np.mean(rc**2))),
        residual_bias=float((d-r).mean()),residual_rmse=float(np.sqrt(np.mean((d-r)**2))),
        centered_residual_rmse=float(np.sqrt(np.mean((dc-rc)**2))),
        residual_r2=float(1-np.dot(dc-rc,dc-rc)/den) if evaluate else np.nan,
        residual_r=float(np.corrcoef(rc,dc)[0,1]) if evaluate and dc.std()>1e-12 else np.nan,
        residual_gain=float(np.dot(dc,rc)/den) if evaluate else np.nan)


def main():
    begin=time.perf_counter();p=json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['source_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    refs=pd.read_csv(SRC/'polar_reference_values.csv');support=pd.read_csv(SRC/'selected_support.csv')
    (ROOT/'signals').mkdir(exist_ok=True);metrics=[];windows=[];secondary=[];parameters=[];precision=[]
    maxphaseerror=0.
    for ri,tag in enumerate(sorted(refs.record.unique())):
        df=refs[refs.record==tag];rt=df.time_second.to_numpy();hr=df.hr_bpm.to_numpy()
        grid=np.arange(1200)/20;observed=np.interp(grid,rt,hr,left=np.nan,right=np.nan)
        slope,intercept=np.polyfit(rt,hr,1);m=np.where(np.isfinite(observed),intercept+slope*grid,np.nan);r=observed-m
        frequencies=np.array([m+alpha*r for alpha in p['alphas']]);phis=np.array([phase_boundaries(f) for f in frequencies])
        saved=dict(time_s=grid,reference_time=rt,reference_hr=hr,trend=m,residual=r,alphas=np.array(p['alphas']),
            frequencies=frequencies,known_boundary_phases=phis)
        starts_by_length={length:np.array(json.loads(support[(support.record==tag)&(support.length==length)].iloc[0].all_offsets_common_starts),dtype=int) for length in p['lengths_frames']}
        labels={}
        for length,starts in starts_by_length.items():
            saved[f'starts_{length}']=starts
            for ai,alpha in enumerate(p['alphas']):
                for off in range(3):
                    for shift in [0,-20,-15,-10,-5]:
                        vals=[]
                        for s in starts:
                            q=np.arange(s,s+length)/20+off+shift;h=np.interp(q,rt,hr,left=np.nan,right=np.nan)
                            trend=intercept+slope*q;vals.append(np.mean(trend+alpha*(h-trend)))
                        label=np.array(vals);assert np.isfinite(label).all();labels[ai,length,off,shift]=label
        null_peaks={(length,sigma):[] for length in p['lengths_frames'] for sigma in p['noise_sigma']}
        for seed in p['seeds']:
            rng=np.random.default_rng(np.random.SeedSequence([seed,ri]));initial=float(rng.uniform(0,2*np.pi));noise=rng.normal(size=1200)
            saved[f'noise_seed{seed}']=noise
            for sigma in p['noise_sigma']:
                traces=np.cos(phis[:,:-1]+initial)+sigma*noise[None]
                traces[~np.isfinite(frequencies)]=np.nan
                for ai,alpha in enumerate(p['alphas']):
                    name=f'alpha{alpha}_seed{seed}_sigma{sigma}';saved[name]=traces[ai]
                    parameters.append(dict(record=tag,signal_id=name,record_index=ri,seed=seed,noise_sigma=sigma,alpha=alpha,
                        initial_phase_rad=initial,trend_intercept=intercept,trend_slope=slope,
                        finite_frames=int(np.isfinite(traces[ai]).sum()),
                        frequency_outside_band=int(((frequencies[ai]<45)|(frequencies[ai]>150)).sum())))
                for length,starts in starts_by_length.items():
                    seg=np.stack([traces[:,s:s+length] for s in starts],axis=1) if len(starts) else np.empty((4,0,length))
                    power=spectrum(seg) if len(starts) else np.empty((4,0,len(FREQ)))
                    fft=FREQ[power.argmax(axis=-1)] if len(starts) else np.empty((4,0))
                    ideal=np.array([60*(phi[starts+length]-phi[starts])/(2*np.pi*(length/20)) for phi in phis])
                    null_peaks[length,sigma].append(fft[0])
                    trend=labels[0,length,0,0]
                    for ai,alpha in enumerate(p['alphas']):
                        target=labels[ai,length,0,0]
                        if len(starts):maxphaseerror=max(maxphaseerror,float(abs(ideal[ai]-target).max()))
                        for readout,values in [('ideal',ideal),('fft',fft)]:
                            metrics.append(dict(record=tag,length=length,seed=seed,noise_sigma=sigma,alpha=alpha,
                                readout=readout,windows=len(starts),**residual_metrics(values[ai],values[0],target,trend)))
                        for wi,s in enumerate(starts):
                            windows.append(dict(record=tag,length=length,seed=seed,noise_sigma=sigma,alpha=alpha,
                                start_frame=int(s),center_s=(s+length/2)/20,target_hr=target[wi],trend_hr=trend[wi],
                                ideal_hr=ideal[ai,wi],ideal_trend_hr=ideal[0,wi],fft_hr=fft[ai,wi],fft_trend_hr=fft[0,wi]))
                        for off in range(3):
                            for shift in [0,-20,-15,-10,-5]:
                                valid=np.isfinite(seg[ai]).all(axis=-1)[None]
                                met=scalar_metrics(power[ai:ai+1],valid,labels[ai,length,off,shift],1200/length)
                                secondary.append(dict(record=tag,length=length,seed=seed,noise_sigma=sigma,alpha=alpha,
                                    offset=off,shift=shift,windows=len(starts),**met))
        for (length,sigma),values in null_peaks.items():
            a=np.array(values);jitter=np.nan
            if a.shape[1]:
                error=a-np.median(a,axis=0);error-=error.mean(axis=1,keepdims=True)
                jitter=float(np.quantile(np.sqrt(np.mean(error**2,axis=1)),.95))
            floor=max(.29296875/np.sqrt(12),jitter) if np.isfinite(jitter) else np.nan
            precision.append(dict(record=tag,length=length,noise_sigma=sigma,windows=a.shape[1],
                display_grid_rms_convention=.29296875/np.sqrt(12),null_centered_jitter_q95=jitter,fft_operational_floor=floor))
        np.savez_compressed(ROOT/'signals'/f'{tag}.npz',**saved)
        print(tag,'paired residual calibration complete',flush=True)
    assert maxphaseerror<=p['phase_check_tolerance_bpm']
    met=pd.DataFrame(metrics);met.to_csv(ROOT/'paired_metrics.csv',index=False)
    pd.DataFrame(windows).to_csv(ROOT/'window_readouts.csv',index=False)
    pd.DataFrame(secondary).to_csv(ROOT/'unchanged_shift_diagnostics.csv',index=False)
    pd.DataFrame(parameters).to_csv(ROOT/'generation_parameters.csv',index=False)
    prec=pd.DataFrame(precision);prec.to_csv(ROOT/'precision_annotations.csv',index=False)
    keys=['record','length','noise_sigma','alpha','readout'];summaries=[]
    quantities=['mae','mae_gain','known_trend_mae_gain','residual_rmse','centered_residual_rmse','residual_r2','residual_r','residual_gain']
    for key,g in met.groupby(keys):
        row=dict(zip(keys,key));row.update(windows=int(g.windows.iloc[0]),nuisance_realizations=len(g),
            target_residual_mean=float(g.target_residual_mean.iloc[0]),target_residual_rms=float(g.target_residual_rms.iloc[0]),target_centered_rms=float(g.target_centered_rms.iloc[0]))
        for col in quantities:
            finite=g[col].dropna()
            for label,q in [('q05',.05),('median',.5),('q95',.95)]:row[col+'_'+label]=float(finite.quantile(q)) if len(finite) else np.nan
        floor=prec[(prec.record==key[0])&(prec.length==key[1])&(prec.noise_sigma==key[2])].fft_operational_floor.iloc[0]
        row['fft_operational_floor']=floor
        if row['windows']<5:status='insufficient_common_support'
        elif row['alpha']==0:status='trend_control'
        elif row['target_centered_rms']<=1e-6:status='no_centered_dynamic_target'
        elif row['readout']=='ideal':status='exact_known_phase_statistic'
        elif row['target_centered_rms']<=floor:status='dynamic_unevaluable_at_this_fft_precision'
        elif row['mae_gain_q05']>0 and row['residual_r2_q05']>0:status='synthetic_stable_increment'
        else:status='no_stable_residual_evidence_at_this_readout'
        row['annotation']=status;summaries.append(row)
    summary=pd.DataFrame(summaries);summary.to_csv(ROOT/'record_summary.csv',index=False)
    old_hash=p['source_hashes'][str(SRC/'decision.json').replace('\\','/')]
    assert hashlib.sha256((SRC/'decision.json').read_bytes()).hexdigest()==old_hash
    (ROOT/'decision.json').write_text(json.dumps(dict(phase_mean_identity_pass=True,maximum_phase_mean_error_bpm=maxphaseerror,
        old_reference_gate='0/6 unchanged',new_5_of_6_gate=False,independent_validation=False,
        new_radar_read=False,training=False,rule_deletion=False,
        exit='Development-only per-record residual calibration complete; no detailed real tracking claim; mean-HR/structure scope retained until genuinely independent traceable-reference data.'),indent=2))
    (ROOT/'execution.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-begin,
        command='uv run --locked --no-sync python audit_ftu_residual_calibration.py',protocol_commit='5623878',
        runtime_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),local=True,
        numpy=np.__version__,pandas=pd.__version__,new_radar_read=False,training=False),indent=2))
    cols=['record','noise_sigma','target_centered_rms','mae_gain_median','residual_r2_median','annotation']
    print(summary[(summary.length==512)&(summary.alpha==1)&(summary.readout=='fft')][cols].to_string(index=False))
    print('Maximum phase-mean identity error:',maxphaseerror)


if __name__=='__main__':main()
