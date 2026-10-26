"""Small representation-level synthetic calibration, no radar input or training."""
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from audit_ftu_structure import spectrum,scalar_metrics,FREQ

ROOT=Path('experiment_runs/ftu_evaluation_controls')
SRC=Path('experiment_runs/ftu_reference_support_v2')
SHIFTS=[0,-20,-15,-10,-5]


def signal(f,phase,noise,sigma):
    valid=np.isfinite(f);x=np.full_like(f,np.nan,dtype=float)
    idx=np.flatnonzero(valid)
    assert (np.diff(idx)==1).all(), 'Do not integrate across missing HR'
    theta=phase+2*np.pi*cumulative_trapezoid(f[valid]/60,dx=.05,initial=0)
    x[valid]=np.cos(theta)+sigma*noise[valid]
    return x


def main():
    begin=time.perf_counter();protocol=json.loads((ROOT/'protocol.json').read_text())
    for name,h in protocol['source_files'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    refs=pd.read_csv(SRC/'polar_reference_values.csv');support=pd.read_csv(SRC/'selected_support.csv')
    old_decision=json.loads((SRC/'decision.json').read_text());assert old_decision['evaluable_records']==0
    (ROOT/'signals').mkdir(exist_ok=True)
    metrics=[];windows=[];parameters=[]
    for ri,tag in enumerate(sorted(refs.record.unique())):
        data=refs[refs.record==tag];t=data.time_second.to_numpy();hr=data.hr_bpm.to_numpy()
        grid=np.arange(1200)/20
        fpos={off:np.interp(grid+off,t,hr,left=np.nan,right=np.nan) for off in range(3)}
        mean=float(hr.mean());slope,intercept=np.polyfit(t,hr,1)
        freq={'positive':fpos[0], 'constant':np.where(np.isfinite(fpos[0]),mean,np.nan),
              'trend':np.where(np.isfinite(fpos[0]),intercept+slope*grid,np.nan)}
        saved=dict(time_s=grid,reference_time=t,reference_hr=hr,
                   positive_frequency_offset0=fpos[0],positive_frequency_offset1=fpos[1],positive_frequency_offset2=fpos[2],
                   constant_frequency=freq['constant'],trend_frequency=freq['trend'])
        for seed in protocol['seeds']:
            rng=np.random.default_rng(np.random.SeedSequence([seed,ri]))
            phase=float(rng.uniform(0,2*np.pi));noise=rng.normal(size=1200)
            saved[f'noise_seed{seed}']=noise
            for sigma in protocol['noise_sigma']:
                designs=[('primary',kind,0,freq[kind]) for kind in freq]
                designs += [('matched_secondary','positive',off,fpos[off]) for off in [1,2]]
                for mode,kind,gen_off,f in designs:
                    x=signal(f,phase,noise,sigma)
                    name=f'{mode}_{kind}_go{gen_off}_s{seed}_noise{sigma}'
                    saved[name]=x
                    parameters.append(dict(record=tag,signal_id=name,seed=seed,record_index=ri,phase_rad=phase,
                        noise_sigma=sigma,mode=mode,generator=kind,generation_offset=gen_off,
                        record_mean_hr=mean,trend_intercept=intercept,trend_slope=slope,
                        finite_frames=int(np.isfinite(x).sum()),min_frequency=float(np.nanmin(f)),max_frequency=float(np.nanmax(f))))
                    for length in protocol['window_lengths']:
                        selected=support[(support.record==tag)&(support.length==length)]
                        starts=np.array(json.loads(selected.iloc[0].all_offsets_common_starts),dtype=int)
                        assert (selected.all_offsets_common_starts==selected.iloc[0].all_offsets_common_starts).all()
                        segments=np.array([x[s:s+length] for s in starts]).reshape(len(starts),length)
                        valid=np.isfinite(segments).all(axis=1)[None]
                        p=spectrum(segments)[None] if len(starts) else np.empty((1,0,len(FREQ)))
                        peak=FREQ[p[0].argmax(axis=1)] if len(starts) else np.array([])
                        offs=range(3) if mode=='primary' else [gen_off]
                        for off in offs:
                            for shift in SHIFTS:
                                ys=np.array([np.interp(np.arange(s,s+length)/20+off+shift,t,hr,left=np.nan,right=np.nan).mean() for s in starts])
                                assert np.isfinite(ys).all()
                                met=scalar_metrics(p,valid,ys,1200/length)
                                metrics.append(dict(record=tag,mode=mode,generator=kind,generation_offset=gen_off,
                                    seed=seed,noise_sigma=sigma,length=length,offset=off,shift=shift,
                                    common_windows=len(starts),**met))
                                for j,s in enumerate(starts):
                                    near=float(p[0,j,abs(FREQ-ys[j])<=1200/length].sum())
                                    fw=f[s:s+length];hann=np.hanning(length)**2
                                    windows.append(dict(record=tag,mode=mode,generator=kind,generation_offset=gen_off,
                                        seed=seed,noise_sigma=sigma,length=length,offset=off,shift=shift,start_frame=int(s),
                                        center_s=(s+length/2)/20,peak_bpm=peak[j],label_hr=ys[j],near_power=near,
                                        mean_generation_hr=float(fw.mean()),hann_energy_mean_generation_hr=float(np.sum(fw*hann)/hann.sum())))
        np.savez_compressed(ROOT/'signals'/f'{tag}.npz',**saved)
        print(tag,'synthetic controls complete',flush=True)
    met=pd.DataFrame(metrics);met.to_csv(ROOT/'metrics.csv',index=False)
    pd.DataFrame(windows).to_csv(ROOT/'window_metrics.csv',index=False)
    pd.DataFrame(parameters).to_csv(ROOT/'generation_parameters.csv',index=False)
    keys=['record','mode','generator','generation_offset','seed','noise_sigma','length','offset']
    gates=[]
    for key,g in met.groupby(keys):
        a=g[g['shift']==0].iloc[0];b=g[g['shift']!=0]
        coverage=bool(a.coverage>=.9);hit=bool(a.within5>=.6)
        power=bool((a.near_power-b.near_power>=.01).all())
        mae=bool((a.mae<b.mae).all());enough=bool(a.common_windows>=5)
        gates.append(dict(zip(keys,key),common_windows=int(a.common_windows),coverage_pass=coverage,
            hit_pass=hit,power_pass=power,mae_pass=mae,
            score_rule_pass=bool(enough and coverage and hit and power and mae),
            minimum_power_advantage=float((a.near_power-b.near_power).min()),
            minimum_mae_advantage=float((b.mae-a.mae).min()),
            original_reference_admitted=False,formal_real_data_pass=False))
    gates=pd.DataFrame(gates);gates.to_csv(ROOT/'score_rule_diagnostics.csv',index=False)
    primary=gates[(gates['mode']=='primary')&(gates.length==512)&(gates.noise_sigma==0)]
    tally=[]
    for (kind,seed),g in primary.groupby(['generator','seed']):
        tally.append(dict(generator=kind,seed=int(seed),offset0_passes=int(g[g.offset==0].score_rule_pass.sum()),
            all_offsets_passes=int(g.groupby('record').score_rule_pass.all().sum())))
    tally=pd.DataFrame(tally);tally.to_csv(ROOT/'noiseless_tally.csv',index=False)
    pos=tally[tally.generator=='positive'];neg=tally[tally.generator!='positive']
    adequate=bool((pos.offset0_passes>=5).all() and (pos.all_offsets_passes>=5).all() and (neg.offset0_passes==0).all())
    (ROOT/'decision.json').write_text(json.dumps(dict(calibration_adequate=adequate,
        previous_reference_admission='0/6 unchanged',real_radar_performance_claim=False,
        independent_validation=False,replacement_rule_selected=False,
        next_step='develop and independently freeze better evaluation before dynamic tracking claims' if not adequate else 'rules pass development calibration only; freeze before independent-record validation',
        full_rda_reconstruction=False,training=False),indent=2))
    (ROOT/'execution.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-begin,
        command='uv run --locked --no-sync python audit_ftu_evaluation_controls.py',
        protocol_commit='a37e447',runtime_git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        numpy=np.__version__,pandas=pd.__version__,local=True,training=False,new_radar_read=False,
        verified_input_hashes=len(protocol['source_files'])),indent=2))
    print(tally.to_string(index=False));print('Calibration adequate:',adequate)


if __name__=='__main__':main()
