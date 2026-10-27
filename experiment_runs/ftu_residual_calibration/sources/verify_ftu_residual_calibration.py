"""Independent full-window identity, synthesis, periodogram and residual checks."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import periodogram

ROOT=Path('experiment_runs/ftu_residual_calibration')
SRC=Path('experiment_runs/ftu_reference_support_v2')


def main():
    begin=time.perf_counter();p=json.loads((ROOT/'protocol.json').read_text())
    params=pd.read_csv(ROOT/'generation_parameters.csv');metrics=pd.read_csv(ROOT/'paired_metrics.csv')
    windows=pd.read_csv(ROOT/'window_readouts.csv');refs=pd.read_csv(SRC/'polar_reference_values.csv')
    nsignals=0;nreadouts=0;maxtrace=0.;maxphase=0.;maxmetric=0.;metricchecks=0
    for tag,g in params.groupby('record'):
        d=np.load(ROOT/'signals'/f'{tag}.npz');df=refs[refs.record==tag]
        t=df.time_second.to_numpy();y=df.hr_bpm.to_numpy();grid=np.arange(1200)/20
        # Centered closed-form linear regression, independent of np.polyfit.
        tt=t-t.mean();slope=np.dot(tt,y-y.mean())/np.dot(tt,tt)
        trend=y.mean()+slope*(grid-t.mean());h=np.interp(grid,t,y,left=np.nan,right=np.nan)
        freqs=np.array([trend+alpha*(h-trend) for alpha in p['alphas']])
        for ai,alpha in enumerate(p['alphas']):
            np.testing.assert_allclose(freqs[ai],d['frequencies'][ai],atol=1e-10,equal_nan=True)
            valid=np.isfinite(freqs[ai]);derivative=np.diff(d['known_boundary_phases'][ai])[valid]*60*20/(2*np.pi)
            np.testing.assert_allclose(derivative,freqs[ai,valid],atol=1e-9)
        for _,row in g.iterrows():
            ai=p['alphas'].index(row.alpha);seed=int(row.seed);sigma=row.noise_sigma
            rng=np.random.default_rng(np.random.SeedSequence([seed,int(row.record_index)]));phi0=rng.uniform(0,2*np.pi);noise=rng.normal(size=1200)
            np.testing.assert_allclose(phi0,row.initial_phase_rad,atol=1e-14)
            f=freqs[ai];valid=np.isfinite(f);count=valid.sum()
            phase=np.r_[0,np.cumsum(f[:count])*(2*np.pi/(60*20))]
            x=np.full(1200,np.nan);x[:count]=np.cos(phase[:-1]+phi0)+sigma*noise[:count]
            err=float(np.nanmax(abs(x-d[row.signal_id])));maxtrace=max(maxtrace,err)
            np.testing.assert_allclose(x,d[row.signal_id],atol=2e-9,equal_nan=True);nsignals+=1
            mask=(windows.record==tag)&(windows.seed==seed)&(windows.noise_sigma==sigma)&(windows.alpha==row.alpha)
            for length,vals in windows[mask].groupby('length'):
                length=int(length);starts=vals.start_frame.to_numpy(dtype=int)
                samples=np.stack([d[row.signal_id][s:s+length] for s in starts])
                hz,power=periodogram(samples,fs=20,window=np.hanning(length),detrend='linear',nfft=4096,axis=-1,scaling='spectrum')
                band=(hz*60>=45)&(hz*60<=150);bpm=hz[band]*60
                peak=bpm[power[:,band].argmax(axis=1)]
                np.testing.assert_array_equal(peak,vals.fft_hr)
                labels=np.array([f[s:s+length].mean() for s in starts])
                np.testing.assert_allclose(labels,vals.target_hr,atol=1e-10)
                maxphase=max(maxphase,float(abs(vals.ideal_hr-labels).max()));nreadouts+=len(vals)
    keys=['record','length','seed','noise_sigma','alpha']
    groups={key:g for key,g in windows.groupby(keys)}
    for _,row in metrics.iterrows():
        key=tuple(row[k] for k in keys);g=groups.get(key)
        if g is None:
            assert row.windows==0 and pd.isna(row.mae);continue
        y=g.target_hr.to_numpy();m=g.trend_hr.to_numpy()
        pred=g[(row.readout+'_hr')].to_numpy();base=g[(row.readout+'_trend_hr')].to_numpy()
        r=y-m;delta=pred-base;rc=r-r.mean();dc=delta-delta.mean()
        values=dict(mae=float(abs(pred-y).mean()),mae_gain=float(abs(base-y).mean()-abs(pred-y).mean()),
            known_trend_mae_gain=float(abs(m-y).mean()-abs(pred-y).mean()),
            residual_rmse=float(np.sqrt(np.mean((delta-r)**2))),centered_residual_rmse=float(np.sqrt(np.mean((dc-rc)**2))))
        values['residual_r2']=float(1-np.sum((dc-rc)**2)/np.sum(rc**2)) if len(g)>=5 and rc.std()>1e-6 else np.nan
        for name,value in values.items():
            np.testing.assert_allclose(value,row[name],atol=2e-8,rtol=1e-8,equal_nan=True)
            if np.isfinite(value):maxmetric=max(maxmetric,abs(value-row[name]))
            metricchecks+=1
    summary=pd.read_csv(ROOT/'record_summary.csv');checks=0
    for key,g in metrics.groupby(['record','length','noise_sigma','alpha','readout']):
        match=summary
        for name,value in zip(['record','length','noise_sigma','alpha','readout'],key):match=match[match[name]==value]
        assert len(match)==1;row=match.iloc[0]
        for col in ['mae','mae_gain','known_trend_mae_gain','residual_r2','centered_residual_rmse']:
            a=np.array(g[col].dropna())
            for suffix,q in [('q05',.05),('median',.5),('q95',.95)]:
                value=np.quantile(a,q) if len(a) else np.nan
                np.testing.assert_allclose(value,row[col+'_'+suffix],atol=2e-8,rtol=1e-8,equal_nan=True);checks+=1
    prec=pd.read_csv(ROOT/'precision_annotations.csv');precisionchecks=0
    for _,row in prec.iterrows():
        a=windows[(windows.record==row.record)&(windows.length==row.length)&(windows.noise_sigma==row.noise_sigma)&(windows.alpha==0)]
        if a.empty:
            assert pd.isna(row.fft_operational_floor);continue
        values=a.pivot(index='seed',columns='start_frame',values='fft_hr').to_numpy()
        median=np.median(values,axis=0)
        jitter=[]
        for trace in values:
            delta=trace-median;delta-=delta.mean();jitter.append(np.sqrt(np.mean(delta**2)))
        floor=max(.29296875/np.sqrt(12),float(np.quantile(jitter,.95)))
        np.testing.assert_allclose(floor,row.fft_operational_floor,atol=1e-12);precisionchecks+=1
    secondary=pd.read_csv(ROOT/'unchanged_shift_diagnostics.csv')
    assert sorted(secondary['shift'].unique())==[-20,-15,-10,-5,0]
    assert sorted(secondary.offset.unique())==[0,1,2]
    assert len(secondary)==6*32*2*4*2*3*5
    for source,h in p['source_hashes'].items():assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==h
    assert maxphase<=p['phase_check_tolerance_bpm']
    result=dict(elapsed_seconds=time.perf_counter()-begin,signals_checked=nsignals,window_readouts_checked=nreadouts,
        paired_metric_checks=metricchecks,conditional_quantile_checks=checks,precision_floor_checks=precisionchecks,
        retained_shift_rows=len(secondary),maximum_trace_difference=maxtrace,
        maximum_phase_mean_error_bpm=maxphase,maximum_metric_difference=maxmetric,old_0_of_6_preserved=True,
        new_radar_read=False,independent_validation_of_real_tracking=False)
    (ROOT/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
