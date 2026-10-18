"""Recheck previous phase audit using condition-specific norms and complex128 sums."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from audit_ftu_adc_phase import pack,load_unpack
root=Path('experiment_runs/ftu_adc_controlled');oldroot=Path('experiment_runs/ftu_adc_phase');started=time.perf_counter()
unpack=load_unpack(root/'sources/current_ftu_loader.py');ci=np.linspace(0,127,64,dtype=int);rows=[];timing=[]
with np.load(oldroot/'synthetic_signals.npz') as old:
    for case in ['frame_constant','continuous_motion']:
        refs={(mode,clutter,dtype):[] for mode in ['legacy','full_fft_crop'] for clutter in ['chirp_mean','none'] for dtype in ['complex64','complex128']}
        for j in range(400):
            t=j/20;times=t+(np.arange(128)*57e-6 if case=='continuous_motion' else np.zeros(128))
            phase=.2*np.sin(2*np.pi*1.2*times)
            adc=(1000*np.exp(1j*phase)[None,None,:,None]*np.exp(2j*np.pi*12*np.arange(250)/250)[None,None,None,:]*np.ones((1,4,1,1))).astype(np.complex64)
            frame=unpack(pack(adc),1)[0]
            for dtype in ['complex64','complex128']:
                x=frame[:,ci].astype(dtype)
                for clutter in ['chirp_mean','none']:
                    clean=x-x.mean(axis=1,keepdims=True) if clutter=='chirp_mean' else x
                    rg=np.fft.fft(clean,axis=-1)
                    for mode,n in [('legacy',8),('full_fft_crop',64)]:refs[(mode,clutter,dtype)].append(rg[:,:n,12].sum(dtype=dtype))
        for (mode,clutter,dtype),vals in refs.items():
            ref=np.array(vals);z=old[case+'_'+mode+'_'+clutter];d=z-ref;n=8 if mode=='legacy' else 64;gain=1000*250*4*n
            rn=float(np.linalg.norm(ref));dn=float(np.linalg.norm(d));defined=rn>gain*np.sqrt(400)*1e-6
            rows.append(dict(case=case,variant=mode+'_'+clutter,reference_precision=dtype,max_abs_error=float(abs(d).max()),difference_l2=dn,
                reference_l2=rn,relative_l2=dn/rn if defined else np.nan,rms_error_over_theoretical_gain=dn/np.sqrt(400)/gain,
                reference_rms_over_theoretical_gain=rn/np.sqrt(400)/gain,relative_defined=defined))
        if case=='continuous_motion':
            t=np.arange(400)/20
            for mode,n in [('legacy',8),('full_fft_crop',64)]:
                tau=float(np.mean(ci[:n])*57e-6);truth=.2*np.sin(2*np.pi*1.2*t);shifted=.2*np.sin(2*np.pi*1.2*(t+tau))
                z=old[case+'_'+mode+'_none'];p=np.r_[0,np.cumsum(np.angle(z[1:].astype(complex)*z[:-1].conj()))]
                center=lambda x:x-x[0]
                timing.append(dict(mode=mode,centroid_delay_ms=tau*1000,phase_rmse_chirp0=float(np.sqrt(np.mean((p-center(truth))**2))),
                    predicted_time_only_rmse=float(np.sqrt(np.mean((center(shifted)-center(truth))**2))),
                    phase_rmse_mean_time=float(np.sqrt(np.mean((p-center(shifted))**2)))))
with np.load(oldroot/'stage_phase_traces.npz') as stages,np.load(oldroot/'synthetic_signals.npz') as signal:
    for case in ['frame_constant','continuous_motion']:
        for mode,n in [('legacy',8),('full_fft_crop',64)]:
            for clutter in ['chirp_mean','none']:
                variant=mode+'_'+clutter;ref=stages[case+'_'+variant+'_angle_target'];d=signal[case+'_'+variant]-ref
                rn=float(np.linalg.norm(ref));dn=float(np.linalg.norm(d));gain=1000000*n;defined=rn>gain*20*1e-6
                rows.append(dict(case=case,variant=variant,reference_precision='original_saved_sum_order',max_abs_error=float(abs(d).max()),difference_l2=dn,
                    reference_l2=rn,relative_l2=dn/rn if defined else np.nan,rms_error_over_theoretical_gain=dn/20/gain,
                    reference_rms_over_theoretical_gain=rn/20/gain,relative_defined=defined,old_floor_statistic_explicitly_not_relative=dn/max(rn,1)))
pd.DataFrame(rows).to_csv(root/'previous_precision_recheck.csv',index=False);pd.DataFrame(timing).to_csv(root/'previous_timing_correction.csv',index=False)
(root/'reporting_correction_execution.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-started,complex128_recheck=True,training=False),indent=2))
print(pd.DataFrame(timing).to_string(index=False));print(pd.DataFrame(rows).to_string(index=False))
