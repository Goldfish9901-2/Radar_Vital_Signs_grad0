"""Verify synthetic stages independently; archive provenance and diagnostic figures."""
import hashlib,json,time,shutil,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_ftu_adc_phase import load_conversion,load_unpack,pack,phase_score,digest

root=Path('experiment_runs/ftu_adc_phase');started=time.perf_counter()
protocol=json.loads((root/'protocol.json').read_text());ci=np.array(protocol['chirp_indices']);unpack=load_unpack(root/'sources/current_ftu_loader.py')
stage_rows=[];traces={};independent_final_error=0.;independent_relative_error=0.
with np.load(root/'synthetic_signals.npz') as data:
    for case in ['frame_constant','continuous_motion']:
        time_frame=np.arange(400)/20;truth=data[case+'_truth'];stages={}
        for clutter in ['chirp_mean','none']:
            for mode in ['legacy','full_fft_crop']:
                stages[(clutter,mode)]={k:[] for k in ['ADC_sample','post_clutter_sample','range_target','doppler_DC','angle_target']}
        for j,t in enumerate(time_frame):
            times=t+(np.arange(128)*57e-6 if case=='continuous_motion' else np.zeros(128))
            phase=.2*np.sin(2*np.pi*1.2*times)
            adc=(1000*np.exp(1j*phase)[None,None,:,None]*np.exp(2j*np.pi*12*np.arange(250)/250)[None,None,None,:]*np.ones((1,4,1,1))).astype(np.complex64)
            frame=unpack(pack(adc),1)[0];selected=frame[:,ci]
            for clutter in ['chirp_mean','none']:
                x=selected-selected.mean(axis=1,keepdims=True) if clutter=='chirp_mean' else selected
                rg=np.fft.fft(x,axis=-1)[...,:125]
                for mode,count in [('legacy',8),('full_fft_crop',64)]:
                    # DC is a direct sum, independent of production FFT implementation.
                    dc=rg[:,:count].sum(axis=1);beam=dc.sum(axis=0)
                    s=stages[(clutter,mode)]
                    for key,value in [('ADC_sample',selected[0,0,0]),('post_clutter_sample',x[0,0,0]),
                        ('range_target',rg[0,0,12]),('doppler_DC',dc[0,12]),('angle_target',beam[12])]:s[key].append(value)
        for (clutter,mode),st in stages.items():
            variant=mode+'_'+clutter;count=8 if mode=='legacy' else 64
            for stage,z in st.items():
                z=np.array(z);gain={'ADC_sample':1000,'post_clutter_sample':1000,'range_target':250000,'doppler_DC':250000*count,'angle_target':1000000*count}[stage]
                stage_rows.append(dict(case=case,variant=variant,stage=stage,median_amplitude_fraction=float(np.median(abs(z))/gain),**phase_score(z,truth,gain)))
                traces[case+'_'+variant+'_'+stage]=z
            err=float(np.max(abs(np.array(st['angle_target'])-data[case+'_'+variant])));independent_final_error=max(independent_final_error,err)
            independent_relative_error=max(independent_relative_error,float(np.linalg.norm(np.array(st['angle_target'])-data[case+'_'+variant])/max(np.linalg.norm(data[case+'_'+variant]),1.)))
            np.testing.assert_allclose(st['angle_target'],data[case+'_'+variant],rtol=2e-5,atol=.1)
    (root/'figures').mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for i,case in enumerate(['frame_constant','continuous_motion']):
        truth=data[case+'_truth'];axes[i,0].plot(time_frame,truth-truth[0],'k--',label='known phase')
        for variant in ['legacy_chirp_mean','full_fft_crop_chirp_mean','legacy_none','full_fft_crop_none']:
            z=data[case+'_'+variant];amplitude=abs(z);count=8 if variant.startswith('legacy_') else 64;gain=1000000*count
            axes[i,1].plot(time_frame,np.maximum(amplitude/gain,1e-15),label=variant)
            if np.all(amplitude>gain*1e-6):
                delta=np.angle(z[1:].astype(complex)*z[:-1].conj());phase=np.r_[0,delta.cumsum()]
                axes[i,0].plot(time_frame,phase,label=variant,alpha=.8)
        axes[i,0].set(title=case+' phase (valid only)',ylabel='Cumulative phase (rad)',xlabel='Frame time (s)')
        axes[i,1].set(title=case+' target amplitude',yscale='log',ylabel='Amplitude / coherent no-clutter gain',xlabel='Frame time (s)')
        for ax in axes[i]:ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(root/'figures'/'synthetic_phase_fidelity.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,mode in zip(axes,['legacy','full_fft_crop']):
        response=abs(data[mode+'_physical_response'])**2;f=data['physical_probe_hz']
        for k in [0,3,4,5,7]:ax.plot(f,response[k],label='index'+str(k))
        ax.set(title=mode+' actual-time response (no clutter)',xlabel='Probe chirp modulation (Hz), conditional PRI',ylabel='Normalized power response');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'figures'/'doppler_physical_response.png',dpi=140);plt.close(fig)
pd.DataFrame(stage_rows).to_csv(root/'stage_phase_metrics.csv',index=False);np.savez_compressed(root/'stage_phase_traces.npz',**traces)
test_start=time.perf_counter();tests=subprocess.run(['uv','run','--locked','--no-sync','python','validate_adc_rda_synthetic.py'],capture_output=True,text=True)
(root/'production_checks.txt').write_text(tests.stdout+tests.stderr);assert tests.returncode==0
validation=dict(elapsed_sec=time.perf_counter()-started,independent_stage_final_max_abs_error=independent_final_error,
    independent_stage_final_max_relative_l2=independent_relative_error,
    production_checks_elapsed_sec=time.perf_counter()-test_start,production_checks_passed=True,training=False)
(root/'synthetic_validation.json').write_text(json.dumps(validation,indent=2))
shutil.copyfile(__file__,root/'sources'/Path(__file__).name)
print(pd.DataFrame(stage_rows)[['case','variant','stage','valid_phase','phase_rmse_rad']].to_string(index=False))
