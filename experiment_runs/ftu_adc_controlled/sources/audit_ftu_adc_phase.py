"""FTU ADC-to-RDA provenance and phase-fidelity gate; never trains a model."""
import ast,hashlib,json,time,shutil,subprocess
from pathlib import Path
from typing import Optional,Tuple
import numpy as np
import pandas as pd

NAMES={'_select_even_indices','_crop_or_pad_last_axis','_range_window_indices','_build_rda_cube_from_frame','convert_adc_cube_to_rda'}
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def load_conversion(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    env=dict(np=np,Optional=Optional,Tuple=Tuple,TARGET_DOPPLER=8,TARGET_ANGLE=16,TARGET_RANGE=8)
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in NAMES],type_ignores=[]),str(path),'exec'),env)
    return env
def load_unpack(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FTUDataLoader')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_reshape_radar_data')
    env=dict(np=np,Optional=Optional)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
    class Constants:NUM_ADC=250;NUM_RX=4;NUM_CHIRPS=128;NUM_FRAMES=1200
    return lambda raw,n:env['_reshape_radar_data'](Constants(),raw,num_frames=n)
def pack(z):
    x=z.transpose(0,2,1,3).reshape(-1,4,125,2)
    a=np.stack([x.real[...,0],x.real[...,1],x.imag[...,0],x.imag[...,1]],axis=-1)
    return np.rint(a).astype('<i2').ravel()
def phase_score(z,truth,reference_amp):
    amplitude=abs(z).astype(float);defined=np.isfinite(z)&(amplitude>reference_amp*1e-6)
    if not defined.all():
        return dict(valid_phase=False,low_amplitude_frames=int((~defined).sum()),phase_rmse_rad=None,increment_rmse_rad=None,peak_bpm=None)
    delta=np.angle(z[1:].astype(np.complex128)*z[:-1].conj());expected=np.diff(truth)
    recovered=np.r_[0,np.cumsum(delta)];target=truth-truth[0]
    power=abs(np.fft.rfft((recovered-recovered.mean())*np.hanning(len(z))))**2
    hz=np.fft.rfftfreq(len(z),.05);band=(hz>=.5)&(hz<=2.75)
    return dict(valid_phase=True,low_amplitude_frames=0,phase_rmse_rad=float(np.sqrt(np.mean((recovered-target)**2))),
        increment_rmse_rad=float(np.sqrt(np.mean((delta-expected)**2))),peak_bpm=float(hz[band][power[band].argmax()]*60))
def synthetic(code,out):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
    env=load_conversion(code/'sources/current_export.py');unpack=load_unpack(code/'sources/current_ftu_loader.py')
    cfg=json.loads((code/'protocol.json').read_text());chirps=np.array(cfg['chirp_indices']);samples=np.arange(250)
    fs=20.;n=400;hr=1.2;pri=57e-6;time_frame=np.arange(n)/fs
    rows=[];stages=[];arrays={'time_s':time_frame,'chirp_indices':chirps,'chirp_time_s':chirps*pri}
    # One known target; fixed beam8, range12, common physical Doppler0 in all arms.
    fast=np.exp(2j*np.pi*12*np.arange(250)/250)
    for case in ['frame_constant','continuous_motion']:
        times=time_frame[:,None]+(np.arange(128)[None,:]*pri if case=='continuous_motion' else np.zeros((1,128)))
        phi=.2*np.sin(2*np.pi*hr*times)
        adc=(1000*np.exp(1j*phi)[:,None,:,None]*fast[None,None,None,:]*np.ones((1,4,1,1))).astype(np.complex64)
        truth=phi[:,0];arrays[case+'_truth']=truth
        raw=pack(adc[:2]);decoded=unpack(raw,2)
        np.testing.assert_allclose(decoded, np.rint(adc[:2].real)+1j*np.rint(adc[:2].imag),atol=0)
        for s in range(0,n,8):adc[s:s+8]=unpack(pack(adc[s:s+8]),min(8,n-s))
        for clutter in ['chirp_mean','none']:
            for mode in ['legacy','full_fft_crop']:
                variant=mode+'_'+clutter;z=[]
                for j,frame in enumerate(adc):
                    cube=env['_build_rda_cube_from_frame'](frame,chirps,samples,8,16,doppler_mode=mode,clutter_mode=clutter)
                    z.append(cube[4,8,12])
                    if j==0:
                        selected=frame[:,chirps];clean=selected-selected.mean(axis=1,keepdims=True) if clutter=='chirp_mean' else selected
                        rg=np.fft.fft(clean,axis=-1)[...,:125]
                        stages.append(dict(case=case,variant=variant,adc_target_rms=float(np.sqrt(np.mean(abs(selected)**2))),
                            post_clutter_rms=float(np.sqrt(np.mean(abs(clean)**2))),range_target_mean_abs=float(abs(rg[...,12]).mean()),rda_target_abs=float(abs(cube[4,8,12]))))
                z=np.array(z);arrays[case+'_'+variant]=z
                reference_amp=1000*250*4*(8 if mode=='legacy' else 64)
                row=dict(case=case,variant=variant,amplitude_fraction=float(np.median(abs(z))/reference_amp),
                    **phase_score(z,truth,reference_amp));rows.append(row)
    # Actual-time response, rather than assigning native Hz to nonuniform sampled FFT bins.
    f=np.linspace(-1/(4*pri),1/(4*pri),1001);arrays['physical_probe_hz']=f
    for mode,nfft in [('legacy',8),('full_fft_crop',64)]:
        count=nfft;idx=chirps[:count];k=np.arange(-4,4)
        kernels=np.exp(-2j*np.pi*k[:,None]*np.arange(count)[None,:]/nfft)
        response=kernels@np.exp(2j*np.pi*idx[:,None]*pri*f[None,:])/count
        arrays[mode+'_physical_response']=response
        arrays[mode+'_nominal_cycles_per_selected_index']=k/nfft
    np.savez_compressed(out/'synthetic_signals.npz',**arrays)
    pd.DataFrame(rows).to_csv(out/'synthetic_metrics.csv',index=False);pd.DataFrame(stages).to_csv(out/'synthetic_stages.csv',index=False)
    failed=[r['variant']+':'+r['case'] for r in rows if not r['valid_phase'] or (r['phase_rmse_rad'] is not None and r['phase_rmse_rad']>.01) or (r['peak_bpm'] is not None and abs(r['peak_bpm']-72)>3)]
    assert all(r['valid_phase'] and r['phase_rmse_rad']<.01 and abs(r['peak_bpm']-72)<3 for r in rows if r['variant'].endswith('_none'))
    result=dict(elapsed_sec=time.perf_counter()-started,training=False,raw_packing_roundtrip=True,
        gate_passed=not failed,failed_conditions=failed,real_hr_ablation_allowed=not failed,
        decision='STOP real HR ablation: audit identified signal-null/phase-distortion stage; do not train',
        note='packing roundtrip validates implementation against its own declared layout, not vendor layout independently')
    (out/'synthetic_gate.json').write_text(json.dumps(result,indent=2));print(pd.DataFrame(rows).to_string(index=False));return result
def provenance(input_root,code,out):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
    selected=pd.read_csv(code/'selected_records.csv');record=selected.iloc[0]
    tag=record.sample_tag;pid=int(record.group_key.rsplit('/',1)[-1])
    meta_path=next(p for p in input_root.rglob(tag+'.json') if p.parent.name=='meta' and 'ftu-rda-exports' in str(p))
    meta=json.loads(meta_path.read_text());scenario=meta['scenario'];distance=meta['distance'];repeat=meta['repeat']
    scenefolder={'Distance':'1. Distance Scenario','Orientation':'2. Orientation Scenario','Angle':'3. Angle Scenario','Elevated':'4. Elevated'}[scenario]
    raw_candidates=[p for p in input_root.rglob('data_Raw_0.bin') if f'P{pid}_' in str(p) and p.parent.name==str(repeat) and p.parent.parent.name==distance and scenefolder in str(p)]
    assert len(raw_candidates)==1,[str(p) for p in raw_candidates]
    raw_path=raw_candidates[0];oldpath=next(p for p in input_root.rglob(tag+'.npz') if 'ftu-rda-exports' in str(p))
    shutil.copyfile(meta_path,out/'historical_sidecar.json')
    for name in ['data_LogFile.txt','data_Raw_LogFile.csv']:
        p=raw_path.parent/name
        if p.exists():shutil.copyfile(p,out/name)
    current=load_conversion(code/'sources/current_export.py');archived=load_conversion(code/'sources/archived_export.py')
    loaders={name:load_unpack(code/'sources'/f'{name}_ftu_loader.py') for name in ['current','archived']}
    raw=np.memmap(raw_path,dtype='<i2',mode='r');assert len(raw)>=1200*4*128*250*2
    # Hash full raw input, and test all 1200 saved frames with the archived ROI.
    rawhash=digest(raw_path);oldhash=digest(oldpath)
    expected=json.loads((code/'expected_rda_hashes.json').read_text())[tag];assert oldhash==expected
    with np.load(oldpath) as z:saved=z['radar']
    bins=np.array(meta['range_selected_bins']);ci=current['_select_even_indices'](128,64);si=np.arange(250)
    differences=[];energy=np.zeros(125,dtype=np.float64);snapshots={}
    for start in range(0,1200,8):
        count=min(8,1200-start);words=4*128*250*2
        rawchunk=np.asarray(raw[start*words:(start+count)*words])
        adc=loaders['archived'](rawchunk,count);np.testing.assert_array_equal(adc,loaders['current'](rawchunk,count))
        for j,frame in enumerate(adc):
            a=archived['_build_rda_cube_from_frame'](frame,ci,si,8,16)
            b=current['_build_rda_cube_from_frame'](frame,ci,si,8,16,doppler_mode='legacy',clutter_mode='chirp_mean')
            np.testing.assert_allclose(a,b,rtol=2e-6,atol=1e-3)
            energy+=np.mean(abs(a),axis=(0,1));actual=a[:,:,bins].astype(np.complex64);target=saved[start+j]
            differences.append(dict(frame=start+j,max_abs_error=float(abs(actual-target).max()),
                relative_l2=float(np.linalg.norm(actual-target)/max(np.linalg.norm(target),1e-30))))
            if start+j in [0,255,1199]:snapshots[f'frame_{start+j}_reproduced']=actual;snapshots[f'frame_{start+j}_saved']=target
    pd.DataFrame(differences).to_csv(out/'reproduction_per_frame.csv',index=False)
    np.savez_compressed(out/'provenance_snapshots.npz',**snapshots,global_range_energy=energy)
    result=dict(sample_tag=tag,raw_file=str(raw_path),raw_sha256=rawhash,raw_bytes=raw_path.stat().st_size,
        historical_rda_sha256=oldhash,sidecar_sha256=digest(meta_path),frames=1200,loader_arrays_equal=True,
        recomputed_global_range_center=int(energy.argmax()),saved_global_range_center=int(meta['range_center_bin']),
        max_relative_l2=max(d['relative_l2'] for d in differences),max_abs_error=max(d['max_abs_error'] for d in differences),
        numerical_reproduction_passed=max(d['relative_l2'] for d in differences)<2e-5 and int(energy.argmax())==int(meta['range_center_bin']),
        elapsed_sec=time.perf_counter()-started,training=False,
        provenance_limit='numeric equivalence of archived candidate and existing export; original generating command/commit not recorded in sidecar')
    (out/'provenance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    assert result['numerical_reproduction_passed']
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--input',type=Path)
    args=p.parse_args()
    if args.input:provenance(args.input,args.code,args.out)
    else:synthetic(args.code,args.out)
