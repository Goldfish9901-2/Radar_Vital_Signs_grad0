"""Frozen six-record ADC 2x2. Vendor-derived independent reader; no learning."""
import ast,json,re,time,hashlib,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from audit_ftu_adc_phase import load_conversion,load_unpack,digest,pack

VARIANTS={'legacy_mean':('legacy','chirp_mean'),'full_mean':('full_fft_crop','chirp_mean'),
          'legacy_none':('legacy','none'),'full_none':('full_fft_crop','none')}
def vendor_reader(raw,frames):
    """Independent port of TI SWRA581B §9.2: pair stream -> chirp columns -> RX.

    This does not import the production unpacker or use synthetic packing.
    Assumes signed16 header-free ordered payload,4RX,250samples,128chirps/frame.
    """
    words=np.asarray(raw,dtype=np.float64).ravel();lvds=np.empty(words.size//2,dtype=np.complex128)
    lvds[0::2]=words[0::4]+1j*words[2::4];lvds[1::2]=words[1::4]+1j*words[3::4]
    # MATLAB reshape(...,samples*RX,chirps).' then per-RX blocks.
    chirp_rows=lvds.reshape(1000,-1,order='F').T
    rx=np.stack([chirp_rows[:,r*250:(r+1)*250].ravel() for r in range(4)])
    return rx.reshape(4,frames,128,250).transpose(1,0,2,3).astype(np.complex64)
def p2(z):
    zz=np.asarray(z,dtype=np.complex128)
    d=np.concatenate([np.zeros_like(zz[...,:1].real),np.angle(zz[...,1:]*zz[...,:-1].conj())],axis=-1).astype(np.float32)
    valid=np.isfinite(zz)&(abs(zz)>0)
    d[...,1:]=np.where(valid[...,1:]&valid[...,:-1],d[...,1:],np.nan)
    d[...,0]=np.where(valid[...,0],0,np.nan)
    return d
def load_post(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'));env=dict(np=np)
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['detrend_linear','zscore_1d']],type_ignores=[]),str(path),'exec'),env)
    return env['detrend_linear'],env['zscore_1d']
def process(z,weights,post,valid):
    delta=p2(z);cum=np.cumsum(delta,axis=-1,dtype=np.float32)
    phase=np.full_like(cum,np.nan)
    for k,v in enumerate(cum):
        if valid[k] and np.isfinite(v).all():phase[k]=post[0](v)
    fused=np.sum(phase*weights[:,None],axis=0)
    signal=post[1](fused) if np.isfinite(fused).all() else np.full(256,np.nan,dtype=np.float32)
    return delta,cum,phase,fused,signal
def readout(trace,label):
    n=len(trace);hz=np.fft.rfftfreq(n,.05);band=(hz>=.5)&(hz<=2.75)
    power=abs(np.fft.rfft(trace*np.hanning(n)))**2
    return dict(peak_bpm=float(hz[band][power[band].argmax()]*60),
        reference_near_power_fraction=float(power[abs(hz-label/60)<=20/n].sum()/(power[band].sum()+1e-20)))
def synthetic(code,out):
    started=time.perf_counter();out.mkdir(exist_ok=True,parents=True)
    cfg=json.loads((code/'protocol.json').read_text());ci=np.array(cfg['chirp_indices']);pri=cfg['pri_s']
    prod=load_conversion(code/'sources/current_export.py');unpack=load_unpack(code/'sources/current_ftu_loader.py')
    rows=[];arrays={};normrows=[]
    for case in cfg['synthetic_cases']:
        name=case['id'];hz=case['hz'];amp=case['phase_amplitude'];initial=case['initial_phase'];n=400;t=np.arange(n)/20
        truth=amp*np.sin(2*np.pi*hz*t+initial)
        z_by={v:[] for v in VARIANTS};direct64={v:[] for v in VARIANTS};direct128={v:[] for v in VARIANTS}
        for timepoint in t:
            phase=amp*np.sin(2*np.pi*hz*(timepoint+np.arange(128)*pri)+initial)
            adc=(1000*np.exp(1j*phase)[None,None,:,None]*np.exp(2j*np.pi*12*np.arange(250)/250)[None,None,None,:]*np.ones((1,4,1,1))).astype(np.complex64)
            raw=pack(adc);decoded=vendor_reader(raw,1);np.testing.assert_array_equal(decoded,unpack(raw,1));frame=decoded[0]
            for variant,(mode,clutter) in VARIANTS.items():
                cube=prod['_build_rda_cube_from_frame'](frame,ci,np.arange(250),8,16,doppler_mode=mode,clutter_mode=clutter)
                z_by[variant].append(cube[4,8,12])
                for dtype,store in [(np.complex64,direct64),(np.complex128,direct128)]:
                    x=frame[:,ci].astype(dtype)
                    if clutter=='chirp_mean':x=x-x.mean(axis=1,keepdims=True,dtype=dtype)
                    rg=np.fft.fft(x,axis=-1)
                    count=8 if mode=='legacy' else 64
                    store[variant].append(rg[:,:count,12].sum(dtype=dtype))
        for variant,(mode,clutter) in VARIANTS.items():
            z=np.array(z_by[variant]);count=8 if mode=='legacy' else 64;gain=1000*250*4*count
            tau=float(ci[:count].mean()*pri)
            theoretical_phase=amp*np.sin(2*np.pi*hz*(t+tau)+initial)
            coherent_phase=np.angle(np.exp(1j*amp*np.sin(2*np.pi*hz*(t[:,None]+ci[None,:count]*pri)+initial)).mean(axis=1))
            valid=bool(np.all(abs(z)>gain*1e-6));recovered=np.r_[0,np.cumsum(np.angle(z[1:].astype(complex)*z[:-1].conj()))]
            centered=lambda x:x-x[0]
            exact=p2(np.exp(1j*coherent_phase)[None])[0]
            row=dict(case=name,variant=variant,role='candidate' if clutter=='none' else 'mechanism_control',frequency_hz=hz,phase_amplitude=amp,
                mean_chirp_time_ms=tau*1000,valid_phase=valid,median_amplitude_fraction=float(np.median(abs(z))/gain),
                rmse_chirp0_rad=float(np.sqrt(np.mean((recovered-centered(truth))**2))) if valid else np.nan,
                rmse_mean_time_rad=float(np.sqrt(np.mean((recovered-centered(theoretical_phase))**2))) if valid else np.nan,
                rmse_coherent_phase_rad=float(np.sqrt(np.mean((recovered-centered(coherent_phase))**2))) if valid else np.nan,
                predicted_chirp0_time_error_rad=float(np.sqrt(np.mean((centered(theoretical_phase)-centered(truth))**2))),
                peak_bpm=readout(recovered,hz*60)['peak_bpm'] if valid else np.nan)
            row['pass']=bool(valid and row['rmse_mean_time_rad']<=.001 and abs(row['peak_bpm']-hz*60)<=3)
            rows.append(row);arrays[name+'_'+variant]=z;arrays[name+'_truth']=truth;arrays[name+'_'+mode+'_coherent_phase']=coherent_phase
            for precision,values in [('complex64',direct64[variant]),('complex128',direct128[variant])]:
                ref=np.array(values);diff=z-ref;refnorm=float(np.linalg.norm(ref));diffnorm=float(np.linalg.norm(diff))
                normrows.append(dict(case=name,variant=variant,reference_precision=precision,
                    max_absolute_difference=float(abs(diff).max()),difference_l2=diffnorm,reference_l2=refnorm,
                    relative_l2=diffnorm/refnorm if refnorm>gain*np.sqrt(n)*1e-6 else np.nan,
                    difference_rms_over_coherent_gain=diffnorm/np.sqrt(n)/gain,
                    reference_rms_over_coherent_gain=refnorm/np.sqrt(n)/gain))
    results=pd.DataFrame(rows);results.to_csv(out/'synthetic_metrics.csv',index=False);pd.DataFrame(normrows).to_csv(out/'precision_by_condition.csv',index=False)
    np.savez_compressed(out/'synthetic_signals.npz',**arrays)
    passed=bool(results[results.role=='candidate']['pass'].all())
    gate=dict(elapsed_sec=time.perf_counter()-started,candidates_pass=passed,controls_required_to_pass=False,
        independent_vendor_reader_matches_production=True,real_ablation_allowed=passed,training=False)
    (out/'synthetic_gate.json').write_text(json.dumps(gate,indent=2));print(results.to_string(index=False));assert passed
def real(input_root,code,out,supplement_root=None):
    started=time.perf_counter();out.mkdir(exist_ok=True,parents=True);(out/'signals').mkdir(exist_ok=True);(out/'evidence').mkdir(exist_ok=True)
    gate=json.loads((code/'synthetic_gate.json').read_text());assert gate['real_ablation_allowed']
    selected=pd.read_csv(code/'selected_records.csv');prod=load_conversion(code/'sources/current_export.py');unpack=load_unpack(code/'sources/current_ftu_loader.py');post=load_post(code/'sources/edacm.py')
    ci=np.array(json.loads((code/'protocol.json').read_text())['chirp_indices']);words=4*128*250*2
    readouts=[];inputrows=[];selection=[];quality=[];hashes={}
    for rec in selected.itertuples():
        record_start=time.perf_counter();tag=rec.sample_tag;pid=int(rec.group_key.rsplit('/',1)[-1])
        meta_path=next(p for p in input_root.rglob(tag+'.json') if p.parent.name=='meta' and 'ftu-rda-exports' in str(p));meta=json.loads(meta_path.read_text())
        scenario={'Distance':'1. Distance Scenario','Orientation':'2. Orientation Scenario','Angle':'3. Angle Scenario'}[meta['scenario']]
        files=[p for p in input_root.rglob('data_Raw_0.bin') if f'P{pid}_' in str(p) and p.parent.name==str(meta['repeat']) and p.parent.parent.name==meta['distance'] and scenario in str(p)]
        if not files and supplement_root is not None:
            files=[p for p in supplement_root.rglob('data_Raw_0.bin') if f'P{pid}_' in str(p) and p.parent.name==str(meta['repeat']) and p.parent.parent.name==meta['distance'] and scenario in str(p)]
        assert len(files)==1,[str(p) for p in files];path=files[0];raw=np.memmap(path,dtype='<i2',mode='r');assert raw.size==1200*words
        oldpath=next(p for p in input_root.rglob(tag+'.npz') if 'ftu-rda-exports' in str(p))
        hashes[tag]=dict(raw_sha256=digest(path),rda_sha256=digest(oldpath),raw_bytes=path.stat().st_size)
        assert hashes[tag]['rda_sha256']==json.loads((code/'expected_rda_hashes.json').read_text())[tag]
        logs={}
        for name in ['data_LogFile.txt','data_Raw_LogFile.csv']:
            p=path.parent/name
            if p.exists():
                text=p.read_text(errors='replace');(out/'evidence'/f'{tag}_{name}').write_bytes(p.read_bytes());logs[name]=text
        rawlog=logs.get('data_Raw_LogFile.csv','');radarlog=logs.get('data_LogFile.txt','')
        full=np.empty((1200,16,125),dtype=np.complex64)
        # Build only common physical Doppler DC; preserve all beam/range to freeze calibration.
        for start in range(0,1200,8):
            n=min(8,1200-start);block=np.asarray(raw[start*words:(start+n)*words]);adc=vendor_reader(block,n)
            np.testing.assert_array_equal(adc,unpack(block,n))
            for j,frame in enumerate(adc):
                full[start+j]=prod['_build_rda_cube_from_frame'](frame,ci,np.arange(250),8,16,doppler_mode='full_fft_crop',clutter_mode='none')[4]
        energy=np.mean(abs(full[:256]),axis=0);rangeprofile=energy.mean(axis=0);rangeprofile[:2]=-np.inf
        center=int(rangeprofile.argmax());roi=prod['_range_window_indices'](125,8,center)
        score=energy[:,roi].copy();score[:,roi<2]=-np.inf
        flat=np.argsort(-score.ravel(),kind='stable')[:3];a,r=np.unravel_index(flat,score.shape);r=roi[r]
        targets=list(zip(a.tolist(),r.tolist()));weights=np.array([energy[aa,rr] for aa,rr in targets],dtype=np.float32);weights/=weights.sum()
        selection.append(dict(sample_tag=tag,subject=rec.group_key,range_center=center,roi=roi.tolist(),targets_angle_range=targets,weights=weights.tolist()))
        with np.load(oldpath) as old:rt=old['time'].astype(float);hr=old['heart_rate'].astype(float)
        times=np.arange(1200)/20;reference=np.interp(times,rt,hr,left=np.nan,right=np.nan)
        saved=dict(starts=np.arange(0,1024,128),weights=weights,targets=np.array(targets),roi=roi,
            reference_time=rt,reference_hr=hr,mapped_reference_hr=reference,radar_time_s=times,
            calibration_energy=energy,full_none_DC_roi=full[:,:,roi])
        for variant,(mode,clutter) in VARIANTS.items():
            dc=np.empty((1200,16,len(roi)),dtype=np.complex64)
            if variant=='full_none':dc[:]=full[:,:,roi]
            else:
                for start in range(0,1200,8):
                    n=min(8,1200-start);adc=vendor_reader(np.asarray(raw[start*words:(start+n)*words]),n)
                    for j,frame in enumerate(adc):dc[start+j]=prod['_build_rda_cube_from_frame'](frame,ci,np.arange(250),8,16,doppler_mode=mode,clutter_mode=clutter)[4][:,roi]
            z=np.stack([dc[:,aa,np.flatnonzero(roi==rr)[0]] for aa,rr in targets]);saved[variant+'_z']=z;saved[variant+'_DC_roi']=dc
            processed={k:[] for k in ['delta','cumulative','phase','fused','signal']}
            ratio=(8 if mode=='legacy' else 64)/64
            for s in saved['starts']:
                s=int(s);seg=z[:,s:s+256];labels=reference[s:s+256];label=float(np.nanmean(labels));scored=s>=256 and np.isfinite(labels).mean()>=.8
                baseline=np.stack([full[s:s+256,aa,rr] for aa,rr in targets])*ratio
                baseline_rms=np.sqrt(np.mean(abs(baseline.astype(complex))**2,axis=1));actual_rms=np.sqrt(np.mean(abs(seg.astype(complex))**2,axis=1))
                valid=np.isfinite(seg).all(axis=1)&(abs(seg)>0).all(axis=1)&(actual_rms>baseline_rms*1e-6)
                # Controls below numerical amplitude floor are marked undefined; never assigned fake phase/HR.
                delta,cumulative,phase,fused,signal=process(seg,weights,post,valid)
                for key,val in zip(processed,[delta,cumulative,phase,fused,signal]):processed[key].append(val)
                for kind,trace,ok in [('fused',signal,bool(valid.all()))]+[(f'single{k}',phase[k],bool(valid[k])) for k in range(3)]:
                    scores=readout(trace,label) if ok else dict(peak_bpm=np.nan,reference_near_power_fraction=np.nan)
                    readouts.append(dict(sample_tag=tag,subject=rec.group_key,variant=variant,kind=kind,window_start=s,
                        scored=scored,valid_phase=ok,label_bpm=label,**scores))
                quality.append(dict(sample_tag=tag,subject=rec.group_key,variant=variant,window_start=s,scored=scored,
                    valid_targets=int(valid.sum()),amplitude_fraction_min=float(np.min(actual_rms/np.maximum(baseline_rms,1e-30))),
                    amplitude_fraction_median=float(np.median(actual_rms/np.maximum(baseline_rms,1e-30))),max_abs_delta=float(abs(delta[:,1:]).max()),
                    fused_pre_z_std=float(fused.std())))
            for key,vals in processed.items():saved[variant+'_'+key]=np.array(vals)
        np.savez_compressed(out/'signals'/f'{tag}.npz',**saved)
        inputrows.append(dict(sample_tag=tag,raw_path=str(path),vendor_equals_production=True,
            radar_log_available=bool(radarlog),device_AR1642='AR1642' in radarlog,capture_DCA1000='DCA1000' in radarlog,
            raw_two_lane='2 lane' in rawlog,zero_filled_packets=re.findall(r'Number of zero filled packets\s*-\s*(\d+)',rawlog),
            out_of_sequence=re.findall(r'Out of sequence count\s*-\s*(\d+)',rawlog),elapsed_sec=time.perf_counter()-record_start,**hashes[tag]))
        print(tag,'complete',flush=True)
    pd.DataFrame(readouts).to_csv(out/'readouts.csv',index=False);pd.DataFrame(quality).to_csv(out/'signal_quality.csv',index=False)
    if supplement_root is not None and (supplement_root/'recovery.json').exists():shutil.copyfile(supplement_root/'recovery.json',out/'official_recovery_execution.json')
    pd.DataFrame(inputrows).to_csv(out/'raw_verification.csv',index=False);(out/'selection.json').write_text(json.dumps(selection,indent=2))
    (out/'input_hashes.json').write_text(json.dumps(hashes,indent=2));(out/'execution.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-started,
        records=6,training=False,delay_search=False,candidate_gate_passed=True,vendor_raw_match=True,scored_windows_per_arm=36),indent=2))
    shutil.make_archive('/kaggle/working/000_ftu_adc_controlled','zip',out)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--input',type=Path)
    args=p.parse_args()
    if args.input:real(args.input,args.code,args.out)
    else:synthetic(args.code,args.out)
