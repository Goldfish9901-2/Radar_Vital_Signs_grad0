"""CPU continuous-record target-selection audit, no fitting or HR-based selection."""
import hashlib,io,json,shutil,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.features.edacm import select_target_bin_indices,edacm_phase,detrend_linear,zscore_1d,target_edacm_signal
from src.features.representations import frequency_features
from src.data.ftu_timebase import frame_times,labels_at_frames


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def increments(z):
    z=np.asarray(z,dtype=np.complex64);i=z.real.astype(np.float32);q=z.imag.astype(np.float32)
    di=np.diff(i,axis=-1,prepend=i[...,:1]);dq=np.diff(q,axis=-1,prepend=q[...,:1])
    return (i*dq-q*di)/(i*i+q*q+np.float32(1e-6))


def process(window,bins,weights):
    z=np.stack([window[:,int(d),int(a),int(r)] for d,a,r in bins]).astype(np.complex64)
    delta=increments(z);cumulative=np.cumsum(delta,axis=-1,dtype=np.float32)
    phase=np.stack([detrend_linear(v) for v in cumulative]);fused=np.sum(phase*weights[:,None],axis=0)
    signal=zscore_1d(fused)
    return dict(z=z,amplitude=abs(z),delta=delta,cumulative=cumulative,phase=phase,
                fused_delta=np.sum(delta*weights[:,None],axis=0),fused=fused,signal=signal,
                x_freq=frequency_features(signal[None])[0][0])


def nrms(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    denom=np.sqrt((np.mean(a*a)+np.mean(b*b))/2)
    return float(np.sqrt(np.mean((a-b)**2))/denom) if denom>1e-15 else 0.


def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-10 and np.std(b)>1e-10 else np.nan


def spectral_distance(a,b):
    def power(x):
        p=abs(np.fft.rfft((x-x.mean())*np.hanning(len(x))))**2
        return p/p.sum() if p.sum()>1e-20 else np.zeros_like(p)
    return float(.5*np.sum(abs(power(a)-power(b))))


def run(root,out,code):
    started=time.perf_counter();out.mkdir()
    protocol=json.loads((code/'protocol.json').read_text());selected=pd.read_csv(code/'selected_records.csv')
    assert len(selected)==54 and (selected.split=='train').all()
    for name in ['protocol.json','selected_records.csv','source_manifest.json']:shutil.copyfile(code/name,out/name)
    paths={p.stem:p for p in root.rglob('*.npz') if 'ftu-rda-exports' in str(p)}
    assert set(selected.sample_tag)<=paths.keys()
    archive,=root.rglob('ftu_timebase_edacm_v2.zip')
    assert sha(archive)==json.loads((code/'source_manifest.json').read_text())['input_archive_sha256']
    source_hashes={};targetrows=[];readouts=[];overlaps=[];fusions=[];mapping=[];validation=[]
    (out/'signals').mkdir();(out/'figures').mkdir()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hz=np.fft.rfftfreq(256,1/20);band=(hz>=.5)&(hz<=2.75);band_ids=np.flatnonzero(band)
    with zipfile.ZipFile(archive) as old:
        prefix='training_exports_ftu_edacm_v2/'
        manifest=pd.read_csv(io.BytesIO(old.read(prefix+'manifest.csv')))
        for record in selected.itertuples():
            tag=record.sample_tag;path=paths[tag];source_hashes[tag]=sha(path)
            with np.load(path) as z:radar=z['radar'];rt=z['time'];hr=z['heart_rate']
            times=frame_times(len(radar),20,radar_start_on_reference_clock_s=0)
            reference=labels_at_frames(rt,hr,times)
            rows=manifest[manifest.sample_tag==tag].sort_values('window_start')
            assert len(rows)==8 and (rows.split=='train').all()
            starts=rows.window_start.to_numpy(dtype=int)
            fixed,weights,fixedmeta=select_target_bin_indices(radar[:256],3)
            all_bins=[];all_weights=[];signals=[];spectra=[];saved={};metadata=[];eligible=[]
            for arm in ['A','B']:
                arm_data=[];arm_bins=[];arm_weights=[]
                for j,row in enumerate(rows.itertuples()):
                    s=int(row.window_start);window=radar[s:s+256];assert len(window)==256
                    bins,w,meta=select_target_bin_indices(window,3) if arm=='A' else (fixed,weights,fixedmeta)
                    data=process(window,bins,w);arm_data.append(data);arm_bins.append(bins);arm_weights.append(w)
                    metadata.append(dict(arm=arm,window_start=s,selection=meta))
                    label=float(np.nanmean(reference[s:s+256]));coverage=float(np.isfinite(reference[s:s+256]).mean())
                    scored=s>=256 and coverage>=.8
                    if arm=='A':
                        eligible.append(scored)
                        name=prefix+'windows/'+row.window_path.split('/windows/',1)[1]
                        with np.load(io.BytesIO(old.read(name))) as v:
                            np.testing.assert_allclose(data['signal'],v['x_time'][0],atol=2e-5,rtol=2e-5)
                            np.testing.assert_allclose(data['x_freq'],v['x_freq'][0],atol=2e-5,rtol=2e-5)
                            np.testing.assert_allclose(label,v['label_heart_rate'],atol=2e-5)
                    for rank,((d,a,r),weight) in enumerate(zip(bins,w)):
                        targetrows.append(dict(sample_tag=tag,subject=record.group_key,condition=record.condition,arm=arm,
                            window_start=s,scored=scored,rank=rank,doppler=int(d),angle=int(a),range=int(r),weight=float(weight)))
                    for kind,phase in [('fused',data['signal'])]+[(f'single{k}',data['phase'][k]) for k in range(3)]:
                        spectrum=abs(np.fft.rfft(phase*np.hanning(256)))**2
                        peak=hz[band_ids[spectrum[band].argmax()]]*60
                        readouts.append(dict(sample_tag=tag,subject=record.group_key,condition=record.condition,arm=arm,
                            window_start=s,time_s=(s+128)/20,scored=scored,kind=kind,label_bpm=label,peak_bpm=peak,
                            hr_band_fraction=float(spectrum[band].sum()/(spectrum[1:].sum()+1e-20)),
                            reference_near_power_fraction=float(spectrum[abs(hz-label/60)<=20/256].sum()/(spectrum[band].sum()+1e-20))))
                    sp=np.fft.rfft(data['phase']*np.hanning(256),axis=1)*w[:,None]
                    incoherent=np.sum(abs(sp[:,band])**2);combined=np.sum(abs(sp[:,band].sum(0))**2)
                    fusions.append(dict(sample_tag=tag,subject=record.group_key,arm=arm,window_start=s,scored=scored,
                        fused_band_power=float(combined),weighted_single_band_power=float(incoherent),
                        fusion_power_ratio=float(combined/incoherent) if incoherent>1e-20 else np.nan,
                        fused_pre_z_std=float(data['fused'].std()),amplitude_min=float(data['amplitude'].min()),
                        amplitude_median=float(np.median(data['amplitude'])),max_abs_raw_delta=float(abs(data['delta']).max())))
                    if j:
                        k=s-starts[j-1];prev=arm_data[j-1];oldbins=arm_bins[j-1];oldw=arm_weights[j-1]
                        if 0<k<256:
                            left=prev['fused_delta'][k+1:];right=data['fused_delta'][1:256-k]
                            dl=np.diff(prev['fused'])[k:];dr=np.diff(data['fused'])[:255-k]
                            pl=prev['fused'][k+1:];pr=data['fused'][1:256-k]
                            oldmap={tuple(b):float(v) for b,v in zip(oldbins,oldw)};newmap={tuple(b):float(v) for b,v in zip(bins,w)}
                            union=set(oldmap)|set(newmap);inter=set(oldmap)&set(newmap)
                            overlaps.append(dict(sample_tag=tag,subject=record.group_key,arm=arm,left_start=int(starts[j-1]),right_start=s,
                                scored=bool(scored and starts[j-1]>=256),shared_targets=len(inter),set_turnover=1-len(inter)/3,
                                weight_l1=sum(abs(oldmap.get(b,0)-newmap.get(b,0)) for b in union),
                                raw_delta_nrms=nrms(left,right),raw_delta_r=corr(left,right),
                                detrended_delta_nrms=nrms(dl,dr),demeaned_delta_nrms=nrms(dl-dl.mean(),dr-dr.mean()),
                                demeaned_phase_nrms=nrms(pl-pl.mean(),pr-pr.mean()),overlap_spectrum_tv=spectral_distance(pl,pr)))
                for key in arm_data[0]:saved[arm+'_'+key]=np.stack([d[key] for d in arm_data])
                saved[arm+'_bins']=np.array(arm_bins);saved[arm+'_weights']=np.array(arm_weights)
                all_bins.extend(np.array(arm_bins).reshape(-1,3).tolist())
            union=np.unique(np.asarray(all_bins),axis=0)
            continuous=np.stack([radar[:,int(d),int(a),int(r)] for d,a,r in union]).astype(np.complex64)
            saved.update(continuous_bins=union,continuous_complex=continuous,continuous_amplitude=abs(continuous),
                         continuous_delta=increments(continuous),radar_time_s=times,reference_time=rt,reference_hr=hr,
                         mapped_reference_hr=reference,starts=starts,scored=np.array(eligible))
            np.savez_compressed(out/'signals'/f'{tag}.npz',**saved)
            (out/'signals'/f'{tag}_selection.json').write_text(json.dumps(metadata,indent=2))
            mapping.append(dict(sample_tag=tag,subject=record.group_key,frames=len(radar),radar_shape=str(radar.shape),
                reference_start=float(rt[0]),reference_end=float(rt[-1]),scored_windows=sum(eligible),raw_sha256=source_hashes[tag]))
            # Show every selected record; no post-hoc ranking or preferred-example selection.
            fig,axes=plt.subplots(2,3,figsize=(15,7))
            rr=pd.DataFrame(readouts);rr=rr[(rr.sample_tag==tag)&(rr.kind=='fused')]
            for rowid,arm in enumerate(['A','B']):
                g=rr[rr.arm==arm];spec=abs(np.fft.rfft(saved[arm+'_signal']*np.hanning(256),axis=-1))**2
                ax=axes[rowid,0];ax.pcolormesh(g.time_s,hz[band]*60,10*np.log10(spec[:,band].T+1e-12),shading='auto')
                ax.plot(g.time_s,g.label_bpm,'w.-',label='HR (assumed time)');ax.plot(g.time_s,g.peak_bpm,'r.-',label='fixed-rule peak')
                ax.axvspan(0,12.8,color='gray',alpha=.3);ax.set(title=arm+' window spectra',ylabel='bpm',ylim=(30,165));ax.legend(fontsize=6)
                ax=axes[rowid,1]
                for rank in range(3):
                    identity=np.ravel_multi_index(saved[arm+'_bins'][:,rank].T,radar.shape[1:])
                    ax.plot((starts+128)/20,identity,'.-',label=f'target rank{rank}')
                ax.set(title=arm+' target identities (D/A/R flattened)',ylabel='Exported RDA flat index');ax.legend(fontsize=6)
                ax=axes[rowid,2]
                for rank in range(3):ax.plot((starts+128)/20,saved[arm+'_weights'][:,rank],'.-',label=f'rank{rank}')
                ax.set(title=arm+' fusion weights',ylabel='Weight',ylim=(0,1));ax.legend(fontsize=6)
            for ax in axes.flat:ax.set_xlabel('Window center (s)')
            fig.suptitle(tag);fig.tight_layout();fig.savefig(out/'figures'/f'{tag}.png',dpi=110);plt.close(fig)
            validation.append(tag);print(tag,'verified original A; scored',sum(eligible),flush=True)
    for name,data in [('target_tracks',targetrows),('readouts',readouts),('overlap',overlaps),('fusion',fusions),('time_mapping',mapping)]:
        pd.DataFrame(data).to_csv(out/f'{name}.csv',index=False)
    (out/'source_hashes.json').write_text(json.dumps(source_hashes,indent=2))
    (out/'execution.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-started,records=len(validation),
        A_matches_original_export=True,source_archive_sha256=sha(archive),training=False,offset_search=False),indent=2))
    (out/'artifact_manifest.json').write_text(json.dumps({p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()},indent=2))
    shutil.make_archive('/kaggle/working/000_ftu_target_audit','zip',out)
