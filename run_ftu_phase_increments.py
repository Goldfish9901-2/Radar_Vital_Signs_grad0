"""Frozen-target, local CPU phase-increment diagnostic. No selection or training."""
import hashlib,json,time,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.features.edacm import detrend_linear,zscore_1d
from src.features.representations import frequency_features

ROOT=Path('experiment_runs/ftu_phase_increments')
SOURCE=Path('experiment_runs/ftu_target_audit')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def increments(z,method):
    z=np.asarray(z,dtype=np.complex64)
    valid=np.isfinite(z)&(abs(z)>0)
    pairs=valid[...,1:]&valid[...,:-1]
    if method=='P0':
        i=z.real;q=z.imag
        d=(i*np.diff(q,prepend=q[...,:1],axis=-1)-q*np.diff(i,prepend=i[...,:1],axis=-1))/(i*i+q*q+np.float32(1e-6))
    else:
        # Promote product to avoid complex64 overflow; downstream stays float32.
        zz=z.astype(np.complex128)
        angle=np.angle(zz[...,1:]*zz[...,:-1].conj())
        d=np.concatenate([np.zeros_like(angle[...,:1]),np.sin(angle) if method=='P1' else angle],axis=-1).astype(np.float32)
    d[...,1:]=np.where(pairs,d[...,1:],np.nan)
    d[...,0]=np.where(valid[...,0],0,np.nan)
    return d
def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-10 and np.std(b)>1e-10 else np.nan
def metrics(g):
    y=g.label_bpm.to_numpy();p=g.peak_bpm.to_numpy()
    cy=y-g.groupby('sample_tag').label_bpm.transform('mean').to_numpy()
    cp=p-g.groupby('sample_tag').peak_bpm.transform('mean').to_numpy()
    return dict(n=len(g),mae=float(abs(p-y).mean()),rmse=float(np.sqrt(np.mean((p-y)**2))),r=corr(y,p),
        centered_rmse=float(np.sqrt(np.mean((cp-cy)**2))),centered_r2=float(1-np.sum((cp-cy)**2)/np.sum(cy*cy)),
        centered_r=corr(cy,cp),centered_std_ratio=float(cp.std()/cy.std()),near_reference_power_fraction=float(g.reference_near_power_fraction.mean()))
def main():
    start=time.perf_counter();ROOT.mkdir(exist_ok=True,parents=True)
    for d in ['signals','figures','sources']: (ROOT/d).mkdir(exist_ok=True)
    protocol=dict(records=54,source=str(SOURCE),methods=['P0','P1','P2'],arms=['A','B'],training=False,
        selection='reuse saved identities and weights; never call selector',scoring='original scored mask AND all targets valid in both arms',
        invalid='zero/nonfinite complex samples invalidate adjacent increments; NaN retained; entire window excluded in all six conditions',
        boundary='first increment zero by window convention; excluded from increment diagnostics',
        precision='P0 original float32; P1/P2 complex128 product of saved complex64 then float32 increments; cumulative float32',
        downstream='original cumulative/detrend/fusion/zscore/Hann power; fixed .5-2.75Hz argmax; no interpolation',
        time='original frame/20 + 0s assumption; no offset search',examples='lowest selection hash per subject',
        decision='only numerical improvement AND consistent reference correspondence warrant frontend revision; otherwise upstream audit, no training')
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    manifest=json.loads((SOURCE/'artifact_manifest.json').read_text());inputs={}
    selected=pd.read_csv(SOURCE/'selected_records.csv');old=pd.read_csv(SOURCE/'readouts.csv')
    assert len(selected)==54
    for name in ['selected_records.csv','readouts.csv']+[f'signals/{t}.npz' for t in selected.sample_tag]:
        inputs[name]=sha(SOURCE/name);assert inputs[name]==manifest[name],name
    (ROOT/'input_hashes.json').write_text(json.dumps(inputs,indent=2));shutil.copyfile(SOURCE/'selected_records.csv',ROOT/'selected_records.csv')
    # Analytic control includes large identifiable increments, to expose sine approximation.
    t=np.arange(256);truth=.35+.6*np.sin(t*.07);truth[0]=0;phase=np.cumsum(truth)
    envelopes={'constant':np.ones(256)*10,'slow':10*np.exp(.7*np.sin(t*.03)),
        'drop':np.where(t<128,10.,.01),'zero':np.where(t==128,0.,10.)}
    controls=[];controlarrays={'true_delta':truth}
    for name,amp in envelopes.items():
        z=amp*np.exp(1j*phase);controlarrays[name+'_z']=z
        for method in protocol['methods']:
            delta=increments(z,method);controlarrays[name+'_'+method]=delta
            finite=np.isfinite(delta[1:]);target=np.sin(truth[1:]) if method=='P1' else truth[1:]
            error=float(np.max(abs(delta[1:][finite]-target[finite])))
            controls.append(dict(envelope=name,method=method,undefined=int((~finite).sum()),max_error_vs_phase=float(np.max(abs(delta[1:][finite]-truth[1:][finite]))),max_error_vs_expected=error))
            if method in ['P1','P2']: assert error<2e-6
            if name=='zero': assert np.flatnonzero(~np.isfinite(delta)).tolist()==[128,129]
    pd.DataFrame(controls).to_csv(ROOT/'positive_control.csv',index=False);np.savez_compressed(ROOT/'positive_control.npz',**controlarrays)
    rows=[];quality=[];validrows=[];checks=[];hz=np.fft.rfftfreq(256,1/20);band=(hz>=.5)&(hz<=2.75);ids=np.flatnonzero(band)
    examples=set(selected.sort_values('selection_hash').groupby('group_key').first().sample_tag)
    for rec in selected.itertuples():
        tag=rec.sample_tag
        with np.load(SOURCE/'signals'/f'{tag}.npz') as data:
            saved={};common=data['scored'].copy()
            for arm in ['A','B']:
                zz=data[arm+'_z'];common&=np.all(np.isfinite(zz)&(abs(zz)>0),axis=(1,2))
                saved[arm+'_bins']=data[arm+'_bins'];saved[arm+'_weights']=data[arm+'_weights']
            saved['starts']=data['starts'];saved['scored']=common
            for j,s in enumerate(data['starts']):validrows.append(dict(sample_tag=tag,window_start=int(s),original_scored=bool(data['scored'][j]),common_valid=bool(common[j])))
            for arm in ['A','B']:
                zz=data[arm+'_z'];w=data[arm+'_weights']
                for method in protocol['methods']:
                    delta=increments(zz,method);cum=np.cumsum(delta,axis=-1,dtype=np.float32)
                    phase=np.full_like(cum,np.nan);fused=np.full((8,256),np.nan,dtype=np.float32);signal=fused.copy();freq=np.full((8,129),np.nan,dtype=np.float32)
                    for j,s in enumerate(data['starts']):
                        if not np.isfinite(cum[j]).all():continue
                        phase[j]=np.stack([detrend_linear(v) for v in cum[j]])
                        fused[j]=np.sum(phase[j]*w[j,:,None],axis=0);signal[j]=zscore_1d(fused[j]);freq[j]=frequency_features(signal[j,None])[0][0]
                        label=float(old[(old.sample_tag==tag)&(old.arm==arm)&(old.kind=='fused')&(old.window_start==s)].label_bpm.iloc[0])
                        for kind,trace in [('fused',signal[j])]+[(f'single{k}',phase[j,k]) for k in range(3)]:
                            power=abs(np.fft.rfft(trace*np.hanning(256)))**2;peak=hz[ids[power[band].argmax()]]*60
                            rows.append(dict(sample_tag=tag,subject=rec.group_key,arm=arm,method=method,kind=kind,window_start=int(s),scored=bool(common[j]),label_bpm=label,peak_bpm=peak,
                                reference_near_power_fraction=float(power[abs(hz-label/60)<=20/256].sum()/(power[band].sum()+1e-20))))
                        if common[j]:
                            for k in range(3):
                                a=abs(zz[j,k].astype(np.complex128));ratio=a[:-1]/a[1:];angle=np.angle(zz[j,k,1:].astype(np.complex128)*zz[j,k,:-1].conj())
                                d=delta[j,k,1:].astype(float);pred=ratio*np.sin(angle)/(1+1e-6/a[1:]**2)
                                quality.append(dict(sample_tag=tag,subject=rec.group_key,arm=arm,method=method,window_start=int(s),rank=k,
                                    abs_delta_max=float(abs(d).max()),abs_delta_p99=float(np.quantile(abs(d),.99)),amplitude_ratio_max=float(ratio.max()),
                                    delta_minus_sine_ratio_r=corr(d-np.sin(angle),ratio),p0_identity_max_error=float(abs(d-pred).max()) if method=='P0' else np.nan,
                                    cumulative_endpoint=float(cum[j,k,-1]),cumulative_span=float(np.ptp(cum[j,k])),cumulative_slope=float(np.polyfit(np.arange(256)/20,cum[j,k],1)[0])))
                    for key,val in dict(delta=delta,cumulative=cum,phase=phase,fused=fused,signal=signal,x_freq=freq).items():
                        saved[f'{arm}_{method}_{key}']=val
                        if method=='P0':
                            error=float(np.nanmax(abs(val-data[arm+'_'+key])))
                            checks.append(dict(sample_tag=tag,arm=arm,array=key,max_abs_error=error))
                            np.testing.assert_allclose(val,data[arm+'_'+key],rtol=2e-5,atol=2e-5)
            np.savez_compressed(ROOT/'signals'/f'{tag}.npz',**saved)
        if tag in examples:
            rr=pd.DataFrame(rows);rr=rr[(rr.sample_tag==tag)&rr.scored]
            fig,axes=plt.subplots(2,4,figsize=(15,6),sharex=True)
            for i,arm in enumerate(['A','B']):
                for k,kind in enumerate(['single0','single1','single2','fused']):
                    ax=axes[i,k]
                    for method in protocol['methods']:
                        g=rr[(rr.arm==arm)&(rr.kind==kind)&(rr.method==method)];ax.plot((g.window_start+128)/20,g.peak_bpm,'.-',label=method)
                    ax.plot((g.window_start+128)/20,g.label_bpm,'k--',label='reference');ax.set(title=arm+' '+kind,ylabel='bpm',xlabel='Window center (s)');ax.legend(fontsize=6)
            fig.suptitle(tag+'; fixed readout, assumed reference time');fig.tight_layout();fig.savefig(ROOT/'figures'/f'{tag}.png',dpi=110);plt.close(fig)
    ro=pd.DataFrame(rows);ro.to_csv(ROOT/'readouts.csv',index=False);pd.DataFrame(quality).to_csv(ROOT/'increment_quality.csv',index=False);pd.DataFrame(validrows).to_csv(ROOT/'validity.csv',index=False)
    p0=ro[ro.method=='P0'].merge(old,on=['sample_tag','arm','kind','window_start'],suffixes=('','_old'))
    np.testing.assert_array_equal(p0.peak_bpm,p0.peak_bpm_old);np.testing.assert_array_equal(p0.label_bpm,p0.label_bpm_old)
    stats=[];records=[]
    for (arm,method,kind),g in ro[ro.scored].groupby(['arm','method','kind']):
        base=dict(arm=arm,method=method,kind=kind)
        stats.append(dict(**base,subject='pooled',**metrics(g)))
        for subject,h in g.groupby('subject'):stats.append(dict(**base,subject=subject,**metrics(h)))
        for tag,h in g.groupby('sample_tag'):records.append(dict(**base,sample_tag=tag,subject=h.subject.iloc[0],**metrics(h)))
    table=pd.DataFrame(stats);cols=[c for c in table if c not in ['arm','method','kind','subject','n']]
    macro=table[table.subject!='pooled'].groupby(['arm','method','kind'])[cols].mean().reset_index();macro['subject']='equal_subject_mean'
    table=pd.concat([table,macro]);table.to_csv(ROOT/'metrics.csv',index=False);pd.DataFrame(records).to_csv(ROOT/'record_metrics.csv',index=False)
    for path in [Path(__file__),Path('src/features/edacm.py'),Path('src/features/representations.py')]:shutil.copyfile(path,ROOT/'sources'/path.name)
    pd.DataFrame(checks).to_csv(ROOT/'p0_reproduction.csv',index=False)
    (ROOT/'execution.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-start,records=54,positive_control_passed=True,p0_array_tolerance=2e-5,p0_readouts_exact=True,input_hashes_verified=len(inputs),training=False,common_scored_windows=int(pd.DataFrame(validrows).common_valid.sum())),indent=2))
    print(table[(table.kind=='fused')&(table.subject=='pooled')].to_string(index=False))
if __name__=='__main__':main()
