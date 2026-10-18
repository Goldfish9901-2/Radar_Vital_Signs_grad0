"""Local independent paired audit summary; no fitting or offset search."""
import json,time,hashlib,zipfile,shutil,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
start=time.perf_counter();root=Path('experiment_runs/ftu_target_audit')
with zipfile.ZipFile('tmp/ftu_target_audit_result/results.zip') as z:
    manifest=json.loads(z.read('artifact_manifest.json'))
    for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    for name in z.namelist():
        rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
        if name.endswith('/'):continue
        dest=root/('kaggle_artifact_manifest.json' if name=='artifact_manifest.json' else name)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
shutil.copyfile('tmp/ftu_target_audit_result/kernel.log',root/'kernel.log')
for name in ['protocol.json','selected_records.csv']:
    prior=subprocess.check_output(['git','show','4ba546d:experiment_runs/ftu_target_audit/'+name])
    if name.endswith('json'):assert json.loads(prior)==json.loads((root/name).read_text())
    else:assert prior==(root/name).read_bytes()
selected=pd.read_csv(root/'selected_records.csv');ro=pd.read_csv(root/'readouts.csv');ov=pd.read_csv(root/'overlap.csv');fu=pd.read_csv(root/'fusion.csv')
r=ro[ro.scored].copy();o=ov[ov.scored].copy();v=fu[fu.scored].copy()
assert r.groupby(['arm','kind']).size().eq(324).all()
assert o.groupby('arm').size().eq(270).all()
assert r.window_start.min()>=256 and o.left_start.min()>=256
for kind in r.kind.unique():
    a=r[(r.arm=='A')&(r.kind==kind)].sort_values(['sample_tag','window_start'])
    b=r[(r.arm=='B')&(r.kind==kind)].sort_values(['sample_tag','window_start'])
    np.testing.assert_array_equal(a[['sample_tag','window_start','label_bpm']],b[['sample_tag','window_start','label_bpm']])
def metrics(g):
    y=g.label_bpm.to_numpy();p=g.peak_bpm.to_numpy()
    cy=y-g.groupby('sample_tag').label_bpm.transform('mean').to_numpy()
    cp=p-g.groupby('sample_tag').peak_bpm.transform('mean').to_numpy()
    def corr(a,b):return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-10 and np.std(b)>1e-10 else np.nan
    return dict(n=len(g),mae=float(abs(p-y).mean()),rmse=float(np.sqrt(np.mean((p-y)**2))),r=corr(y,p),
        centered_rmse=float(np.sqrt(np.mean((cp-cy)**2))),centered_r2=float(1-np.sum((cp-cy)**2)/np.sum(cy*cy)) if np.sum(cy*cy)>1e-10 else np.nan,
        centered_r=corr(cy,cp),centered_std_ratio=float(cp.std()/cy.std()) if cy.std()>1e-10 else np.nan,
        near_reference_power_fraction=float(g.reference_near_power_fraction.mean()))
rows=[];record_metrics=[]
for (arm,kind),g in r.groupby(['arm','kind']):
    rows.append(dict(arm=arm,kind=kind,subject='pooled',**metrics(g)))
    for subject,h in g.groupby('subject'):rows.append(dict(arm=arm,kind=kind,subject=subject,**metrics(h)))
    for tag,h in g.groupby('sample_tag'):
        row=dict(arm=arm,kind=kind,sample_tag=tag,**metrics(h))
        row['peak_adjacent_abs_change']=float(abs(np.diff(h.sort_values('window_start').peak_bpm)).mean())
        record_metrics.append(row)
table=pd.DataFrame(rows);cols=[c for c in table if c not in ['arm','kind','subject','n']]
macro=table[table.subject!='pooled'].groupby(['arm','kind'])[cols].mean().reset_index();macro['subject']='equal_subject_mean'
table=pd.concat([table,macro]);table.to_csv(root/'metrics.csv',index=False)
pd.DataFrame(record_metrics).to_csv(root/'record_metrics.csv',index=False)
oc=['set_turnover','weight_l1','raw_delta_nrms','detrended_delta_nrms','demeaned_delta_nrms','demeaned_phase_nrms','overlap_spectrum_tv']
o.groupby(['arm','subject'])[oc].mean().to_csv(root/'subject_consistency.csv')
o.groupby('arm')[oc].agg(['mean','median','max']).to_csv(root/'consistency_summary.csv')
v.groupby(['arm','subject'])[['fusion_power_ratio','fused_pre_z_std','amplitude_min','amplitude_median','max_abs_raw_delta']].median().to_csv(root/'subject_fusion.csv')
quality=[];spikes=[]
for tag in selected.sample_tag:
    with np.load(root/'signals'/f'{tag}.npz') as z:
        for j in range(1,len(z['starts'])):
            shift=int(z['starts'][j]-z['starts'][j-1])
            np.testing.assert_array_equal(z['B_z'][j-1,:,shift:],z['B_z'][j,:,:256-shift])
            np.testing.assert_array_equal(z['B_fused_delta'][j-1,shift+1:],z['B_fused_delta'][j,1:256-shift])
        mapped=np.interp(z['radar_time_s'],z['reference_time'],z['reference_hr'],left=np.nan,right=np.nan)
        labels=[np.nanmean(mapped[int(s):int(s)+256]) for s in z['starts']]
        observed=ro[(ro.sample_tag==tag)&(ro.arm=='A')&(ro.kind=='fused')].sort_values('window_start').label_bpm
        np.testing.assert_allclose(labels,observed,atol=2e-5)
        assert np.array_equal(z['B_bins'],np.repeat(z['B_bins'][:1],8,axis=0))
        np.testing.assert_array_equal(z['B_weights'],np.repeat(z['B_weights'][:1],8,axis=0))
        np.testing.assert_array_equal(z['A_bins'][0],z['B_bins'][0]);np.testing.assert_array_equal(z['A_weights'][0],z['B_weights'][0])
        for arm in ['A','B']:
            signal=z[arm+'_fused'];phase=z[arm+'_phase'];weights=z[arm+'_weights']
            np.testing.assert_allclose(signal,np.sum(phase*weights[:,:,None],axis=1),atol=1e-6)
            amp=z[arm+'_amplitude'][z['scored']];delta=z[arm+'_delta'][z['scored']]
            quality.append(dict(sample_tag=tag,arm=arm,epsilon_dominated_fraction=float(np.mean(amp**2<=1e-6)),
                                abs_delta_p99=float(np.quantile(abs(delta),.99)),abs_delta_max=float(abs(delta).max())))
            wi,rank,t=np.unravel_index(np.argmax(abs(delta[:,:,1:])),delta[:,:,1:].shape);t+=1
            original_wi=np.flatnonzero(z['scored'])[wi];complex_signal=z[arm+'_z'][original_wi,rank]
            angle=float(np.angle(complex_signal[t]*np.conj(complex_signal[t-1])))
            ratio=float(amp[wi,rank,t-1]/amp[wi,rank,t])
            spikes.append(dict(sample_tag=tag,arm=arm,window_start=int(z['starts'][original_wi]),rank=rank,local_frame=t,
                signed_edacm_delta=float(delta[wi,rank,t]),conjugate_increment_rad=angle,previous_current_amplitude_ratio=ratio,
                current_amplitude=float(amp[wi,rank,t]),previous_amplitude=float(amp[wi,rank,t-1])))
pd.DataFrame(quality).to_csv(root/'raw_signal_quality.csv',index=False)
pd.DataFrame(spikes).to_csv(root/'posthoc_max_increment_audit.csv',index=False)
# Fixed hash-first record per subject for detailed single-target/fusion displays.
for _,row in selected.sort_values('selection_hash').groupby('group_key',sort=True).first().iterrows():
    with np.load(root/'signals'/f'{row.sample_tag}.npz') as z:
        hz=np.fft.rfftfreq(256,1/20);band=(hz>=.5)&(hz<=2.75);times=(z['starts']+128)/20
        powers={}
        for arm in ['A','B']:
            traces=np.concatenate([z[arm+'_phase']*z[arm+'_weights'][:,:,None],z[arm+'_fused'][:,None,:]],axis=1)
            powers[arm]=10*np.log10(abs(np.fft.rfft(traces*np.hanning(256),axis=-1))[:,:,band]**2+1e-20)
        lo=min(p.min() for p in powers.values());hi=max(p.max() for p in powers.values())
        fig,axes=plt.subplots(2,4,figsize=(15,6),sharex=True,sharey=True)
        ref=ro[(ro.sample_tag==row.sample_tag)&(ro.arm=='A')&(ro.kind=='fused')].sort_values('window_start').label_bpm
        for i,arm in enumerate(['A','B']):
            for k in range(4):
                ax=axes[i,k];im=ax.pcolormesh(times,hz[band]*60,powers[arm][:,k].T,shading='auto',vmin=lo,vmax=hi)
                ax.plot(times,ref,'w.-',lw=1);ax.set(title=arm+(' weighted single'+str(k) if k<3 else ' fused pre-z'),xlabel='Window center (s)',ylabel='bpm')
        fig.suptitle(row.sample_tag+'; shared raw-phase power scale, white=reference (assumed time)')
        fig.tight_layout(rect=[0,0,.92,.95]);cax=fig.add_axes([.94,.15,.012,.65])
        fig.colorbar(im,cax=cax,label='Raw weighted phase power (dB, arbitrary units)')
        fig.savefig(root/'figures'/f'{row.sample_tag}_single_fusion.png',dpi=120);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,metric in zip(axes,['raw_delta_nrms','detrended_delta_nrms','overlap_spectrum_tv']):
    for j,arm in enumerate(['A','B']):
        values=o[o.arm==arm].groupby('sample_tag')[metric].mean()
        ax.scatter(np.full(len(values),j),values,s=10,alpha=.5)
        ax.plot([j-.2,j+.2],[values.mean()]*2,'k-')
    ax.set(xticks=[0,1],xticklabels=['A','B'],ylabel=metric,title='Scored overlaps: record means')
fig.tight_layout();fig.savefig(root/'consistency.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4));subjects=sorted(selected.group_key.unique())
for ax,metric in zip(axes,['mae','centered_r2']):
    for arm in ['A','B']:
        g=table[(table.arm==arm)&(table.kind=='fused')].set_index('subject').loc[subjects]
        ax.plot(range(6),g[metric],'o-',label=arm)
    ax.set_xticks(range(6),[s.rsplit('/',1)[-1] for s in subjects]);ax.set(xlabel='Subject',ylabel=metric);ax.legend()
axes[1].axhline(0,color='gray',ls='--');fig.tight_layout();fig.savefig(root/'reference_correspondence.png',dpi=160);plt.close(fig)
(root/'local_validation.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-start,
    original_artifact_hashes=len(manifest),records=54,scored_windows_per_arm=324,scored_overlap_pairs_per_arm=270,
    same_labels_windows=True,frozen_targets_weights=True,initial_AB_equal=True,fusion_recomputed=True,
    fixed_overlap_complex_and_delta_exact=True,reference_mapping_recomputed=True,protocol_selection_unchanged=True),indent=2))
print(table.to_string(index=False))
print('CONSISTENCY',o.groupby('arm')[oc].mean().to_string())
print('FUSION',v.groupby('arm').fusion_power_ratio.agg(['mean','median','min','max']).to_string())
print('POWER RATIO BELOW1',v.groupby('arm').fusion_power_ratio.apply(lambda s:(s<1).mean()).to_string())
