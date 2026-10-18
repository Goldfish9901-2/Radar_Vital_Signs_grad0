"""Independent real-data metrics and paired controls; no fitting."""
import json,time,zipfile,hashlib,shutil,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

start=time.perf_counter();root=Path('experiment_runs/ftu_real_d')
with zipfile.ZipFile('tmp/ftu_real_d_result/results.zip') as z:
    manifest=json.loads(z.read('artifact_manifest.json'))
    for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    for name in z.namelist():
        assert '/' not in name and '\\' not in name
        (root/('kaggle_artifact_manifest.json' if name=='artifact_manifest.json' else name)).write_bytes(z.read(name))
shutil.copyfile('tmp/ftu_real_d_result/kernel.log',root/'kernel.log')
assert json.loads((root/'protocol.json').read_text())==json.loads(subprocess.check_output(
    ['git','show','b04aa8e:experiment_runs/ftu_real_d/protocol.json']))
f=pd.read_csv(root/'inputs.csv');p=pd.read_csv(root/'predictions.csv');s=pd.read_csv(root/'summary.csv')
old=Path('experiment_runs/ftu_record_learnability/record_split.csv')
assert (root/'record_split.csv').read_bytes()==old.read_bytes()
perms=pd.read_csv(root/'permutations.csv')
for run,g in perms.groupby('run'):
    ids=g['index'].to_numpy();donors=g.donor.to_numpy()
    assert (f.iloc[ids].split=='train').all() and (f.iloc[donors].split=='train').all()
    assert np.array_equal(f.iloc[ids].sample_tag.to_numpy(),f.iloc[donors].sample_tag.to_numpy())
    assert np.array_equal(np.sort(ids),np.sort(donors))
for shuffle in [101,202,303]:
    mappings=[perms[perms.run==f'seed{seed}_shuffle{shuffle}'][['index','donor']].to_numpy() for seed in [42,43,44]]
    assert all(np.array_equal(mappings[0],m) for m in mappings[1:])
arrays=np.load(root/'features.npz');raw=arrays['original'];masked=arrays['masked'];band=arrays['mask']
assert band.sum()==29 and not masked[:,~band].any()
np.testing.assert_array_equal(raw[:,band],masked[:,band])
baseline=pd.read_csv(root/'baselines.csv')
hz=np.fft.rfftfreq(256,1/20)
np.testing.assert_allclose(baseline.peak_bpm,hz[np.flatnonzero(band)[raw[:,band].argmax(1)]]*60)
for method in ['peak','mean','median']:
    g=f.copy();g['run']=method;g['pred_bpm']=baseline[method+'_bpm'];p=pd.concat([p,g],ignore_index=True)

def calc(g):
    y=g.label_bpm.to_numpy();pred=g.pred_bpm.to_numpy()
    cy=y-g.groupby('sample_tag').label_bpm.transform('mean').to_numpy()
    cp=pred-g.groupby('sample_tag').pred_bpm.transform('mean').to_numpy()
    def pair(a,b,prefix):
        sa,sb=a.std(),b.std()
        return {prefix+'rmse':float(np.sqrt(np.mean((a-b)**2))),
                prefix+'r':float(np.corrcoef(a,b)[0,1]) if sa>1e-10 and sb>1e-10 else np.nan,
                prefix+'std_ratio':float(sb/sa) if sa>1e-10 else np.nan}
    return dict(n=len(g),records=g.sample_tag.nunique(),mae=float(abs(y-pred).mean()),
                centered_r2=float(1-np.sum((cy-cp)**2)/np.sum(cy*cy)) if np.sum(cy*cy)>1e-10 else np.nan,
                **pair(y,pred,''),**pair(cy,cp,'centered_'))

rows=[];recordrows=[];maxdelta=0
for (run,split),g in p.groupby(['run','split']):
    m=calc(g);rows.append(dict(run=run,split=split,subject='pooled',**m))
    if run.startswith('seed'):
        ref=s[(s.run==run)&(s.split==split)].iloc[0]
        for k in ['mae','rmse','r','std_ratio','centered_rmse','centered_r','centered_std_ratio','centered_r2']:
            maxdelta=max(maxdelta,abs(m[k]-ref[k]));np.testing.assert_allclose(m[k],ref[k],atol=2e-5,rtol=1e-5)
    for subject,h in g.groupby('group_key'):rows.append(dict(run=run,split=split,subject=subject,**calc(h)))
    for tag,h in g.groupby('sample_tag'):recordrows.append(dict(run=run,split=split,sample_tag=tag,**calc(h)))
metrics=pd.DataFrame(rows);cols=['mae','rmse','r','std_ratio','centered_rmse','centered_r','centered_std_ratio','centered_r2']
macro=metrics[metrics.subject!='pooled'].groupby(['run','split'])[cols].mean().reset_index();macro['subject']='equal_subject_mean'
metrics=pd.concat([metrics,macro],ignore_index=True);metrics.to_csv(root/'metrics.csv',index=False)
pd.DataFrame(recordrows).to_csv(root/'record_metrics.csv',index=False)
metrics['arm']=metrics.run.map(lambda r:'true' if r.endswith('_true') else ('shuffle' if '_shuffle' in r else r))
aggregate=metrics.groupby(['arm','split','subject'])[cols].mean().reset_index()
aggregate.to_csv(root/'aggregate_metrics.csv',index=False)
paired=[]
for seed in [42,43,44]:
    for shuffle in [101,202,303]:
        for split in ['train','val','evaluation']:
            for subject in metrics.subject.unique():
                a=metrics[(metrics.run==f'seed{seed}_true')&(metrics.split==split)&(metrics.subject==subject)].iloc[0]
                b=metrics[(metrics.run==f'seed{seed}_shuffle{shuffle}')&(metrics.split==split)&(metrics.subject==subject)].iloc[0]
                paired.append(dict(seed=seed,shuffle_seed=shuffle,split=split,subject=subject,
                    mae_improvement=b.mae-a.mae,centered_rmse_improvement=b.centered_rmse-a.centered_rmse,
                    centered_r2_improvement=a.centered_r2-b.centered_r2))
pd.DataFrame(paired).to_csv(root/'paired_contrasts.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4))
subjects=sorted(f.group_key.unique())
for ax,metric in zip(axes,['centered_r2','centered_rmse']):
    for arm in ['true','shuffle','mean']:
        g=aggregate[(aggregate.arm==arm)&(aggregate.split=='evaluation')].set_index('subject').loc[subjects]
        ax.plot(range(6),g[metric],'o-',label=arm)
    ax.set_xticks(range(6),[t.rsplit('/',1)[-1] for t in subjects]);ax.set(xlabel='Subject',ylabel=metric);ax.legend()
axes[0].axhline(0,color='gray',ls='--');fig.tight_layout();fig.savefig(root/'subject_results.png',dpi=170);plt.close(fig)
history=pd.read_csv(root/'history.csv');fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,split in zip(axes,['train','val']):
    for run,g in history[history.split==split].groupby('run'):
        ax.plot(g.epoch,g.centered_r2,alpha=.8 if run.endswith('_true') else .3,
                color='tab:blue' if run.endswith('_true') else 'tab:orange')
    ax.axhline(0,color='gray',ls='--');ax.set(title=split,xlabel='Epoch',ylabel='Record-centered R2 (correct labels)')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],color='tab:blue',label='True labels'),
                       Line2D([0],[0],color='tab:orange',label='Shuffled labels')])
fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(3,2,figsize=(11,9))
for ax,subject in zip(axes.flat,subjects):
    tag=sorted(f[(f.group_key==subject)&(f.split=='evaluation')].sample_tag.unique())[0]
    g=p[(p.sample_tag==tag)&p.run.str.startswith('seed')].copy()
    g['arm']=np.where(g.run.str.endswith('_true'),'true','shuffle')
    ref=f[f.sample_tag==tag].sort_values('window_start');ax.plot((ref.window_start+128)/20,ref.label_bpm,'k-o',label='reference')
    for arm in ['true','shuffle']:
        h=g[g.arm==arm].groupby('window_start').pred_bpm.mean();ax.plot((h.index+128)/20,h.values,'o-',label=arm+' mean')
    ax.set(title=tag,xlabel='Window center (s, assumed 20 Hz)',ylabel='HR bpm');ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(root/'example_trajectories.png',dpi=150);plt.close(fig)
(root/'local_validation.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-start,
    original_files_verified=len(manifest),prediction_groups_verified=len(s),max_metric_delta=maxdelta,
    exact_split=True,frozen_protocol_semantically_unchanged=True,permutation_records_verified=True,mask_verified=True),indent=2))
print(aggregate[(aggregate.subject.isin(['pooled','equal_subject_mean']))].to_string(index=False))
print('TRUE/SHUFFLE EVALUATION SUBJECTS')
print(aggregate[(aggregate.split=='evaluation')&aggregate.arm.isin(['true','shuffle'])].to_string(index=False))
