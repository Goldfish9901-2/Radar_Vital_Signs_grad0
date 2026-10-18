"""Independent metrics for paired training objectives; no fitting."""
import json,time,zipfile,hashlib,shutil,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

start=time.perf_counter();root=Path('experiment_runs/ftu_objectives')
with zipfile.ZipFile('tmp/ftu_objectives_result/results.zip') as z:
    hashes=json.loads(z.read('artifact_manifest.json'))
    for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    for name in z.namelist():
        assert '/' not in name and '\\' not in name
        (root/('kaggle_artifact_manifest.json' if name=='artifact_manifest.json' else name)).write_bytes(z.read(name))
shutil.copyfile('tmp/ftu_objectives_result/kernel.log',root/'kernel.log')
assert json.loads((root/'protocol.json').read_text())==json.loads(subprocess.check_output(['git','show','5fcc2ab:experiment_runs/ftu_objectives/protocol.json']))
assert (root/'record_split.csv').read_bytes()==Path('experiment_runs/ftu_real_d/record_split.csv').read_bytes()
f=pd.read_csv(root/'inputs.csv');p=pd.read_csv(root/'predictions.csv');p['checkpoint']='best'
last=pd.read_csv(root/'epoch60_predictions_train_val.csv');assert set(last.split)=={'train','val'};last['checkpoint']='epoch60'
summary=pd.read_csv(root/'summary.csv');history=pd.read_csv(root/'history.csv')
assert len(summary)==72 and summary.epochs.eq(60).all()
assert history.groupby(['run','split']).epoch.apply(lambda s:list(s)==list(range(1,61))).all()
ex=json.loads((root/'execution.json').read_text())
for key in ['initialization_hashes','batch_order_hashes']:
    for seed in [42,43,44]:assert len({v for k,v in ex[key].items() if f'_seed{seed}_' in k})==1
arrays=np.load(root/'features.npz');old=np.load('experiment_runs/ftu_real_d/features.npz')
for name in ['original','masked','mask']:np.testing.assert_array_equal(arrays[name],old[name])
perms=pd.read_csv(root/'permutations.csv');oldperm=pd.read_csv('experiment_runs/ftu_real_d/permutations.csv')
for run,g in perms.groupby('run'):
    ref=oldperm[oldperm.run==run.split('_',1)[1]][['index','donor']]
    np.testing.assert_array_equal(g[['index','donor']],ref)

def calc(g):
    y=g.label_bpm.to_numpy();p=g.pred_bpm.to_numpy()
    a=y-g.groupby('sample_tag').label_bpm.transform('mean').to_numpy()
    b=p-g.groupby('sample_tag').pred_bpm.transform('mean').to_numpy()
    def pair(a,b,prefix):
        sa,sb=a.std(),b.std()
        return {prefix+'rmse':float(np.sqrt(np.mean((a-b)**2))),prefix+'r':float(np.corrcoef(a,b)[0,1]) if sa>1e-10 and sb>1e-10 else np.nan,
                prefix+'std_ratio':float(sb/sa) if sa>1e-10 else np.nan}
    return dict(mae=float(abs(y-p).mean()),centered_r2=float(1-np.sum((a-b)**2)/np.sum(a*a)) if np.sum(a*a)>1e-10 else np.nan,
                **pair(y,p,''),**pair(a,b,'centered_'))

rows=[];record_rows=[];maxdelta=0
for (checkpoint,run,split),g in pd.concat([p,last]).groupby(['checkpoint','run','split']):
    objective,seed,arm=run.split('_');meta=dict(checkpoint=checkpoint,run=run,objective=objective,seed=int(seed[4:]),arm='true' if arm=='true' else 'shuffle',split=split)
    m=calc(g);rows.append(dict(**meta,subject='pooled',**m))
    if checkpoint=='best':
        ref=summary[(summary.run==run)&(summary.split==split)].iloc[0]
        for k,v in m.items():maxdelta=max(maxdelta,abs(v-ref[k]));np.testing.assert_allclose(v,ref[k],atol=2e-5,rtol=1e-5)
    for subject,h in g.groupby('group_key'):rows.append(dict(**meta,subject=subject,**calc(h)))
    for tag,h in g.groupby('sample_tag'):record_rows.append(dict(**meta,sample_tag=tag,**calc(h)))
metrics=pd.DataFrame(rows);cols=list(m)
keys=['checkpoint','run','objective','seed','arm','split']
macro=metrics[metrics.subject!='pooled'].groupby(keys)[cols].mean().reset_index();macro['subject']='equal_subject_mean'
metrics=pd.concat([metrics,macro]);metrics.to_csv(root/'metrics.csv',index=False)
pd.DataFrame(record_rows).to_csv(root/'record_metrics.csv',index=False)
aggregate=metrics.groupby(['checkpoint','objective','arm','split','subject'])[cols].mean().reset_index()
aggregate.to_csv(root/'aggregate_metrics.csv',index=False)
pairs=[]
for (checkpoint,objective,seed,split,subject),g in metrics.groupby(['checkpoint','objective','seed','split','subject']):
    a=g[g.arm=='true'].iloc[0]
    for _,b in g[g.arm=='shuffle'].iterrows():pairs.append(dict(checkpoint=checkpoint,objective=objective,seed=seed,shuffle_run=b.run,split=split,subject=subject,
        rmse_improvement=b.centered_rmse-a.centered_rmse,r2_improvement=a.centered_r2-b.centered_r2))
pd.DataFrame(pairs).to_csv(root/'label_contrasts.csv',index=False)
contrasts=[]
for (checkpoint,seed,split,subject),g in metrics.groupby(['checkpoint','seed','split','subject']):
    for _,a in g[g.objective=='absolute'].iterrows():
        b=g[g.run==a.run.replace('absolute_','dynamic_')].iloc[0]
        contrasts.append(dict(checkpoint=checkpoint,seed=seed,label_condition=a.run.split('_')[-1],split=split,subject=subject,
            rmse_improvement=a.centered_rmse-b.centered_rmse,r2_improvement=b.centered_r2-a.centered_r2))
pd.DataFrame(contrasts).to_csv(root/'objective_contrasts.csv',index=False)
fig,axes=plt.subplots(2,2,figsize=(11,8))
for i,obj in enumerate(['absolute','dynamic']):
    for j,split in enumerate(['train','val']):
        ax=axes[i,j]
        for run,g in history[(history.run.str.startswith(obj+'_'))&(history.split==split)].groupby('run'):
            ax.plot(g.epoch,g.centered_r2,color='tab:blue' if run.endswith('_true') else 'tab:orange',alpha=.8 if run.endswith('_true') else .3)
        ax.axhline(0,color='gray',ls='--');ax.set(title=obj+' / '+split,xlabel='Epoch',ylabel='Record-centered R2')
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([0],[0],color='tab:blue',label='True labels'),Line2D([0],[0],color='tab:orange',label='Shuffled labels')],loc='upper center',ncol=2)
fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(root/'learning_curves.png',dpi=160);plt.close(fig)
subjects=sorted(f.group_key.unique());fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,obj in zip(axes,['absolute','dynamic']):
    for arm in ['true','shuffle']:
        g=aggregate[(aggregate.checkpoint=='best')&(aggregate.objective==obj)&(aggregate.arm==arm)&(aggregate.split=='evaluation')].set_index('subject').loc[subjects]
        ax.plot(range(6),g.centered_r2,'o-',label=arm)
    ax.axhline(0,color='gray',ls='--');ax.set_xticks(range(6),[s.rsplit('/',1)[-1] for s in subjects]);ax.set(title=obj,xlabel='Subject',ylabel='Unseen-capture centered R2');ax.legend()
fig.tight_layout();fig.savefig(root/'subject_results.png',dpi=170);plt.close(fig)
pd.DataFrame([dict(run=run,**calc(g)) for run,g in last[last.split=='train'].groupby('run')]).to_csv(root/'epoch60_training_fit.csv',index=False)
(root/'local_validation.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-start,original_artifact_hashes=len(hashes),
    prediction_groups_verified=72,max_metric_delta=maxdelta,exactly60_epochs_all24=True,paired_initialization_and_record_orders=True,
    unchanged_D_features_split_and_permutations=True,frozen_protocol_unchanged=True),indent=2))
print(aggregate[aggregate.subject=='equal_subject_mean'].to_string(index=False))
print('EVALUATION SUBJECTS')
print(aggregate[(aggregate.checkpoint=='best')&(aggregate.split=='evaluation')&~aggregate.subject.isin(['pooled','equal_subject_mean'])][['objective','arm','subject','centered_rmse','centered_r2','centered_r']].to_string(index=False))
