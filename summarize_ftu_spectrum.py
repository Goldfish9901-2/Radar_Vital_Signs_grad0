"""Independent local verification, stratification, and plots; no training."""
import json,time,zipfile,hashlib,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.training.ftu_spectrum_factorial import PROTOCOL,metrics,transform

start=time.perf_counter();root=Path('experiment_runs/ftu_spectrum_factorial')
with zipfile.ZipFile('tmp/ftu_spectrum_result/results.zip') as z:
    manifest=json.loads(z.read('artifact_manifest.json'))
    for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    for name in z.namelist():
        assert '/' not in name and '\\' not in name
        (root/('kaggle_artifact_manifest.json' if name=='artifact_manifest.json' else name)).write_bytes(z.read(name))
shutil.copyfile('tmp/ftu_spectrum_result/kernel.log',root/'kernel.log')
p=pd.read_csv(root/'predictions.csv');summary=pd.read_csv(root/'summary.csv')
selection=json.loads((root/'selection.json').read_text())
assert summary.query("split=='val'").groupby('arm').centered_rmse.mean().sort_index().idxmin()==selection['selected_arm']
x=np.load(root/'development_features.npz')['x_freq']
np.testing.assert_array_equal(x,np.load('experiment_runs/ftu_record_learnability/synthetic_features.npz')['x_freq'])
old=pd.read_csv('experiment_runs/ftu_record_learnability/summary.csv')
reproduction=[]
for seed in PROTOCOL['seeds']:
    for split,oldsplit in [('train','train'),('val','val'),('development','evaluation')]:
        a=summary[(summary.arm=='A')&(summary.seed==seed)&(summary.split==split)].iloc[0]
        b=old[(old.seed==seed)&(old.split==oldsplit)].iloc[0]
        reproduction.append(dict(seed=seed,split=split,rmse_delta=a.centered_rmse-b.centered_rmse,
                                 best_epoch_matches=bool(a.best_epoch==b.best_epoch)))
pd.DataFrame(reproduction).to_csv(root/'A_reproduction.csv',index=False)

# Baselines read ORIGINAL per-window-normalized features, before optional bin scaling.
baseline=[]
for prefix in ['development','final']:
    f=pd.read_csv(root/f'{prefix}_inputs.csv');x=np.load(root/f'{prefix}_features.npz')['x_freq']
    peak=x.argmax(1);i=np.arange(len(x));assert ((peak>0)&(peak<128)).all()
    delta=.5*(x[i,peak-1]-x[i,peak+1])/(x[i,peak-1]-2*x[i,peak]+x[i,peak+1])
    for arm,values in [('peak',peak*20/256*60),('quadratic',(peak+delta)*20/256*60)]:
        g=f.copy();g['pred_bpm']=values;g['arm']=arm;g['seed']=-1;baseline.append(g)
baseline=pd.concat(baseline,ignore_index=True);baseline.to_csv(root/'baseline_predictions.csv',index=False)
allp=pd.concat([p,baseline],ignore_index=True)
rows=[];strata=[];verification_deltas=[]
for (arm,seed,split),f in allp.groupby(['arm','seed','split']):
    f=f.copy()
    f['cy']=f.label_bpm-f.groupby('sample_tag').label_bpm.transform('mean')
    f['cp']=f.pred_bpm-f.groupby('sample_tag').pred_bpm.transform('mean')
    def calc(g):return metrics(g.label_bpm.to_numpy(),g.pred_bpm.to_numpy(),g.cy.to_numpy(),g.cp.to_numpy())
    row=dict(arm=arm,seed=seed,split=split,**calc(f));rows.append(row)
    if arm in PROTOCOL['arms']:
        expected=summary[(summary.arm==arm)&(summary.seed==seed)&(summary.split==split)].iloc[0]
        for key in ['mae','rmse','r','std_ratio','centered_rmse','centered_r','centered_std_ratio']:
            verification_deltas.append(abs(row[key]-expected[key]))
            # Kaggle used float32 predictions/group means; CSV reload computes float64.
            np.testing.assert_allclose(row[key],expected[key],atol=2e-5,rtol=1e-6)
    f['frequency_band']=pd.cut(f.frequency_hz,PROTOCOL['strata']['frequency_edges_hz'],right=False).astype(str)
    f['noise_band']=pd.cut(f.relative_noise_std,PROTOCOL['strata']['noise_edges'],right=False).astype(str)
    for category in ['frequency_band','noise_band','frequency_hz']:
        for value,g in f.groupby(category):
            strata.append(dict(arm=arm,seed=seed,split=split,stratum=category,value=str(value),**calc(g)))
pd.DataFrame(rows).to_csv(root/'independent_metrics.csv',index=False)
pd.DataFrame(strata).to_csv(root/'stratified_metrics.csv',index=False)
paired=[]
for split,f in summary.groupby('split'):
    for first,second in [('A','B'),('A','C'),('C','D'),('B','D')]:
        for seed in PROTOCOL['seeds']:
            a=f[(f.arm==first)&(f.seed==seed)].iloc[0];b=f[(f.arm==second)&(f.seed==seed)].iloc[0]
            paired.append(dict(split=split,contrast=first+'-'+second,seed=seed,
                               centered_rmse_improvement=a.centered_rmse-b.centered_rmse,
                               rmse_improvement=a.rmse-b.rmse))
pd.DataFrame(paired).to_csv(root/'paired_contrasts.csv',index=False)
sc=np.load(root/'scaler.npz');hz=np.fft.rfftfreq(256,1/20);band=(hz>=.5)&(hz<=2.75)
pd.DataFrame(dict(bin=np.arange(129),hz=hz,train_mean=sc['mean'],train_std=sc['scale'],
                  multiplier=1/sc['scale'],retained=band)).to_csv(root/'bin_scaling.csv',index=False)
energy=[]
x=np.load(root/'development_features.npz')['x_freq'][:768]
for arm in PROTOCOL['arms']:
    tx=transform(x,sc['mean'],sc['scale'],arm)
    energy.append(dict(arm=arm,in_band_mean_square=float((tx[:,band]**2).mean()),
                       out_band_mean_square=float((tx[:,~band]**2).mean()),
                       out_band_total_fraction=float((tx[:,~band]**2).sum()/(tx**2).sum())))
pd.DataFrame(energy).to_csv(root/'input_energy.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,split in zip(axes,['val','final_on_grid','final_between_grid']):
    for j,arm in enumerate(PROTOCOL['arms']):
        v=summary[(summary.arm==arm)&(summary.split==split)].centered_rmse
        ax.scatter(np.full(3,j),v,s=28);ax.plot([j-.2,j+.2],[v.mean()]*2,'k-')
    ax.axhline(4.6875,color='gray',ls='--');ax.set(xticks=range(4),xticklabels=list(PROTOCOL['arms']),title=split,ylabel='Centered RMSE (bpm)')
fig.tight_layout();fig.savefig(root/'factorial_results.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4))
s=pd.DataFrame(strata)
for ax,category in zip(axes,['frequency_band','noise_band']):
    for arm in PROTOCOL['arms']:
        g=s[(s.arm==arm)&(s.split=='final_between_grid')&(s.stratum==category)].groupby('value').centered_rmse.mean()
        ax.plot(range(len(g)),g.values,'o-',label=arm);ax.set_xticks(range(len(g)),g.index,rotation=20)
    ax.set(title=category,ylabel='Centered RMSE (bpm)');ax.legend()
fig.tight_layout();fig.savefig(root/'stratified_results.png',dpi=170);plt.close(fig)
(root/'local_validation.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-start,
    verified_original_hashes=len(manifest),verified_prediction_groups=len(summary),
    exact_original_features=True,max_metric_float32_csv_delta=float(max(verification_deltas)),
    A_reproduction_max_rmse_delta=float(abs(pd.DataFrame(reproduction).rmse_delta).max()),
    selected_arm=selection['selected_arm']),indent=2))
print(summary.groupby(['arm','split'])[['rmse','centered_rmse','r','centered_r','std_ratio','centered_std_ratio']].mean().to_string())
print(selection);print('baseline',pd.DataFrame(rows).query("seed==-1").to_string(index=False))
