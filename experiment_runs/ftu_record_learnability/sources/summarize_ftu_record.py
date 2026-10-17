"""Local independent verification and figures; no fitting."""
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root=Path('experiment_runs/ftu_record_learnability')
start=time.perf_counter()
with zipfile.ZipFile('tmp/ftu_record_result/results.zip') as z:
    hashes=json.loads(z.read('artifact_manifest.json'))
    for name,digest in hashes.items():
        assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    for name in z.namelist():
        assert '/' not in name and '\\' not in name
        if name=='artifact_manifest.json': name_out='kaggle_artifact_manifest.json'
        else: name_out=name
        (root/name_out).write_bytes(z.read(name))
shutil.copyfile('tmp/ftu_record_result/kernel.log',root/'kernel.log')
p=pd.read_csv(root/'predictions.csv'); summary=pd.read_csv(root/'summary.csv')
rows=[]
for (run,split), f in p.groupby(['run','split']):
    y=f.label_bpm.to_numpy(); pred=f.pred_bpm.to_numpy()
    a=y-f.groupby('sample_tag').label_bpm.transform('mean').to_numpy()
    b=pred-f.groupby('sample_tag').pred_bpm.transform('mean').to_numpy()
    rmse=np.sqrt(np.mean((a-b)**2)); r=np.corrcoef(a,b)[0,1]
    row=summary[(summary.run==run)&(summary.split==split)].iloc[0]
    np.testing.assert_allclose([rmse,r,abs(y-pred).mean()],
                               [row.centered_rmse,row.centered_r,row.mae],atol=1e-7)
    rows.append(dict(run=run,split=split,centered_rmse=rmse,centered_r=r))
pd.DataFrame(rows).to_csv(root/'independent_scores.csv',index=False)
history=pd.read_csv(root/'history.csv')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for run,f in history.groupby('run'):
    for split,ls in [('train','--'),('val','-')]:
        g=f[f.split==split]
        axes[0].plot(g.epoch,g.centered_rmse,ls,label=f'{run}/{split}',alpha=.7)
        axes[1].plot(g.epoch,g.centered_r,ls,alpha=.7)
axes[0].set(xlabel='Epoch',ylabel='Record-centered RMSE (bpm)')
axes[1].set(xlabel='Epoch',ylabel='Record-centered Pearson r')
axes[0].legend(fontsize=5); fig.tight_layout(); fig.savefig(root/'learning_curves.png',dpi=160); plt.close(fig)
evaluation=p[(p.split=='evaluation') & p.run.str.startswith('synthetic')]
fig,ax=plt.subplots(figsize=(5,5))
for run,f in evaluation.groupby('run'): ax.scatter(f.label_bpm,f.pred_bpm,s=7,alpha=.5,label=run)
ax.plot([40,150],[40,150],'k--');ax.set(xlabel='Reference bpm',ylabel='Predicted bpm');ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(root/'synthetic_readout.png',dpi=160);plt.close(fig)
execution=json.loads((root/'execution.json').read_text())
gate=all((summary[(summary.domain=='synthetic') & summary.split.isin(['val','evaluation'])].centered_rmse<=4.6875)
         & (summary[(summary.domain=='synthetic') & summary.split.isin(['val','evaluation'])].centered_r>=.9))
assert gate==execution['synthetic_gate_pass']
if not gate: assert not (summary.domain=='real').any()
# Post-hoc non-trained spectral readout: establishes whether this synthetic
# representation retains frequency. This does not change the preregistered gate.
sf=pd.read_csv(root/'synthetic_inputs.csv')
sx=np.load(root/'synthetic_features.npz')['x_freq']
peak=np.argmax(sx,axis=1)
assert ((peak>0)&(peak<128)).all()
i=np.arange(len(peak)); left=sx[i,peak-1]; middle=sx[i,peak]; right=sx[i,peak+1]
delta=.5*(left-right)/(left-2*middle+right)
sf['fft_peak_bpm']=peak*20/256*60
sf['quadratic_peak_bpm']=(peak+delta)*20/256*60
sf.to_csv(root/'synthetic_spectral_baseline_predictions.csv',index=False)
baseline=[]
for split,f in sf.groupby('split'):
    for method in ['fft_peak_bpm','quadratic_peak_bpm']:
        a=f.label_bpm-f.groupby('sample_tag').label_bpm.transform('mean')
        b=f[method]-f.groupby('sample_tag')[method].transform('mean')
        baseline.append(dict(split=split,method=method,mae=float(abs(f.label_bpm-f[method]).mean()),
                             centered_rmse=float(np.sqrt(np.mean((a-b)**2))),centered_r=float(np.corrcoef(a,b)[0,1])))
pd.DataFrame(baseline).to_csv(root/'synthetic_spectral_baselines.csv',index=False)
(root/'local_validation.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-start,
    kaggle_hashes_verified=len(hashes),prediction_scores_verified=len(rows),synthetic_gate_pass=gate),indent=2))
print(summary.to_string(index=False));print(execution)
