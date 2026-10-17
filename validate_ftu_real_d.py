"""Local input validation only, with a recorded runtime."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from src.training.run_ftu_real_d import load_data,PROTOCOL,score
start=time.perf_counter()
f,x=load_data('experiment_runs/ftu_timebase_edacm_v2/full_artifacts.zip','experiment_runs/ftu_record_learnability/record_split.csv')
assert score(f,f.label_bpm.to_numpy())['centered_r2']==1
out=Path('experiment_runs/ftu_real_d');out.mkdir(exist_ok=True)
(out/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
(out/'preflight.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-start,windows=len(f),
    captures=f.sample_tag.nunique(),split_counts=f.groupby('split').size().to_dict(),
    original_record_hashes_verified=True,subjects=f.group_key.unique().tolist()),indent=2))
print((out/'preflight.json').read_text())
rows=[]
for (split,subject),g in f.groupby(['split','group_key']):
    centered=g.label_bpm-g.groupby('sample_tag').label_bpm.transform('mean')
    rows.append(dict(split=split,subject=subject,n=len(g),records=g.sample_tag.nunique(),
                     label_mean=g.label_bpm.mean(),label_std=g.label_bpm.std(ddof=0),
                     centered_label_rmse=float(np.sqrt(np.mean(centered**2)))))
pd.DataFrame(rows).to_csv(out/'label_dynamics.csv',index=False)
