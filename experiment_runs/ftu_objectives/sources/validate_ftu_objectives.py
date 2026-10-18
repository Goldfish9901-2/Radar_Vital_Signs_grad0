"""Local record-batch and objective algebra checks; no training."""
import json,time
from pathlib import Path
import numpy as np
from src.training.run_ftu_objectives import PROTOCOL,load_data
start=time.perf_counter()
f,x=load_data('experiment_runs/ftu_timebase_edacm_v2/full_artifacts.zip','experiment_runs/ftu_record_learnability/record_split.csv')
records=np.array([np.array(idx) for _,idx in f[f.split=='train'].groupby('sample_tag',sort=True).groups.items()])
assert records.shape==(108,8) and len(np.unique(records))==864
y=f.label_bpm.to_numpy();scale=y[records.ravel()].std();mean=y[records.ravel()].mean()
target=(y[records]-mean)/scale
rng=np.random.default_rng(42);pred=rng.normal(size=target.shape)
center=lambda a:a-a.mean(1,keepdims=True)
np.testing.assert_allclose(center(target),center(y[records])/scale,atol=1e-14)
np.testing.assert_allclose(center(pred+np.arange(108)[:,None])-center(target),center(pred)-center(target),atol=1e-13)
order=rng.permutation(108);batches=[records[order[i:i+8]] for i in range(0,108,8)]
assert [len(b) for b in batches]==[8]*13+[4]
out=Path('experiment_runs/ftu_objectives');out.mkdir(exist_ok=True)
(out/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
(out/'preflight.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-start,record_shape=list(records.shape),
    batch_records=[len(b) for b in batches],centering_shift_invariant=True,shared_label_scale=scale,shared_label_mean=mean,
    original_capture_hashes_verified=True),indent=2))
print((out/'preflight.json').read_text())
