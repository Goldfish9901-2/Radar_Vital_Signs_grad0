"""Local identity and metric validation only; never trains a model."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from src.training.ftu_tcn_protocol import donor_indices
from src.training.audit_predictions import metrics
start=time.perf_counter()
frame=pd.read_csv('experiment_runs/ftu_timebase_edacm_v2/training_exports_ftu_edacm_v2/manifest.csv')
test=frame[frame.split=='test'].reset_index(drop=True)
identity=test[['group_key','sample_tag','window_start']]
donors=donor_indices(identity)
assert np.array_equal(donors[donors],np.arange(704))
assert np.array_equal(test.window_start.to_numpy(),test.window_start.to_numpy()[donors])
assert (test.group_key.to_numpy()!=test.group_key.to_numpy()[donors]).all()
# Between-record level recognition must not masquerade as temporal tracking.
toy=pd.DataFrame(dict(record=np.repeat([0,1],8),label_bpm=np.r_[np.arange(8)+60,np.arange(8)+100],
                     pred_bpm=np.r_[np.full(8,63.5),np.full(8,103.5)]))
assert metrics(toy,{})['pearson_r']>.99
for col in ['label_bpm','pred_bpm']:toy[col]-=toy.groupby('record')[col].transform('mean')
assert metrics(toy,{})['pearson_r'] is None and metrics(toy,{})['r_var']==0
result=dict(test_windows=len(test),swap_pairs=len(test)//2,labels_not_used=True,
            permutation_involution=True,record_centering_sanity=True,elapsed_sec=time.perf_counter()-start)
Path('tmp/ftu_tcn_protocol_validation.json').write_text(json.dumps(result,indent=2))
print(result)
