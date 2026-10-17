"""Local data/selection sanity tests; no regression or neural fitting."""
import json,time
from pathlib import Path
import numpy as np
from src.training.ftu_position_protocol import load_arrays,choose_record,artificial_spectra,PROTOCOL
start=time.perf_counter()
f,xt,xf,_=load_arrays('experiment_runs/ftu_timebase_edacm_v2/full_artifacts.zip')
tag,ids=choose_record(f)
assert len(np.unique(xf[ids],axis=0))==8
for kind in PROTOCOL['A']['shapes']:
    a=artificial_spectra(kind)
    assert np.array_equal(a.argmax(axis=1),np.arange(129))
    np.testing.assert_allclose(a.mean(axis=1),0,atol=1e-6)
    np.testing.assert_allclose(a.std(axis=1),1,atol=1e-6)
assert np.array_equal(np.sort(artificial_spectra('single_bin'),axis=1),
                      np.repeat(np.sort(artificial_spectra('single_bin')[0])[None],129,axis=0))
result=dict(selected_record=tag,labels=f.iloc[ids].label_heart_rate.tolist(),
    population_hr_std=float(f.iloc[ids].label_heart_rate.std(ddof=0)),
    windows=8,distinct_frequency_vectors=8,synthetic_argmax_and_normalization=True,
    elapsed_sec=time.perf_counter()-start)
Path('tmp/ftu_position_local_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
