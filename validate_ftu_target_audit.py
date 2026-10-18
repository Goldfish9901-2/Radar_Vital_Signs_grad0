"""Sub-minute algebra checks on synthetic complex signals; no fitting."""
import json,time
from pathlib import Path
import numpy as np
from src.data.audit_ftu_targets import process,increments,nrms,spectral_distance
from src.features.edacm import target_edacm_signal,select_target_bin_indices
start=time.perf_counter();rng=np.random.default_rng(42)
radar=(rng.normal(size=(512,2,3,4))+1j*rng.normal(size=(512,2,3,4))).astype(np.complex64)
bins,w,_=select_target_bin_indices(radar[:256],3)
a=process(radar[:256],bins,w);b=process(radar[128:384],bins,w)
original,_=target_edacm_signal(radar[:256]);np.testing.assert_array_equal(a['signal'],original)
np.testing.assert_allclose(a['fused_delta'][129:],b['fused_delta'][1:128],atol=0,rtol=0)
assert spectral_distance(a['fused'][129:]+17,a['fused'][129:])<1e-5
assert nrms(np.ones(3),np.ones(3))==0
out=Path('experiment_runs/ftu_target_audit')
(out/'local_preflight.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-start,
    original_pipeline_exact=True,fixed_overlap_increments_exact=True,phase_origin_invariance=True),indent=2))
print((out/'local_preflight.json').read_text())
