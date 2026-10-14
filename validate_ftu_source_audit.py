"""Deterministic negative-control invariants; CPU, synthetic arrays, no training."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from src.training.audit_feature_controls import control_plan, perturb, frequency_from_time
from src.training.audit_predictions import metrics
from src.features.representations import frequency_features
from src.data.audit_ftu_continuous import power_windows

start = time.perf_counter()
rng = np.random.default_rng(1)
t = rng.normal(size=(6,7,256)).astype(np.float32)
f = frequency_from_time(t)
plan = control_plan(['p1']*3+['p2']*3,256,129,11)
assert np.all((plan['donors'] < 3) != (np.arange(6) < 3))
before = t.copy()
shuffled, untouched = perturb(t,f,plan,'time_shuffle_time_only')
assert np.array_equal(untouched,f) and np.array_equal(before,t)
assert np.array_equal(np.sort(shuffled,axis=-1),np.sort(t,axis=-1))
_, fs = perturb(t,f,plan,'time_shuffle_reencoded')
assert np.allclose(fs,frequency_from_time(shuffled))
dt,df = perturb(t,f,plan,'replace_both')
assert np.array_equal(dt,t[plan['donors']]) and np.array_equal(df,f[plan['donors']])
ft,ff = perturb(t,f,plan,'freq_shuffle_freq_only')
assert np.array_equal(ft,t) and np.array_equal(np.sort(ff,axis=-1),np.sort(f,axis=-1))
for i in range(len(t)):
    expected,_,_ = frequency_features(t[i])
    expected = (expected-expected.mean())/expected.std()
    assert np.allclose(f[i],expected,atol=2e-6)
constant = metrics(pd.DataFrame({'label_bpm':[60.,70.,80.], 'pred_bpm':[70.]*3}), {'train_mean':70.})
assert constant['r_var'] == 0 and constant['pearson_r'] is None
assert constant['gain_vs_train_mean'] == 0
tracking = metrics(pd.DataFrame({'label_bpm':[60.,70.,80.], 'pred_bpm':[60.,70.,80.]}), {})
assert tracking['r_var'] == 1 and tracking['pearson_r'] == 1 and tracking['mae'] == 0
try:
    control_plan(['p1']*6,256,129,11)
except ValueError: pass
else: raise AssertionError('Single-subject donor pool was accepted')
wave = np.sin(2*np.pi*1.25*np.arange(1200)/20)
power,freq,starts = power_windows(wave)
assert power.shape == (1,8,129) and np.array_equal(starts,np.arange(8)*128)
assert np.all(freq[np.argmax(power[0],axis=-1)]*60 == 75.)
result = {'status':'passed','elapsed_sec':time.perf_counter()-start,
    'checks':['cross-subject donors','branch isolation','paired donor identity','permutation distribution',
              'input immutability','coherent FFT regeneration','constant versus tracking metrics',
              'single-subject rejection','continuous 75 bpm ridge and window time grid']}
out = Path('experiment_runs/ftu_source_audit_2026-10-13/control_invariants.json')
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
