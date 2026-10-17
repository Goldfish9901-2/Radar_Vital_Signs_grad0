"""Pre-run checks only: no model fitting or final-set outcome inspection."""
import json,time
from pathlib import Path
import numpy as np
from src.training.ftu_spectrum_factorial import generate,PROTOCOL,transform,score
start=time.perf_counter();f,x=generate()
old=np.load('experiment_runs/ftu_record_learnability/synthetic_features.npz')['x_freq']
max_replay_delta=float(abs(x-old).max())
np.testing.assert_allclose(x,old,atol=2e-6,rtol=1e-5)
x=old
train=x[(f.split=='train').to_numpy()].astype(float)
mean=train.mean(0);scale=train.std(0)
band=(np.fft.rfftfreq(256,1/20)>=.5)&(np.fft.rfftfreq(256,1/20)<=2.75)
assert band.sum()==29
a=transform(x,mean,scale,'A');b=transform(x,mean,scale,'B')
c=transform(x,mean,scale,'C');d=transform(x,mean,scale,'D')
assert np.array_equal(c[:,band],a[:,band]) and not c[:,~band].any()
assert np.array_equal(d[:,band],b[:,band]) and not d[:,~band].any()
assert score(f,f.label_bpm.to_numpy())['rmse']==0
out=Path('experiment_runs/ftu_spectrum_factorial');out.mkdir(exist_ok=True)
(out/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
(out/'preflight.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-start,
    original_features_reused_exactly=True,replayed_max_abs_delta=max_replay_delta,retained_bins=29,mask_after_scaling=True,
    final_outcomes_not_inspected=True),indent=2))
print((out/'preflight.json').read_text())
