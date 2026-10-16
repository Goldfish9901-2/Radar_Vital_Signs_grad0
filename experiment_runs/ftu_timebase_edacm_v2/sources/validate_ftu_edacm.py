"""Small deterministic export integration test, using no real HR to select parameters."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
import pandas as pd
from src.data.ftu_timebase import frame_times, labels_at_frames
from src.features.representations import frequency_features

started = time.perf_counter()
with tempfile.TemporaryDirectory(dir="tmp") as temp:
    root = Path(temp).resolve()
    src = root / "exports/FTU"
    src.mkdir(parents=True)
    t = np.arange(512)/20
    radar = np.broadcast_to(np.exp(1j * .1 * np.sin(2*np.pi*1.2*t))[:,None,None,None],
                            (512,2,2,2)).copy()
    np.savez(src / "record.npz", radar=radar, time=[1.,12.,25.], heart_rate=[60.,71.,84.])
    pd.DataFrame([dict(status="ok",sample_tag="record",npz_path=str(src/"record.npz"),
                       participant_id="1")]).to_csv(src / "manifest.csv",index=False)
    frozen = root / "split.csv"
    pd.DataFrame([dict(group_key="FTU/participant/1",split="test")]).to_csv(frozen,index=False)
    out=root/"output"
    subprocess.run([sys.executable,"-m","src.data.build_training_dataset","--exports-dir",str(src.parent),
        "--output-dir",str(out),"--datasets","FTU","--representation","edacm_only",
        "--normalize","none","--split-manifest",str(frozen),"--split-mode","stable_grouped"],check=True,
        stdout=subprocess.DEVNULL)
    rows=pd.read_csv(out/"manifest.csv")
    assert len(rows)==3 and set(rows.split)=={"test"}
    expected=labels_at_frames([1,12,25],[60,71,84],frame_times(512,20,radar_start_on_reference_clock_s=0))
    for row in rows.itertuples():
        with np.load(row.window_path) as z:
            np.testing.assert_allclose(z['label_series_heart_rate'],expected[row.window_start:row.window_end],atol=1e-5)
            assert z['x_time'].shape==(1,256) and z['x_freq'].shape==(1,129)
            np.testing.assert_allclose(z['x_freq'],frequency_features(z['x_time'])[0],atol=1e-6)
    assert np.isnan(expected[:20]).all() and np.isnan(expected[501:]).all()
    for bad in [[0,0],[1,0]]:
        try: labels_at_frames(bad,[60,70],t)
        except ValueError: pass
        else: raise AssertionError("invalid timestamp accepted")
result=dict(passed=True,elapsed_sec=time.perf_counter()-started,
            checks=["formal export", "frozen group", "physical time", "boundary NaN", "dual branches"])
Path("tmp/ftu_edacm_local_validation.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result))
