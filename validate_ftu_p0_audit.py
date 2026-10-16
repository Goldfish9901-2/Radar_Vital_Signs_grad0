"""Independent checks on saved P0 numerical evidence (no training)."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from src.data.audit_ftu_p0 import digest, write_json


def main():
    start=time.perf_counter(); out=Path('experiment_runs/ftu_p0_audit_2026-10-14')
    summary=pd.read_csv(out/'mode_summary.csv'); modes=pd.read_csv(out/'modes.csv')
    checks=[]
    with np.load(out/'mode_arrays.npz') as saved:
        for row in summary.itertuples():
            x=saved[row.case+'__'+row.variant].astype(np.float64)
            identical=all(np.array_equal(x[0], a) for a in x[1:])
            assert identical==row.all_arrays_exactly_equal
            corr=np.corrcoef(x); pairs=corr[np.triu_indices(7,1)]
            assert abs(pairs.min()-row.min_pair_correlation)<1e-10
            fft=abs(np.fft.rfft((x-x.mean(axis=1,keepdims=True))*np.hanning(x.shape[1]),axis=1))**2
            f=np.fft.rfftfreq(x.shape[1],.05)
            centroid=(fft*f).sum(axis=1)/fft.sum(axis=1)
            table=modes[(modes.case==row.case)&(modes.variant==row.variant)].sort_values('mode')
            assert np.allclose(centroid,table.spectral_centroid_hz,atol=1e-12)
            assert np.allclose(x.std(axis=1),table.weight,atol=1e-6)
            checks.append(dict(case=row.case,variant=row.variant,identical=identical))
    old=summary[summary.variant=='historical']
    changed=summary[summary.variant=='sequential_sum_diagnostic']
    assert old.all_arrays_exactly_equal.all() and (old.total_clipped_coefficients>0).all()
    assert not changed.all_arrays_exactly_equal.any() and (changed.total_clipped_coefficients==0).all()
    local=json.loads((out/'local_summary.json').read_text())
    for path, expected in local['inputs'].items(): assert digest(path)==expected
    remote=out/'validation.json'
    if remote.exists():
        r=json.loads(remote.read_text())
        assert r['manifest_rows']==r['mode_rows']
        assert not r['unmatched_sources']
        rows=pd.read_csv(out/'exported_window_modes.csv')
        assert len(rows)==3520 and rows.all_arrays_exactly_equal.all()
        assert rows.frequency_arrays_identical.all() and (rows.numerical_rank==1).all()
        assert (rows.min_weight==1).all() and (rows.max_weight==1).all()
        examples=json.loads((out/'exported_example_metadata.json').read_text())
        with np.load(out/'exported_examples.npz') as z:
            assert len(examples)==24 and len(z.files)==48
            for example in examples:
                for branch in ('x_time','x_freq'):
                    x=z[example['key']+'__'+branch]
                    assert all(np.array_equal(x[0], a) for a in x[1:])
                x=z[example['key']+'__x_time'].astype(np.float64)
                p=abs(np.fft.rfft((x-x.mean(axis=1,keepdims=True))*np.hanning(x.shape[1]),axis=1))**2
                f=np.fft.rfftfreq(x.shape[1],.05)
                centers=(p*f).sum(axis=1)/p.sum(axis=1)
                assert np.allclose(centers,[m['spectral_centroid_hz'] for m in example['modes']],atol=1e-12)
                assert example['feature_meta']['mode_weights']==[1.]*7
        timing=pd.read_csv(out/'raw_timing_audit.csv')
        used=timing[timing.saved_window_count.notna()]
        assert len(used)==440 and used.saved_window_count.sum()==3520
        assert used.raw_label_reproduction_max_bpm.max()<1e-4
        assert (used.nominal_same_start_min_window_coverage==1).all()
        assert (timing.configured_fs_hz.dropna()==20).all()
        assert timing.configured_fs_hz.notna().sum()==428
        assert (used.frame_rate_basis=='historical_pipeline_constant_unverified').sum()==28
    write_json(out/'numerical_validation.json',dict(passed=True,checks=checks,
        elapsed_sec=time.perf_counter()-start, source_sha256=digest(__file__),
        meaning='Evidence integrity verified; this does not mean the P0 scientific gates passed.'))
    print('P0 evidence validation passed; mode gate remains FAIL.')


if __name__=='__main__': main()
