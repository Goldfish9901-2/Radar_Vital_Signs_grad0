"""Independent readout/reference/summary checks on frozen acquired outputs."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.signal import periodogram

ROOT = Path('experiment_runs/synchronized_real_pilot')
DATA = Path('tmp/sync_real_compact/sync_real_pilot')
if not DATA.exists(): DATA = ROOT


def main():
    began = time.perf_counter()
    protocol = json.loads((ROOT/'protocol.json').read_text())
    assert hashlib.sha256(Path('run_synchronized_real_pilot.py').read_bytes()).hexdigest() == protocol['runner_sha256']
    source_hashes = json.loads((ROOT/'source_hashes.json').read_text())
    for name, digest in source_hashes.items():
        assert hashlib.sha256((ROOT/'sources'/name).read_bytes()).hexdigest() == digest
    frame = pd.read_csv(DATA/'window_predictions.csv')
    official = pd.read_csv(DATA/'record_metrics.csv')
    largest_ref, largest_peak, largest_metric, checked = 0., 0., 0., 0
    for (pid, scenario), group in frame.groupby(['subject', 'scenario']):
        member = next(m for m in protocol['members'] if pid in m['name'] and scenario in m['name'])
        stem = Path(member['name']).stem
        lead = protocol['ecg_primary_lead'][pid]
        beats = pd.read_csv(DATA/f'{stem}_lead{lead}_beats.csv').beat_time_s.to_numpy()
        phase_path = DATA/f'{stem}_phase20.npz'
        if not phase_path.exists(): phase_path = ROOT/'signals'/phase_path.name
        phase = np.load(phase_path)['phase']
        for row in group.itertuples():
            if np.isfinite(row.reference):
                start, end = row.start_s+row.reference_offset_s, row.end_s+row.reference_offset_s
                target = 60*(np.interp(end, beats, np.arange(len(beats)))-np.interp(start, beats, np.arange(len(beats))))/(end-start)
                largest_ref = max(largest_ref, abs(target-row.reference))
                assert abs(target-row.reference) < 1e-8
            first = round(row.start_s*20)
            trace = phase[first:first+512]
            axis = np.arange(512)-255.5
            detrended = trace-trace.mean()-axis*np.sum(axis*trace)/np.sum(axis**2)
            frequencies, power = periodogram(detrended, fs=20, window=np.hanning(513)[:-1], nfft=4096, detrend=False)
            allowed = np.flatnonzero((frequencies*60 >= 45)&(frequencies*60 <= 150))
            predicted = frequencies[allowed[np.argmax(power[allowed])]]*60
            largest_peak = max(largest_peak, abs(predicted-row.prediction))
            assert abs(predicted-row.prediction) < 1e-8
            checked += 1
        for start, subset in group.groupby('start_s'):
            assert len(subset) == 3
            assert np.all(subset.common_support == np.isfinite(subset.reference).all())
    for row in official.itertuples():
        group = frame[(frame.subject == row.subject)&(frame.scenario == row.scenario)&(frame.reference_offset_s == row.reference_offset_s)]
        valid = group[group.radar_valid & np.isfinite(group.reference)].sort_values('start_s')
        assert len(valid) == row.windows_scored
        if len(valid) < 3: continue
        t, y, p = (valid[k].to_numpy() for k in ['center_s', 'reference', 'prediction'])
        x = t-t.mean()
        yr = y-y.mean()-x*np.sum(x*y)/np.sum(x*x)
        pr = p-p.mean()-x*np.sum(x*p)/np.sum(x*x)
        values = {'mae': np.mean(abs(p-y)), 'mean_oracle_mae': np.mean(abs(y-y.mean())),
                  'median_oracle_mae': np.mean(abs(y-np.median(y))),
                  'trend_oracle_mae': np.mean(abs(yr)), 'detrended_rmse': np.sqrt(np.mean((pr-yr)**2)),
                  'trend_detrended_rmse': np.sqrt(np.mean(yr**2))}
        for name, value in values.items():
            difference = abs(value-getattr(row, name))
            largest_metric = max(largest_metric, difference)
            assert difference < 1e-8
    expected = {Path(m['name']).name for m in protocol['members']}
    acquisition = json.loads((DATA/'acquisition.json').read_text())
    assert {Path(x['name']).name for x in acquisition} == expected
    execution = json.loads((DATA/'execution.json').read_text())
    result = {'elapsed_seconds': time.perf_counter()-began, 'window_readouts_checked': checked,
              'maximum_reference_error_bpm': largest_ref, 'maximum_fft_error_bpm': largest_peak,
              'maximum_summary_error': largest_metric, 'completed_records': execution['records_completed'],
              'failed_records': execution['records_failed'], 'source_and_protocol_hash_verified': True,
              'source_snapshots_verified': len(source_hashes),
              'reference_detection_independently_adjudicated': False, 'clock_error_bound_verified': False}
    (ROOT/'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
