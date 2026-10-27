"""Fixed-position ECG snapshots and two-lead disagreement; no detector changes."""
from pathlib import Path
import hashlib
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('experiment_runs/synchronized_real_pilot')
RAW = Path('tmp/sync_real_results/sync_real_pilot')


def main():
    began = time.perf_counter()
    records = json.loads((RAW/'records.json').read_text())
    files, rows = [], []
    fig, axes = plt.subplots(3, 4, figsize=(17, 9))
    for row, pid in enumerate(['GDN0002', 'GDN0007', 'GDN0006']):
        for scenario_index, scenario in enumerate(['Resting', 'Valsalva']):
            record = next(r for r in records if r['subject'] == pid and r['scenario'] == scenario)
            path = next(RAW.glob(f'{pid}_*_{scenario}_reference_ecg.npz'))
            files.append({'file': str(path).replace('\\', '/'), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size})
            with np.load(path, allow_pickle=False) as data:
                fs, origin = float(data['fs']), float(data['start_s'])
                first = record['score_start_s']+60
                cut = np.arange(round((first-origin)*fs), round((first+5-origin)*fs))
                times = origin+cut/fs
                stem = path.stem.replace('_reference_ecg', '')
                beats = [pd.read_csv(RAW/f'{stem}_lead{i}_beats.csv').beat_time_s.to_numpy() for i in [1, 2]]
                for lead in [1, 2]:
                    values = data[f'ecg{lead}']
                    assert len(values)/fs >= 184-1/fs
                    ax = axes[row, 2*scenario_index+lead-1]
                    ax.plot(times-first, values[cut], linewidth=.65)
                    for peak in beats[lead-1][(beats[lead-1] >= first)&(beats[lead-1] < first+5)]:
                        ax.axvline(peak-first, color='tab:red', alpha=.5, linewidth=.8)
                    ax.set_title(f'{pid} {scenario} ECG{lead}'); ax.set_xlabel('Time (s)'); ax.set_ylabel('mV'); ax.grid(alpha=.2)
                score_beats = [b[(b >= record['score_start_s'])&(b < record['score_end_s'])] for b in beats]
                fractions = [float(np.mean(np.min(abs(a[:, None]-b[None, :]), axis=1) <= .05))
                             for a, b in [(score_beats[0], score_beats[1]), (score_beats[1], score_beats[0])]]
                rows.append({'subject': pid, 'scenario': scenario, 'ECG1_peaks': len(score_beats[0]),
                             'ECG2_peaks': len(score_beats[1]), 'ECG1_matched_within_50ms_fraction': fractions[0],
                             'ECG2_matched_within_50ms_fraction': fractions[1]})
    fig.suptitle('Raw ECG with frozen automatic R peaks: first 5s at scoring start +60s; no edits / no adjudication')
    fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig(ROOT/'ecg_reference_snapshots.png', dpi=150); plt.close(fig)
    pd.DataFrame(rows).to_csv(ROOT/'reference_qa.csv', index=False)
    (ROOT/'local_ecg_preservation.json').write_text(json.dumps({'files': files, 'elapsed_seconds': time.perf_counter()-began,
        'raw_ecg_changed': False, 'detector_changed': False, 'manual_adjudication': False,
        'matching_tolerance_is_diagnostic_not_accuracy_bound': True}, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__': main()
