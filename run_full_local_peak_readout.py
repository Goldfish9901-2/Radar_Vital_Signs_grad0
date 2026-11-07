"""Frozen single-factor full-spectrum local-maximum readout comparison."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path('experiment_runs/full_local_peak_readout')
PARENT = Path('experiment_runs/real_band_truncation')
SYN = Path('experiment_runs/ellipse_readout_controls')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def select(power, bpm):
    peaks, props = signal.find_peaks(power, prominence=(None, None))
    keep = (bpm[peaks] >= 45) & (bpm[peaks] <= 150)
    candidates = peaks[keep]
    if not len(candidates):
        return np.nan, dict(candidate_count=0)
    index = int(candidates[np.argmax(power[candidates])])
    prominence = float(props['prominences'][np.flatnonzero(peaks == index)[0]])
    return float(bpm[index]), dict(candidate_count=len(candidates), selected_bin=index,
        selected_power=float(power[index]), prominence_power=prominence,
        prominence_fraction=prominence/power[index],
        selected_amplitude_rad=2*np.sqrt(power[index])/256,
        power_over_band_median=power[index]/np.median(power[(bpm >= 45) & (bpm <= 150)]))


def summarize(group, label):
    group = group.sort_values('start_s')
    ref = group.reference.to_numpy()
    result = dict(record=group.record.iloc[0], stratum=label, windows=len(group))
    for name, col in [('original', 'prediction'), ('candidate', 'new_prediction')]:
        pred = group[col].to_numpy()
        valid = np.isfinite(pred) & np.isfinite(ref)
        # Preserve original adjacency even for diagnostic subsets.
        adjacent = valid[1:] & valid[:-1] & (np.abs(np.diff(group.start_s)-2) < 1e-8)
        residual = np.diff(pred)-np.diff(ref)
        result.update({name+'_outputs': int(valid.sum()),
            name+'_mae_bpm': float(np.mean(np.abs(pred[valid]-ref[valid]))) if valid.any() else None,
            name+'_over5': int((np.abs(pred[valid]-ref[valid]) > 5).sum()),
            name+'_change_pairs': int(adjacent.sum()),
            name+'_change_rmse_bpm': float(np.sqrt(np.mean(residual[adjacent]**2))) if adjacent.any() else None})
    finite = np.isfinite(group.new_prediction)
    delta = (abs(group.new_prediction-group.reference)-abs(group.prediction-group.reference)).to_numpy()
    result.update(improved=int((delta < -1e-10).sum()), worsened=int((delta > 1e-10).sum()),
        equal=int((abs(delta) <= 1e-10).sum()), no_output=int((~finite).sum()),
        changed_predictions=int((finite & (group.new_prediction != group.prediction)).sum()),
        original_mae_on_candidate_support=float(abs(group.loc[finite, 'prediction']-group.loc[finite, 'reference']).mean()))
    return result


def main():
    began = time.perf_counter()
    assert not (ROOT/'windows.csv').exists(), 'Preserve completed outputs'
    tracked = ['run_full_local_peak_readout.py', (ROOT/'protocol.json').as_posix(), (ROOT/'source_hashes.json').as_posix()]
    for name in tracked:
        committed = subprocess.run(['git', 'show', 'HEAD:'+name], check=True, capture_output=True).stdout
        # Git stores LF while Windows text writes may use CRLF.
        assert committed.replace(b'\r\n', b'\n') == Path(name).read_bytes().replace(b'\r\n', b'\n'), name
    frozen = subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    for name, expected in json.loads((ROOT/'source_hashes.json').read_text()).items():
        assert digest(Path(name)) == expected, name
    # Select entirely from radar spectra before joining existing ECG annotations.
    radar = pd.read_csv(PARENT/'radar/windows.csv')
    rows = []
    for record, group in radar.groupby('record', sort=False):
        archive = np.load(PARENT/'radar'/f'{record}_spectra.npz')
        assert np.allclose(archive['window_starts_s'], group.start_s, atol=0, rtol=0)
        bpm = archive['bpm']
        for row, power in zip(group.itertuples(index=False), archive['power']):
            prediction, diag = select(power, bpm)
            original = float(bpm[np.flatnonzero((bpm >= 45) & (bpm <= 150))[np.argmax(power[(bpm >= 45) & (bpm <= 150)])]])
            assert original == row.prediction
            rows.append(dict(record=record, start_s=row.start_s, prediction=original,
                winner_kind=row.winner_kind, new_prediction=prediction,
                amplitude_relative_to_original=2*np.sqrt(diag.get('selected_power', np.nan))/256/row.selected_amplitude_rad,
                **diag))
    windows = pd.DataFrame(rows)
    ref = pd.read_csv(PARENT/'annotated_windows.csv')
    windows = windows.merge(ref[['record', 'start_s', 'reference', 'registered_focus']], on=['record', 'start_s'], validate='one_to_one')
    windows['original_error_bpm'] = abs(windows.prediction-windows.reference)
    windows['new_error_bpm'] = abs(windows.new_prediction-windows.reference)
    windows['error_delta_bpm'] = windows.new_error_bpm-windows.original_error_bpm
    windows['cut_flank'] = windows.winner_kind.str.contains('cut_')
    unchanged = windows[~windows.cut_flank]
    assert len(unchanged) == 136 and (unchanged.prediction == unchanged.new_prediction).all()
    assert windows.cut_flank.sum() == 20
    windows.to_csv(ROOT/'windows.csv', index=False)
    windows[windows.cut_flank].to_csv(ROOT/'cut_flank_windows.csv', index=False)
    windows[windows.registered_focus].to_csv(ROOT/'registered_focus.csv', index=False)
    summaries = []
    for _, group in windows.groupby('record', sort=False):
        for label, subset in [('all', group), ('cut_flank', group[group.cut_flank]), ('genuine', group[~group.cut_flank])]:
            summaries.append(summarize(subset, label))
    pd.DataFrame(summaries).to_csv(ROOT/'summary.csv', index=False)
    synthetic = pd.read_csv(SYN/'windows.csv')
    synthetic = synthetic[synthetic.heartbeat == 'present'].copy()
    bpm = np.fft.rfftfreq(4096, 1/20)*60
    results = []
    for tag, group in synthetic.groupby('tag', sort=False):
        data = np.load(SYN/'signals'/f'{tag}.npz')
        for row in group.itertuples(index=False):
            power = data['power'][['truth', 'recovered'].index(row.source), [30., 32., 34.].index(row.start_s)]
            value, diag = select(power, bpm)
            results.append(dict(tag=tag, source=row.source, start_s=row.start_s,
                original=row.prediction_bpm, candidate=value, error_to72=abs(value-72), **diag))
    syn = pd.DataFrame(results)
    assert len(syn) == 180 and (syn.original == syn.candidate).all() and (syn.error_to72 <= 1.171875).all()
    syn.to_csv(ROOT/'synthetic72.csv', index=False)
    save_json(ROOT/'execution.json', dict(freeze_commit=frozen, elapsed_seconds=time.perf_counter()-began,
        environment='local uv-managed .venv', real_windows=len(windows), synthetic_windows=len(syn),
        unchanged_genuine=136, no_training=True, new_subjects_accessed=False))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    main()
