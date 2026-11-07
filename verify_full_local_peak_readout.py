"""Independent catalogue/periodogram verification and descriptive plots."""
from pathlib import Path
import hashlib
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path('experiment_runs/full_local_peak_readout')
PARENT = Path('experiment_runs/real_band_truncation')
SYN = Path('experiment_runs/ellipse_readout_controls')


def main():
    started = time.perf_counter()
    checks = json.loads((ROOT/'source_hashes.json').read_text())
    for name, expected in checks.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    windows = pd.read_csv(ROOT/'windows.csv')
    catalogue = pd.read_csv(PARENT/'radar/peaks.csv')
    candidates = catalogue[catalogue.full_local & catalogue.in_band]
    for row in windows.itertuples():
        group = candidates[(candidates.record == row.record) & (candidates.start_s == row.start_s)]
        best = group.sort_values(['power', 'bin'], ascending=[False, True]).iloc[0]
        assert best.bpm == row.new_prediction
        assert len(group) == row.candidate_count
        assert np.isclose(best.prominence_power, row.prominence_power, rtol=1e-12)
    summary = pd.read_csv(ROOT/'summary.csv')
    max_diff = 0.
    for row in summary.itertuples():
        g = windows[windows.record == row.record].sort_values('start_s')
        if row.stratum != 'all':
            g = g[g.cut_flank if row.stratum == 'cut_flank' else ~g.cut_flank]
        for name, col in [('original', 'prediction'), ('candidate', 'new_prediction')]:
            err = g[col].to_numpy()-g.reference.to_numpy()
            mae = np.mean(abs(err))
            adjacency = np.isclose(np.diff(g.start_s), 2, rtol=0, atol=1e-8)
            change = np.sqrt(np.mean(np.diff(err)[adjacency]**2))
            max_diff = max(max_diff, abs(mae-getattr(row, name+'_mae_bpm')),
                           abs(change-getattr(row, name+'_change_rmse_bpm')))
            assert int((abs(err) > 5).sum()) == getattr(row, name+'_over5')
            assert adjacency.sum() == getattr(row, name+'_change_pairs')
        delta = abs(g.new_prediction-g.reference)-abs(g.prediction-g.reference)
        assert (delta < -1e-10).sum() == row.improved
        assert (delta > 1e-10).sum() == row.worsened
        assert (abs(delta) <= 1e-10).sum() == row.equal
    assert max_diff < 1e-10
    assert len(windows) == 156 and windows.cut_flank.sum() == 20
    assert (windows.loc[~windows.cut_flank, 'prediction'] == windows.loc[~windows.cut_flank, 'new_prediction']).all()
    assert windows.new_prediction.notna().all()
    focus = windows[windows.registered_focus].sort_values('start_s')
    low = focus[focus.cut_flank]
    assert low.new_prediction.tolist() == [67.08984375, 67.96875]
    # Reconstruct synthetic spectra independently from archived phase20.
    synthetic = pd.read_csv(ROOT/'synthetic72.csv')
    max_power_relative = 0.
    for tag, group in synthetic.groupby('tag'):
        archive = np.load(SYN/'signals'/f'{tag}.npz')
        for row in group.itertuples():
            source = ['truth', 'recovered'].index(row.source)
            start = round(row.start_s*20)
            freq, density = signal.periodogram(archive['phase20'][source, start:start+512],
                fs=20, window='hann', nfft=4096, detrend='linear', scaling='spectrum', return_onesided=True)
            power = density*256**2/2
            power[[0, -1]] *= 2
            original = archive['power'][source, [30., 32., 34.].index(row.start_s)]
            max_power_relative = max(max_power_relative, float(np.max(abs(power-original))/np.max(original)))
            # Strict neighbour extrema suffice on these nonplateau spectra.
            indices = np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] > power[2:]))+1
            bpm = freq*60
            indices = indices[(bpm[indices] >= 45) & (bpm[indices] <= 150)]
            chosen = bpm[indices[np.argmax(power[indices])]]
            assert chosen == row.candidate == row.original and abs(chosen-72) <= 1.171875
    assert max_power_relative < 1e-12
    (ROOT/'plots').mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for n, (record, g) in enumerate(windows.groupby('record', sort=False)):
        ax, err = axes[n]
        x = g.start_s+12.8
        ax.plot(x, g.reference, color='black', label='ECG window HR')
        ax.plot(x, g.prediction, color='tab:orange', label='Original cropped argmax', alpha=.75)
        ax.plot(x, g.new_prediction, color='tab:blue', label='Full local peak', alpha=.85)
        ax.scatter(x[g.cut_flank], g.new_prediction[g.cut_flank], s=22, color='tab:blue')
        ax.set(title=record, ylabel='HR (bpm)', xlabel='Window centre (s)')
        err.axhline(0, color='gray', linewidth=.8)
        err.plot(x, g.error_delta_bpm, color='tab:purple')
        err.set(title='Change in absolute error (new - original)', ylabel='Error difference (bpm)', xlabel='Window centre (s)')
        ax.legend(fontsize=8)
    fig.savefig(ROOT/'plots/comparison.png', dpi=160)
    plt.close(fig)
    result = dict(elapsed_seconds=time.perf_counter()-started, source_files_verified=len(checks),
        real_windows_verified=156, unchanged_genuine=136, cut_flank=20, synthetic_windows_verified=180,
        independent_method='Previously frozen full-spectrum peak catalogue; direct error differences; synthetic phase periodogram and strict-neighbour extrema',
        max_metric_difference=max_diff, max_synthetic_relative_power_difference=max_power_relative,
        focus_second_peaks_preserved=True, original_archives_unchanged=True)
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    manifest = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in ROOT.rglob('*') if path.is_file() and path.name != 'artifact_manifest.json'}
    (ROOT/'artifact_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
