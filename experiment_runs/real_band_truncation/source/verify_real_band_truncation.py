"""Independent checks and fixed visual summaries of the two-record spectrum audit."""
from pathlib import Path
import hashlib
import json
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


ROOT = Path('experiment_runs/real_band_truncation')
INPUT = Path('experiment_runs/synchronized_real_pilot_fixed')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    began = time.perf_counter()
    protocol = read(ROOT/'protocol.json')
    assert all(digest(Path(name)) == expected for name, expected in read(ROOT/'source_hashes.json').items())
    assert all(digest(ROOT/'radar'/name) == expected for name, expected in read(ROOT/'radar_manifest.json').items())
    windows = pd.read_csv(ROOT/'annotated_windows.csv')
    peaks = pd.read_csv(ROOT/'annotated_peaks.csv')
    frozen_peaks = pd.read_csv(ROOT/'radar/peaks.csv')
    frozen_windows = pd.read_csv(ROOT/'radar/windows.csv')
    pd.testing.assert_frame_equal(peaks[frozen_peaks.columns], frozen_peaks)
    pd.testing.assert_frame_equal(windows[frozen_windows.columns], frozen_windows)
    original = pd.read_csv(INPUT/'window_predictions.csv')
    original = original[original.reference_offset_s == 0]
    original_metrics = pd.read_csv(INPUT/'record_metrics.csv')
    groups = peaks.groupby(['record', 'start_s'])
    max_power_relative_error = 0.
    errors = []
    spectra = {}
    for record in protocol['records']:
        spectrum = dict(np.load(ROOT/'radar'/f'{record}_spectra.npz'))
        spectra[record] = spectrum
        phase = np.load(INPUT/'signals'/f'{record}_phase20.npz')['phase']
        table = windows[windows.record == record]
        assert len(table) == 78
        pid, _, scenario = record.split('_')
        source = original[(original.subject == pid) & (original.scenario == scenario)].sort_values('start_s')
        assert np.array_equal(table.start_s, source.start_s)
        assert np.array_equal(table.prediction, source.prediction)
        assert np.array_equal(table.reference, source.reference)
        bpm = spectrum['bpm']
        band = (bpm >= 45) & (bpm <= 150)
        band_bins = np.flatnonzero(band)
        gain = signal.windows.hann(512, sym=False).sum()
        for k, row in enumerate(table.itertuples(index=False)):
            first = round(row.start_s*20)
            trace = phase[first:first+512]
            _, density = signal.periodogram(trace, fs=20, window=signal.windows.hann(512, sym=False),
                                            nfft=4096, detrend='linear', scaling='spectrum')
            factor = np.full(2049, 2.)
            factor[[0, -1]] = 1
            reconstructed = density*gain**2/factor
            power = spectrum['power'][k]
            relative = float(np.max(abs(reconstructed-power))/np.max(power))
            max_power_relative_error = max(max_power_relative_error, relative)
            assert relative < 1e-12
            winner = int(band_bins[np.argmax(reconstructed[band])])
            assert bpm[winner] == row.prediction and winner == row.selected_bin
            genuine = set(signal.find_peaks(power)[0].tolist())
            group = groups.get_group((record, row.start_s))
            assert set(group.loc[group.full_local, 'bin']) == genuine
            genuine_sorted = sorted(genuine)
            prominence = signal.peak_prominences(power, genuine_sorted)[0]
            saved_local = group[group.full_local].sort_values('bin')
            errors.append(float(np.max(abs(saved_local.prominence_power.to_numpy()-prominence))))
            errors.append(float(np.max(abs(saved_local.prominence_fraction.to_numpy()-prominence/power[genuine_sorted]))))
            assert row.selected_is_full_local == (winner in genuine)
            amplitude = 2*np.sqrt(power)/gain
            errors.append(float(np.max(abs(group.amplitude_rad.to_numpy()-amplitude[group.bin.to_numpy()]))))
            median = np.median(power[band])
            qualified = (group.full_local & group.in_band & (group.relative_amplitude >= .1) &
                         (group.prominence_fraction >= .1) & (group.power_over_band_median >= 3))
            assert np.array_equal(qualified, group.radar_qualified)
            errors.append(float(np.max(abs(group.relative_amplitude-amplitude[group.bin]/amplitude[winner]))))
            errors.append(float(np.max(abs(group.power_over_band_median-power[group.bin]/median))))
            ordered_crop = group[group.in_band].sort_values(['power', 'bin'], ascending=[False, True])
            assert ordered_crop.crop_rank.tolist() == list(range(1, len(ordered_crop)+1))
            ordered_full = group[group.kind != 'crop_endpoint_only'].sort_values(['power', 'bin'], ascending=[False, True])
            assert ordered_full.full_rank.tolist() == list(range(1, len(ordered_full)+1))
            ordered_band = group[group.full_local & group.in_band].sort_values(['power', 'bin'], ascending=[False, True])
            assert ordered_band.genuine_band_rank.tolist() == list(range(1, len(ordered_band)+1))
            for ordered, prefix in [(ordered_crop, 'crop'), (ordered_full, 'full')]:
                main_peak, secondary = ordered.iloc[0], ordered.iloc[1]
                errors.append(abs(main_peak.amplitude_rad/secondary.amplitude_rad-getattr(row, prefix+'_main_second_amplitude_ratio')))
                errors.append(abs(main_peak.power/secondary.power-getattr(row, prefix+'_main_second_power_ratio')))
            if row.winner_kind == 'lower_cut_descending_flank':
                assert winner == band_bins[0] and winner not in genuine
                assert power[winner-1] > power[winner] >= power[winner+1]
                parent_index = int(np.where(bpm == row.outside_parent_bpm)[0][0])
                assert parent_index < winner and parent_index in genuine
                assert np.all(np.diff(power[parent_index:winner+1]) <= 0)
            if row.winner_kind == 'genuine_full_local_peak':
                assert winner in genuine
            matches = qualified & (abs(group.bpm-row.reference) <= 2.34375) & (abs(group.bpm-row.prediction) > 2.34375)
            assert matches.sum() == row.distinct_reference_candidate_count
            hidden = abs(row.prediction-row.reference) > 2.34375 and row.selected_boundary_region and matches.any()
            assert bool(hidden) == row.boundary_hidden_candidate
            errors.append(float(np.max(abs(group.absolute_distance_bpm-abs(group.bpm-row.reference)))))
            errors.append(abs(abs(row.prediction-row.reference)-row.absolute_error_bpm))
        valid = table.radar_valid & table.common_support & np.isfinite(table.reference)
        previous = original_metrics[(original_metrics.subject == pid) & (original_metrics.scenario == scenario) &
                                    (original_metrics.reference_offset_s == 0)].iloc[0]
        errors.append(abs(table.loc[valid, 'absolute_error_bpm'].mean()-previous.mae))
    assert len(windows) == 156 and len(peaks) == 13603 and windows.registered_focus.sum() == 4
    assert max(errors) < 1e-7, max(errors)
    plots = ROOT/'plots'
    plots.mkdir(exist_ok=True)
    for record in protocol['records']:
        data = spectra[record]
        table = windows[windows.record == record]
        amplitude = 2*np.sqrt(data['power'])/256
        log_amplitude = np.log10(np.maximum(amplitude, 1e-12))
        band = (data['bpm'] >= 45) & (data['bpm'] <= 150)
        times = table.center_s.to_numpy()
        vmin, vmax = float(log_amplitude.max()-6), float(log_amplitude.max())
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
        im = axes[0].pcolormesh(data['bpm'], times, log_amplitude, shading='nearest', vmin=vmin, vmax=vmax)
        axes[0].set(xlabel='Frequency (bpm)', ylabel='Window center (s)', title='Full original spectrum, 0-600 bpm')
        axes[0].axvline(45, color='white', ls=':', lw=.8)
        axes[0].axvline(150, color='white', ls=':', lw=.8)
        axes[1].pcolormesh(data['bpm'][band], times, log_amplitude[:, band], shading='nearest', vmin=vmin, vmax=vmax)
        axes[1].plot(table.prediction, times, 'r.', ms=3, label='Original winner')
        axes[1].plot(table.reference, times, color='white', lw=1, label='Existing ECG reference')
        for row in table[table.boundary_hidden_candidate].itertuples():
            axes[1].plot(row.matched_candidate_bpm, row.center_s, 'go', ms=4, fillstyle='none')
        axes[1].set(xlabel='Frequency (bpm)', title='Original 45-150 crop; green circles mark annotated candidates')
        axes[1].legend(fontsize=8)
        fig.suptitle(record)
        fig.colorbar(im, ax=axes, label='log10 coherent amplitude proxy (rad)', fraction=.025, pad=.03)
        fig.savefig(plots/f'{record}_all_windows.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    focus = windows[windows.registered_focus]
    fig, axes = plt.subplots(4, 2, figsize=(13, 13))
    for index, row in enumerate(focus.itertuples(index=False)):
        data = spectra[row.record]
        k = int(np.flatnonzero(np.isclose(data['window_starts_s'], row.start_s, atol=1e-8, rtol=0))[0])
        amplitude = 2*np.sqrt(data['power'][k])/256
        bpm = data['bpm']
        group = groups.get_group((row.record, row.start_s))
        left, right = axes[index]
        left.plot(bpm, np.maximum(amplitude, 1e-12))
        left.axvspan(45, 150, color='orange', alpha=.15)
        left.set(xlim=(0, 600), yscale='log', xlabel='Frequency (bpm)', ylabel='Amplitude proxy (rad)',
                 title=f'{row.start_s:.3f}s, full spectrum')
        right.plot(bpm, amplitude, lw=1)
        right.axvspan(35, 45, color='gray', alpha=.2)
        right.axvline(45, color='gray', ls=':')
        right.axvline(row.reference, color='green', ls='--', label=f'ECG {row.reference:.3f}')
        right.plot(row.prediction, row.selected_amplitude_rad, 'rx', ms=8, label=f'Original {row.prediction:.3f}')
        selected = group[group.radar_qualified & group.in_band]
        right.plot(selected.bpm, selected.amplitude_rad, 'k.', ms=4, label='Frozen qualified peaks')
        if row.boundary_hidden_candidate:
            right.plot(row.matched_candidate_bpm, row.matched_candidate_amplitude_rad, 'go', fillstyle='none', ms=8,
                       label=f'ECG-near rank {int(row.matched_candidate_crop_rank)}')
        region = (bpm >= 35) & (bpm <= 85)
        right.set(xlim=(35, 85), ylim=(0, amplitude[region].max()*1.1), xlabel='Frequency (bpm)',
                  title=f'{row.winner_kind}; original crop begins45')
        right.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots/'registered_transitions.png', dpi=150)
    plt.close(fig)
    old = {}
    for folder in ['synchronized_real_pilot', 'synchronized_real_pilot_fixed', 'ellipse_applicability',
                   'ellipse_phase_controls', 'ellipse_readout_controls']:
        root = ROOT.parent/folder
        manifest = read(root/'artifact_manifest.json')
        assert all(digest(root/name) == expected for name, expected in manifest.items())
        old[folder] = len(manifest)
    source = ROOT/'source'
    source.mkdir(exist_ok=True)
    for name in ['audit_real_band_truncation.py', 'score_real_band_truncation.py',
                 'verify_real_band_truncation.py', 'run_ellipse_readout_controls.py', 'run_synchronized_real_pilot.py']:
        shutil.copyfile(name, source/name)
    result = dict(status='passed', elapsed_seconds=time.perf_counter()-began, windows=156,
                  all_frozen_peaks=13603, focus_windows=4, original_predictions_and_references_exact=True,
                  original_mae_reproduced=True, radar_catalogue_unchanged_after_annotation=True,
                  periodogram_max_relative_power_difference=max_power_relative_error,
                  max_absolute_metric_difference=max(errors), original_artifacts_unchanged=old)
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
