"""Verify all saved readout windows and summarize paired synthetic controls."""
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

from synchronized_ellipse_fixed import ellipse_phase


ROOT = Path('experiment_runs/ellipse_readout_controls')
PARENT = Path('experiment_runs/ellipse_phase_controls')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.perf_counter()
    protocol = read(ROOT/'protocol.json')
    assert all(digest(Path(name)) == expected for name, expected in read(ROOT/'source_hashes.json').items())
    frame = pd.read_csv(ROOT/'windows.csv')
    peaks = pd.read_csv(ROOT/'peaks.csv')
    calibrations = {row['tag']: row for row in read(ROOT/'calibrations.json')}
    assert len(frame) == 360 and frame['readout_available'].all()
    assert len(calibrations) == 60 and all(row['accepted'] for row in calibrations.values())
    assert not frame.duplicated(['tag', 'source', 'start_s']).any()
    cached = {}
    errors = []
    alternate_power_errors = []
    bpm = np.fft.rfftfreq(4096, .05)*60
    band = (bpm >= 45) & (bpm <= 150)
    band_indices = np.flatnonzero(band)
    hann = .5-.5*np.cos(2*np.pi*np.arange(512)/512)
    target = abs(bpm-72) <= 1.171875
    lower = band & (bpm <= 47.34375)
    upper = band & (bpm >= 147.65625)
    for tag, group in frame.groupby('tag', sort=False):
        data = dict(np.load(ROOT/'signals'/f'{tag}.npz'))
        cached[tag] = data
        row = group.iloc[0]
        parent_tag = f"{row['case']}_seed{row['seed']}_noise{row['noise_std']:g}"
        old = dict(np.load(PARENT/'signals'/f'{parent_tag}.npz'))
        if row['heartbeat'] == 'present':
            assert np.array_equal(data['iq'], old['iq'])
            assert np.array_equal(data['true_phase_rad'], old['true_phase_rad'])
            assert np.array_equal(data['recovered_phase_rad'], old['raw_recovered_phase_rad'])
        else:
            t = data['time_s']
            truth = old['true_phase_rad']-.08*np.sin(2*np.pi*1.2*t)
            assert np.array_equal(truth, data['true_phase_rad'])
            radius = np.ones_like(truth)
            observed = truth.copy()
            if row['case'].startswith('D_'):
                radius += .12*np.cos(4*truth)
            if row['case'] == 'D_warp':
                observed += .6*np.sin(2*truth)
            latent = radius[:, None]*np.column_stack([np.cos(observed), np.sin(observed)])
            parent_protocol = read(PARENT/'protocol.json')
            expected_iq = (latent+old['true_center_shift_circle'])@np.array(parent_protocol['affine'])
            expected_iq += parent_protocol['center_native']
            expected_iq += old['noise_native']
            errors.append(float(np.max(abs(data['iq']-expected_iq))))
        recovered, low, metadata = ellipse_phase(data['iq'][:, 0], data['iq'][:, 1], 100)
        errors.append(float(np.max(abs(recovered-data['recovered_phase_rad']))))
        errors.append(abs(metadata['calibration_radial_rmse']-calibrations[tag]['calibration_radial_rmse']))
        phase20 = np.stack([signal.resample_poly(data['true_phase_rad'], 1, 5), signal.resample_poly(recovered, 1, 5)])
        errors.append(float(np.max(abs(phase20-data['phase20']))))
        low20 = signal.resample_poly(low.astype(float), 1, 5) > .01
        assert np.array_equal(low20, data['low_amplitude20'])
        for source_index, source in enumerate(['truth', 'recovered']):
            for window_index, start in enumerate([30., 32., 34.]):
                record = group[(group.source == source) & (group.start_s == start)].iloc[0]
                trace = phase20[source_index, round(start*20):round(start*20)+512]
                weighted = signal.detrend(trace)*hann
                power = abs(np.fft.rfft(weighted, n=4096))**2
                saved_power = data['power'][source_index, window_index]
                errors.append(float(np.max(abs(power-saved_power))))
                index = int(band_indices[np.argmax(power[band])])
                assert bpm[index] == record.prediction_bpm
                amplitude = 2*np.sqrt(power)/sum(hann)
                exact = 2*abs(sum(weighted*np.exp(-2j*np.pi*1.2*np.arange(512)/20)))/sum(hann)
                errors.extend([abs(exact-record.target_exact_amplitude_rad),
                               abs(amplitude[index]-record.selected_amplitude_rad),
                               abs(power[target].sum()-record.target_region_power),
                               abs(power[target].sum()/power[band].sum()-record.target_region_power_fraction),
                               abs(float(low20[round(start*20):round(start*20)+512].mean())-record.low_amplitude_fraction)])
                nearest = np.argmin(abs(bpm-72))
                errors.append(abs(amplitude[nearest]-record.target_nearest_bin_amplitude_rad))
                errors.append(abs(amplitude[target].max()-record.target_region_max_amplitude_rad))
                assert record.selected_band_endpoint == (index in [band_indices[0], band_indices[-1]])
                assert record.selected_boundary_region == bool(lower[index] or upper[index])
                assert record.within_72_reference == (abs(bpm[index]-72) <= 1.171875)
                for region, label in [(lower, 'lower'), (upper, 'upper')]:
                    local_index = int(np.flatnonzero(region)[np.argmax(power[region])])
                    assert bpm[local_index] == record[f'{label}_region_peak_bpm']
                    errors.append(abs(amplitude[local_index]-record[f'{label}_region_peak_amplitude_rad']))
                errors.append(abs(amplitude[band_indices[0]]-record.lower_endpoint_amplitude_rad))
                errors.append(abs(amplitude[band_indices[-1]]-record.upper_endpoint_amplitude_rad))
                # Enumerate maxima directly rather than using scipy.find_peaks.
                spectral_band = power[band]
                maxima = []
                for k, global_index in enumerate(band_indices):
                    left = k == 0 or spectral_band[k] >= spectral_band[k-1]
                    right = k == len(band_indices)-1 or spectral_band[k] >= spectral_band[k+1]
                    if left and right:
                        maxima.append(int(global_index))
                maxima.sort(key=lambda j: (-power[j], j))
                saved_peaks = peaks[(peaks.tag == tag) & (peaks.source == source) & (peaks.start_s == start)].sort_values('rank')
                assert saved_peaks.bpm.tolist() == [float(bpm[j]) for j in maxima]
                errors.append(float(np.max(abs(saved_peaks.amplitude_rad.to_numpy()-amplitude[maxima]))))
                competitors = [j for j in maxima if abs(bpm[j]-72) > 2.34375]
                if competitors:
                    assert record.competitor_bpm == bpm[competitors[0]]
                    errors.append(abs(record.competitor_amplitude_rad-amplitude[competitors[0]]))
                else:
                    assert pd.isna(record.competitor_bpm)
                # Independent least-squares linear detrending, with roundoff-tolerant power comparison.
                design = np.c_[np.ones(512), np.arange(512)/512]
                independent = trace-design@np.linalg.lstsq(design, trace, rcond=None)[0]
                alternate_power = abs(np.fft.rfft(independent*hann, n=4096))**2
                alternate_power_errors.append(float(np.max(abs(alternate_power-saved_power))))
    assert max(errors) < 1e-8, max(errors)
    assert max(alternate_power_errors) < 1e-8, max(alternate_power_errors)
    keys = ['case', 'seed', 'noise_std', 'heartbeat', 'source']
    runs = frame.groupby(keys, sort=False).agg(
        windows=('start_s', 'size'), available=('readout_available', 'all'),
        calibration_accepted=('calibration_accepted', 'all'), all_within72=('within_72_reference', 'all'),
        proximity72_fraction=('within_72_reference', 'mean'), prediction_min_bpm=('prediction_bpm', 'min'),
        prediction_max_bpm=('prediction_bpm', 'max'), mean_abs_distance_to72=('prediction_bpm', lambda x: abs(x-72).mean()),
        target_amplitude_rad=('target_exact_amplitude_rad', 'mean'),
        selected_amplitude_rad=('selected_amplitude_rad', 'mean'),
        competitor_amplitude_rad=('competitor_amplitude_rad', 'mean'),
        target_power_fraction=('target_region_power_fraction', 'mean'),
        boundary_window_fraction=('selected_boundary_region', 'mean'),
        endpoint_window_fraction=('selected_band_endpoint', 'mean')).reset_index()
    assert len(runs) == 120 and (runs.windows == 3).all()
    # Distance to72 is a descriptive frequency distance for absent conditions, not MAE.
    runs['positive_readout_normal'] = pd.array([bool(row.calibration_accepted and row.all_within72)
                                               if row.heartbeat == 'present' else pd.NA
                                               for row in runs.itertuples()], dtype='boolean')
    runs.to_csv(ROOT/'run_summaries.csv', index=False)
    summary = runs.groupby(['case', 'noise_std', 'heartbeat', 'source']).agg(
        runs=('seed', 'size'), frequency_min_bpm=('prediction_min_bpm', 'min'),
        frequency_max_bpm=('prediction_max_bpm', 'max'),
        mean_distance_to72_bpm=('mean_abs_distance_to72', 'mean'),
        mean_target_amplitude_rad=('target_amplitude_rad', 'mean'),
        mean_selected_amplitude_rad=('selected_amplitude_rad', 'mean'),
        mean_competitor_amplitude_rad=('competitor_amplitude_rad', 'mean'),
        mean_boundary_fraction=('boundary_window_fraction', 'mean'))
    summary.to_csv(ROOT/'condition_summary.csv')
    present = runs[runs.heartbeat == 'present']
    join_keys = ['case', 'seed', 'noise_std']
    paired = present[present.source == 'truth'].merge(present[present.source == 'recovered'], on=join_keys, suffixes=('_truth', '_recovered'))
    old_metrics = pd.read_csv(PARENT/'metrics.csv')
    paired = paired.merge(old_metrics[join_keys+['test_phase_rmse_rad', 'meets_phase_reference']], on=join_keys)
    def decision(row):
        if row.positive_readout_normal_truth and row.positive_readout_normal_recovered:
            return 'both_normal'
        if row.positive_readout_normal_truth:
            return 'truth_normal_recovery_abnormal'
        if row.positive_readout_normal_recovered:
            return 'truth_abnormal_recovery_normal'
        return 'both_abnormal'
    paired['exit_category'] = paired.apply(decision, axis=1)
    paired.to_csv(ROOT/'paired_readout.csv', index=False)
    pair_keys = ['case', 'seed', 'noise_std', 'source']
    heartbeat_pairs = runs[runs.heartbeat == 'present'].merge(runs[runs.heartbeat == 'absent'], on=pair_keys, suffixes=('_present', '_absent'))
    heartbeat_pairs['target_amplitude_difference_rad'] = heartbeat_pairs.target_amplitude_rad_present-heartbeat_pairs.target_amplitude_rad_absent
    heartbeat_pairs.to_csv(ROOT/'heartbeat_pairs.csv', index=False)
    plots = ROOT/'plots'
    plots.mkdir(exist_ok=True)
    for case in protocol['cases']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, heartbeat in zip(axes, ['present', 'absent']):
            # First declared seed, higher declared noise, all windows, selected without scores.
            tag = f'{case}_seed20261106_noise0.005_{heartbeat}'
            for source_index, source in enumerate(['truth', 'recovered']):
                amplitudes = 2*np.sqrt(cached[tag]['power'][source_index])/256
                for k, start in enumerate([30, 32, 34]):
                    ax.plot(bpm[band], np.maximum(amplitudes[k, band], 1e-12),
                            color=['tab:blue', 'tab:orange'][source_index], alpha=.7,
                            ls=['-', '--', ':'][k], label=f'{source}, {start}s')
            ax.axvline(72, color='black', lw=.8)
            ax.set(xlabel='Frequency (bpm)', ylabel='Coherent amplitude (rad)',
                   title=f'{case}, heartbeat {heartbeat}', yscale='log', ylim=(1e-12, .2))
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plots/f'{case}_spectra.png', dpi=150)
        plt.close(fig)
    old_artifacts = {}
    for folder in ['synchronized_real_pilot', 'synchronized_real_pilot_fixed', 'ellipse_applicability', 'ellipse_phase_controls']:
        root = ROOT.parent/folder
        manifest = read(root/'artifact_manifest.json')
        assert all(digest(root/name) == expected for name, expected in manifest.items())
        old_artifacts[folder] = len(manifest)
    source = ROOT/'source'
    source.mkdir(exist_ok=True)
    for name in ['run_ellipse_readout_controls.py', 'verify_ellipse_readout_controls.py',
                 'synchronized_ellipse_fixed.py', 'run_synchronized_real_pilot.py']:
        shutil.copyfile(name, source/name)
    result = dict(status='passed', elapsed_seconds=time.perf_counter()-started,
                  windows=360, all_window_peaks_checked=True, conditions=60, source_runs=120,
                  paired_positive_runs=len(paired), exit_counts=paired.exit_category.value_counts().to_dict(),
                  all_present_inputs_exactly_reused=True, absent_noise_and_observation_pairing_verified=True,
                  max_absolute_numeric_difference=max(errors),
                  independent_detrend_power_max_difference=max(alternate_power_errors),
                  original_artifacts_unchanged=old_artifacts)
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
