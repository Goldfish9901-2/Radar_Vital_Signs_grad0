"""Local fixed paired scoring; independent archived-signal/reference checks."""
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

ROOT = Path('experiment_runs/peak_subject_validation')
INPUT = Path('tmp/peak_subject_validation_results/peak_subject_validation')


def main():
    started = time.perf_counter()
    protocol = json.loads((ROOT/'protocol.json').read_text())
    records = json.loads((INPUT/'records.json').read_text())
    windows = pd.read_csv(INPUT/'windows.csv')
    assert len(records) == 12
    assert sorted({r['subject'] for r in records}) == protocol['subjects']
    # Frozen remote code is compared byte-for-byte against committed inputs.
    for name, expected in json.loads((ROOT/'source_hashes.json').read_text()).items():
        local = Path(name)
        assert hashlib.sha256(local.read_bytes()).hexdigest() == expected, name
        frozen = INPUT.parent/'frozen_validation_code'/local.name
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == expected, frozen
    assert json.loads((INPUT.parent/'frozen_validation_code/validation_protocol.json').read_text()) == protocol
    # Independent reference integral from saved beats; selected lead and validity
    # recomputed without consulting either radar prediction.
    max_ref_diff = 0.
    max_power_diff = 0.
    verified_windows = 0
    windows['original_is_full_local_peak'] = False
    windows['original_is_band_endpoint'] = False
    for record in records:
        name = record['record']
        g = windows[windows.record == name]
        if not len(g):
            assert not record.get('interval_valid', False)
            continue
        assert len(g) == 234
        beats = json.loads((INPUT/f'{name}_ecg_beats.json').read_text())
        starts = sorted(g.start_s.unique())
        refs = np.full((2, 78, 3), np.nan)
        for lead in range(2):
            times = np.array(beats[lead]['beats_s'])
            rr = np.diff(times)
            for i, s in enumerate(starts):
                for k, offset in enumerate(protocol['offsets_s']):
                    a, b = s+offset, s+25.6+offset
                    if not any(first <= a and last >= b for first, last in beats[lead]['valid_runs_s']):
                        continue
                    overlap = np.maximum(0., np.minimum(b, times[1:])-np.maximum(a, times[:-1]))
                    use = overlap > 0
                    if abs(overlap.sum()-25.6) > 1e-6 or np.any((rr[use] < .3) | (rr[use] > 2)):
                        continue
                    refs[lead, i, k] = np.dot(overlap[use], 60/rr[use])/25.6
        for lead in range(2):
            actual = g.pivot(index='start_s', columns='offset_s', values=f'lead{lead+1}_reference')[protocol['offsets_s']].to_numpy()
            assert np.array_equal(np.isfinite(refs[lead]), np.isfinite(actual))
            finite = np.isfinite(actual)
            if finite.any():
                max_ref_diff = max(max_ref_diff, float(np.max(abs(actual[finite]-refs[lead][finite]))))
        support = np.isfinite(refs).all(axis=2)
        primary = protocol['ecg_primary_lead'][record['subject']]-1
        selected = 1-primary if support[primary].mean() < .8 and support[1-primary].mean() >= .8 else primary
        assert record['selected_lead'] == selected+1
        both = np.isfinite(refs).all(axis=0)
        concordant = (~both | (abs(refs[0]-refs[1]) <= 5)).all(axis=1)
        valid = support[selected] & concordant
        actual_valid = g[g.offset_s == 0].sort_values('start_s').reference_valid.to_numpy()
        assert np.array_equal(valid, actual_valid)
        if record['calibration_passed']:
            archive = np.load(INPUT/f'{name}_spectra.npz')
            phase = np.load(INPUT/f'{name}_phase20.npz')
            base = g[g.offset_s == 0].sort_values('start_s')
            for index, row in enumerate(base.itertuples()):
                first = round(row.start_s*20)
                trace = phase['phase'][first:first+512]
                freq, spectrum = signal.periodogram(trace, fs=20, window='hann', nfft=4096,
                    detrend='linear', scaling='spectrum')
                power = spectrum*256**2/2
                power[[0, -1]] *= 2
                original_power = archive['power'][index]
                max_power_diff = max(max_power_diff, float(np.max(abs(power-original_power))/max(np.max(original_power), 1e-30)))
                bpm = freq*60
                band = np.flatnonzero((bpm >= 45) & (bpm <= 150))
                original = bpm[band[np.argmax(power[band])]]
                original_index = int(band[np.argmax(power[band])])
                # Independent strict extrema matches all nonplateau real spectra;
                # explicit plateau fallback preserves the frozen SciPy definition.
                if np.any(power[1:] == power[:-1]):
                    peaks = signal.find_peaks(power)[0]
                else:
                    peaks = np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] > power[2:]))+1
                peaks = peaks[(bpm[peaks] >= 45) & (bpm[peaks] <= 150)]
                mask = (windows.record == name) & (windows.start_s == row.start_s)
                windows.loc[mask, 'original_is_full_local_peak'] = bool(original_index in peaks)
                windows.loc[mask, 'original_is_band_endpoint'] = bool(original_index in [band[0], band[-1]])
                candidate = bpm[peaks[np.argmax(power[peaks])]] if len(peaks) else np.nan
                assert original == row.original
                assert (np.isnan(candidate) and np.isnan(row.candidate)) or candidate == row.candidate
                reliable = np.mean(phase['low_amplitude'][first:first+512]) <= .01
                assert bool(reliable) == row.radar_valid
                verified_windows += 1
        else:
            assert not g.radar_valid.any() and g.original.isna().all() and g.candidate.isna().all()
    assert max_ref_diff < 1e-8 and max_power_diff < 1e-12
    summaries = []
    for record in records:
        for offset in protocol['offsets_s']:
            g = windows[(windows.record == record['record']) & (windows.offset_s == offset)].sort_values('start_s')
            row = dict(subject=record['subject'], scenario=record['scenario'], record=record['record'], offset_s=offset,
                planned_windows=78, actual_windows=len(g), calibration_passed=record.get('calibration_passed', False),
                reference_any_valid=record.get('reference_any_valid', False),
                reference_80pct_valid=record.get('reference_80pct_valid', False),
                selected_lead=record.get('selected_lead'), backup_switched=record.get('backup_switched', False))
            if len(g):
                common = g.reference_valid & g.radar_valid & g.original.notna() & g.candidate.notna()
                assert np.array_equal(common, g.common_valid)
                missing_other = (g.lead1_reference.isna() if record['selected_lead'] == 2 else g.lead2_reference.isna())
                row.update(reference_valid_windows=int(g.reference_valid.sum()),
                    reference_uncorroborated_windows=int((g.reference_valid & missing_other).sum()),
                    original_output_windows=int((g.radar_valid & g.original.notna()).sum()),
                    candidate_output_windows=int((g.radar_valid & g.candidate.notna()).sum()),
                    common_windows=int(common.sum()), candidate_no_peak=int((g.radar_valid & g.candidate.isna()).sum()))
                f = g[common]
                delta = abs(f.candidate-f.reference)-abs(f.original-f.reference)
                row.update(improved=int((delta < -1e-10).sum()), worsened=int((delta > 1e-10).sum()), equal=int((abs(delta) <= 1e-10).sum()),
                    changed=int((f.original != f.candidate).sum()),
                    original_nonlocal_winners=int((~f.original_is_full_local_peak).sum()),
                    unchanged_genuine_over5=int((f.original_is_full_local_peak & (f.original == f.candidate) & (abs(f.original-f.reference) > 5)).sum()))
                for model in ['original', 'candidate']:
                    error = f[model].to_numpy()-f.reference.to_numpy()
                    adjacent = np.isclose(np.diff(f.start_s), 2, atol=1e-8, rtol=0)
                    row[model+'_mae'] = float(np.mean(abs(error))) if len(error) else np.nan
                    row[model+'_over5_fraction'] = float(np.mean(abs(error) > 5)) if len(error) else np.nan
                    row[model+'_change_rmse'] = float(np.sqrt(np.mean(np.diff(error)[adjacent]**2))) if adjacent.any() else np.nan
                    row['adjacent_pairs'] = int(adjacent.sum())
                for metric in ['mae', 'over5_fraction', 'change_rmse']:
                    row['delta_'+metric] = row['candidate_'+metric]-row['original_'+metric]
            else:
                for key in ['reference_valid_windows', 'reference_uncorroborated_windows', 'original_output_windows',
                            'candidate_output_windows', 'common_windows', 'candidate_no_peak', 'improved', 'worsened', 'equal', 'changed']:
                    row[key] = 0
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(ROOT/'record_summary.csv', index=False)
    metrics = ['original_mae', 'candidate_mae', 'delta_mae', 'original_change_rmse', 'candidate_change_rmse',
               'delta_change_rmse', 'original_over5_fraction', 'candidate_over5_fraction', 'delta_over5_fraction']
    subjects = []
    for (subject, offset), g in summary.groupby(['subject', 'offset_s'], sort=False):
        eligible = g[g.common_windows > 0]
        subjects.append(dict(subject=subject, offset_s=offset, scored_records=len(eligible),
            common_windows=int(g.common_windows.sum()), planned_windows=156,
            **{metric: float(eligible[metric].mean()) if len(eligible) else np.nan for metric in metrics}))
    subject_summary = pd.DataFrame(subjects)
    subject_summary.to_csv(ROOT/'subject_summary.csv', index=False)
    aggregates = []
    for offset, group in subject_summary.groupby('offset_s', sort=False):
        for label, selected in [('any_scored', group[group.scored_records > 0]), ('both_records', group[group.scored_records == 2])]:
            aggregates.append(dict(offset_s=offset, scope=label, subjects=len(selected),
                **{metric: float(selected[metric].mean()) if len(selected) else np.nan for metric in metrics}))
    pd.DataFrame(aggregates).to_csv(ROOT/'equal_subject_summary.csv', index=False)
    windows.to_csv(ROOT/'windows.csv', index=False)
    coverage = summary[summary.offset_s == 0]
    result = dict(verification_seconds=time.perf_counter()-started, records_intended=12, windows_intended=936,
        calibration_passed=int(coverage.calibration_passed.sum()), reference_any_valid_records=int(coverage.reference_any_valid.sum()),
        reference80_valid_records=int(coverage.reference_80pct_valid.sum()), reference_windows=int(coverage.reference_valid_windows.sum()),
        original_output_windows=int(coverage.original_output_windows.sum()), candidate_output_windows=int(coverage.candidate_output_windows.sum()),
        common_windows=int(coverage.common_windows.sum()), real_spectra_verified=verified_windows,
        max_reference_difference=max_ref_diff, max_relative_spectrum_difference=max_power_diff,
        backup_switches=int(coverage.backup_switched.sum()), new_subjects=protocol['subjects'], reserved21_unchanged=True)
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (ROOT/'plots').mkdir(exist_ok=True)
    fig, axes = plt.subplots(6, 2, figsize=(12, 15), constrained_layout=True)
    for n, subject in enumerate(protocol['subjects']):
        for j, scenario in enumerate(['Resting', 'Valsalva']):
            g = windows[(windows.subject == subject) & (windows.scenario == scenario) & (windows.offset_s == 0)].sort_values('start_s')
            ax = axes[n, j]
            ax.set_title(subject+' '+scenario+f' (paired {int(g.common_valid.sum())}/78)')
            if len(g):
                x = g.start_s+12.8
                ax.plot(x, g.reference.where(g.reference_valid), color='black', label='ECG valid')
                ax.plot(x, g.original.where(g.radar_valid), color='tab:orange', label='Original', alpha=.8)
                ax.plot(x, g.candidate.where(g.radar_valid), color='tab:blue', label='Full local peak', alpha=.8)
                if not g.radar_valid.any():
                    ax.text(.5, .4, 'No reliable radar output', transform=ax.transAxes, ha='center')
                if not g.reference_valid.any():
                    ax.text(.5, .4, 'No valid ECG window reference', transform=ax.transAxes, ha='center')
            else:
                ax.text(.5, .5, 'Record unavailable', transform=ax.transAxes, ha='center')
            ax.set_xlabel('Window centre (s)')
            ax.set_ylabel('HR (bpm)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='outside upper center', ncol=3)
    fig.savefig(ROOT/'plots/all12records.png', dpi=140)
    plt.close(fig)
    result['verification_and_plots_seconds'] = time.perf_counter()-started
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(coverage.to_string(index=False))
    print(subject_summary[subject_summary.offset_s == 0].to_string(index=False))


if __name__ == '__main__':
    main()
