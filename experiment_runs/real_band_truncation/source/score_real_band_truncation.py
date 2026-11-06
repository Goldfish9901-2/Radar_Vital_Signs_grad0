"""Stage2 ECG annotation after a committed, immutable radar catalogue."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

import numpy as np
import pandas as pd


ROOT = Path('experiment_runs/real_band_truncation')
INPUT = Path('experiment_runs/synchronized_real_pilot_fixed')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.perf_counter()
    protocol = read(ROOT/'protocol.json')
    for name, expected in read(ROOT/'source_hashes.json').items():
        assert digest(Path(name)) == expected, name
    freeze = read(ROOT/'radar_freeze.json')
    committed = subprocess.run(['git', 'show', freeze['commit']+':'+(ROOT/'radar_manifest.json').as_posix()],
                               check=True, capture_output=True).stdout
    assert hashlib.sha256(committed).hexdigest() == digest(ROOT/'radar_manifest.json')
    manifest = read(ROOT/'radar_manifest.json')
    assert all(digest(ROOT/'radar'/name) == expected for name, expected in manifest.items())
    assert not (ROOT/'annotated_windows.csv').exists(), 'Preserve the completed annotation'
    # First ECG value decoding in this two-stage audit occurs below the radar-freeze check.
    reference = pd.read_csv(INPUT/'window_predictions.csv')
    reference = reference[reference.reference_offset_s == 0].copy()
    reference['record'] = reference.subject+'_'+reference.scenario.map({'Resting': '1', 'Valsalva': '2'})+'_'+reference.scenario
    windows = pd.read_csv(ROOT/'radar/windows.csv')
    columns = ['record', 'start_s', 'reference', 'secondary_lead_reference']
    windows = windows.merge(reference[columns], on=['record', 'start_s'], validate='one_to_one')
    windows['absolute_error_bpm'] = abs(windows.prediction-windows.reference)
    tolerance = protocol['match_tolerance_bpm']
    windows['error_primary'] = windows.absolute_error_bpm > tolerance
    windows['error_over5'] = windows.absolute_error_bpm > 5
    peaks = pd.read_csv(ROOT/'radar/peaks.csv')
    peaks = peaks.merge(windows[['record', 'start_s', 'reference', 'prediction']],
                        on=['record', 'start_s'], validate='many_to_one')
    peaks['signed_distance_bpm'] = peaks.bpm-peaks.reference
    peaks['absolute_distance_bpm'] = abs(peaks.signed_distance_bpm)
    peaks['reference_match_primary'] = peaks.absolute_distance_bpm <= tolerance
    peaks['reference_match_secondary5'] = peaks.absolute_distance_bpm <= 5
    peaks['distinct_from_original_winner'] = abs(peaks.bpm-peaks.prediction) > tolerance
    peaks['qualified_distinct_reference_candidate'] = (peaks.radar_qualified & peaks.reference_match_primary &
                                                       peaks.distinct_from_original_winner)
    peak_groups = peaks.groupby(['record', 'start_s'])
    candidate_rows = []
    for row in windows.itertuples(index=False):
        group = peak_groups.get_group((row.record, row.start_s))
        full_band = group[group.full_local & group.in_band]
        qualified = full_band[full_band.radar_qualified]
        main = qualified[qualified.reference_match_primary & qualified.distinct_from_original_winner]
        secondary = qualified[qualified.reference_match_secondary5 & qualified.distinct_from_original_winner]
        nearest = main.sort_values(['absolute_distance_bpm', 'crop_rank']).iloc[0] if len(main) else None
        candidate_rows.append(dict(
            any_full_reference_peak=bool(full_band.reference_match_primary.any()),
            any_qualified_reference_peak=bool(qualified.reference_match_primary.any()),
            distinct_reference_candidate_count=len(main), secondary5_candidate_count=len(secondary),
            qualified_band_peak_count=len(qualified), full_band_peak_count=len(full_band),
            boundary_hidden_candidate=bool(row.error_primary and row.selected_boundary_region and len(main)),
            secondary5_boundary_candidate=bool(row.error_primary and row.selected_boundary_region and len(secondary)),
            matched_candidate_bpm=float(nearest.bpm) if nearest is not None else None,
            matched_candidate_amplitude_rad=float(nearest.amplitude_rad) if nearest is not None else None,
            matched_candidate_relative_amplitude=float(nearest.relative_amplitude) if nearest is not None else None,
            matched_candidate_crop_rank=int(nearest.crop_rank) if nearest is not None else None,
            matched_candidate_prominence_fraction=float(nearest.prominence_fraction) if nearest is not None else None,
            matched_candidate_power_over_median=float(nearest.power_over_band_median) if nearest is not None else None))
    windows = pd.concat([windows.reset_index(drop=True), pd.DataFrame(candidate_rows)], axis=1)
    events = read(Path('experiment_runs/ellipse_applicability/future_peak_readout_objects.json'))
    windows['registered_focus'] = False
    for event in events['events']:
        for side in ['from_window', 'to_window']:
            selection = (windows.record == events['record']) & np.isclose(windows.start_s, event[side]['start_s'], atol=1e-8, rtol=0)
            assert selection.sum() == 1
            assert windows.loc[selection, 'prediction'].iloc[0] == event[side]['prediction']
            windows.loc[selection, 'registered_focus'] = True
    summaries = []
    for record, group in windows.groupby('record', sort=False):
        error = group[group.error_primary]
        boundary_error = error[error.selected_boundary_region]
        valid = group.radar_valid & group.common_support & np.isfinite(group.reference)
        summaries.append(dict(record=record, windows=len(group), original_scored=int(valid.sum()),
            original_mae_bpm=float(group.loc[valid, 'absolute_error_bpm'].mean()),
            error_windows=len(error), over5_windows=int(group.error_over5.sum()),
            boundary_windows=int(group.selected_boundary_region.sum()), boundary_error_windows=len(boundary_error),
            cut_flank_error_windows=int(boundary_error.winner_kind.str.contains('cut_').sum()),
            error_windows_with_full_reference_peak=int(error.any_full_reference_peak.sum()),
            error_windows_with_qualified_reference_peak=int(error.any_qualified_reference_peak.sum()),
            boundary_hidden_candidates=int(group.boundary_hidden_candidate.sum()),
            secondary5_boundary_candidates=int(group.secondary5_boundary_candidate.sum()),
            hidden_cut_flank=int((group.boundary_hidden_candidate & group.winner_kind.str.contains('cut_')).sum())))
    windows.to_csv(ROOT/'annotated_windows.csv', index=False)
    peaks.to_csv(ROOT/'annotated_peaks.csv', index=False)
    pd.DataFrame(summaries).to_csv(ROOT/'record_summary.csv', index=False)
    windows[windows.registered_focus].to_csv(ROOT/'registered_focus.csv', index=False)
    result = dict(elapsed_seconds=time.perf_counter()-started, windows=len(windows), peaks=len(peaks),
                  radar_freeze_commit=freeze['commit'], radar_outputs_unchanged=True,
                  reference_source='Existing primary ECG window labels at0s; no new ECG processing')
    (ROOT/'annotation_execution.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    main()
