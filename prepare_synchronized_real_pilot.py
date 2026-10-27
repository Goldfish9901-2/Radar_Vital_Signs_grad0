"""Freeze the one-shot pilot before any new participant signal arrays are read."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path('experiment_runs/synchronized_real_pilot')
ROOT.mkdir(parents=True, exist_ok=True)
evidence = ROOT/'provenance'
evidence.mkdir(exist_ok=True)
source = Path('tmp/sync_readiness_kaggle_v2/sync_readiness')
for p in source.iterdir():
    if p.is_file(): shutil.copyfile(p, evidence/p.name)
members = json.loads((source/'archive_members.json').read_text())
subjects = sorted(range(1, 11), key=lambda i: hashlib.sha256(f'sync-real-v1/GDN{i:04d}'.encode()).hexdigest())[:3]
chosen = [m for m in members if any(m['name'].startswith(f'GDN{i:04d}/') for i in subjects)
          and (m['name'].endswith('_Resting.mat') or m['name'].endswith('_Valsalva.mat'))]
assert len(chosen) == 6
protocol = {
    'version': 1, 'date': '2026-10-27', 'status': 'frozen before reading signal arrays or validation scores',
    'question': 'Does one fixed phase/FFT readout add information beyond individual constants and local time trends under published synchronization?',
    'dataset': 'GUARDIAN clinical, DOI 10.6084/m9.figshare.12186516.v2',
    'paper': 'https://www.nature.com/articles/s41597-020-00629-5',
    'archive': {'url': 'https://ndownloader.figshare.com/files/22515785', 'size': 2090795424,
                'official_full_archive_md5': '1376f77a618611b32eea8e07b4382a55'},
    'selection': 'First three SHA256(sync-real-v1/GDNxxxx) IDs within the first official archive; filenames and conditions only; no label or signal inspection',
    'subjects': [f'GDN{i:04d}' for i in subjects], 'members': chosen,
    'other_subjects': 'Not accessed; no replacement for unavailable or failed records',
    'prior_exposure': 'No matching records found in repository audit; prospective operationally untouched evidence, not a registry-certified provenance claim',
    'scoring_intervals': {'Resting': '[30,210) seconds after released record origin',
                          'Valsalva': 'First maneuver release (second of exactly six button-pulse midpoints) + [30,210) seconds; require end before next maneuver'},
    'selection_does_not_guarantee': 'Unexpected body movement or measurable HR residual variation; neither is assumed from scenario name',
    'frontend': {'instrument_adapter': 'CW I/Q; no range or beam dimension; direct algebraic ellipse fit to first 30s at 100Hz calibration subsampling only; no fallback',
                 'reliability': 'Positive-definite ellipse, radial RMSE <=0.2, axis ratio <=100; amplitude radius <0.1 marked low; >1% low per window invalid',
                 'phase': 'P2 principal conjugate increments at native fs followed by cumulative sum; antialiased resample_poly to 20Hz',
                 'scope': 'New frozen CW adapter. This does not test FTU ADC-to-RDA conversion and is not the encrypted official MATLAB elreko implementation'},
    'readout': '512 frames/25.6s; hop 2s; linear detrend, periodic Hann, 4096-point FFT, global maximum in45-150 bpm; no interpolation, HR-guided target, peak or lag selection',
    'reference': {'primary': 'Raw synchronized ECG; pinned NeuroKit2 0.2.12 ecg_clean/ecg_peaks method neurokit, correct_artifacts=False; 250Hz antialias resample',
                  'ecg_lead': 'Use paper code ECG_CHANNEL mapping, fixed before outcomes; second lead provides agreement diagnostics',
                  'anchor': 'Published corrected sample origin, each signal own saved sample rate; window [start,end), reported at center',
                  'target': 'Time integral of 60/RR, piecewise constant on each inter-R-peak interval, divided by full window duration; no extrapolation',
                  'abnormal': 'No beat editing or zero filling. Split finite ECG runs >=10s, exclude 2s run edges. Any overlapping RR outside0.3-2s or incomplete support invalidates window; save beats and failures',
                  'limitation': 'Automatic R peaks not adjudicated; agreement of two leads is a diagnostic, not a clinical reference accuracy guarantee'},
    'ecg_primary_lead': {'GDN0002': 2, 'GDN0007': 2, 'GDN0006': 1},
    'synchronization': {'published_method': 'Dedicated Gold codes recorded on both systems; sampling-rate ratio and lag estimated from codes, not HR',
                        'clock_error_bound_s': None, 'limitation': 'Released sync sequences removed; quantitative per-record residual error not yet established',
                        'primary_offset_s': 0, 'fixed_sensitivity_offsets_s': [-0.5, 0, 0.5],
                        'sensitivity_is_not': 'A measured clock bound or an opportunity to select best lag',
                        'support': 'Intersection of reference-valid windows at all three offsets; identical radar and window selection'},
    'baselines': 'Same-scoring-interval mean, median and OLS time trend use labels offline as oracle diagnostics; not deployable estimates. No concatenation of independent records.',
    'metrics': ['window MAE', 'adjacent-window delta HR RMSE with gaps excluded', 'same-scoring-interval linearly detrended residual RMSE/R2/r',
                'reference/prediction std', 'two-lead window reference disagreement', 'coverage and signal reliability'],
    'statistics': 'Per record, then equal subject weight; three independent subjects only. No window/seed pseudo-replication or significance claim.',
    'interpretation': 'Report raw effects without pass quota. Reference residual variation must be interpreted against detector disagreement and unknown clock error; insufficiently resolved dynamics get mean-HR-only scope. No claim of fine tracking without reference QA and a justified clock-error budget.',
    'execution': 'Kaggle CPU because raw data absent locally; fixed one run, no NN, no further FTU synthesis, no hyperparameter sweep',
    'dependencies': {'neurokit2': '0.2.12'},
    'runner_sha256': hashlib.sha256(Path('run_synchronized_real_pilot.py').read_bytes()).hexdigest(),
    'metadata_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence.iterdir() if p.is_file()},
}
(ROOT/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
print('Selected subjects:', protocol['subjects'], 'bytes:', sum(x['size'] for x in chosen))
