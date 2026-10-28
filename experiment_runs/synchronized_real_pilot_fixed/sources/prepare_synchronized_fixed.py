"""Freeze the solver repair and blinded ECG review separately from outcomes."""
from pathlib import Path
import hashlib
import json

root = Path('experiment_runs/synchronized_real_pilot_fixed')
root.mkdir(parents=True, exist_ok=True)
old_path = Path('experiment_runs/synchronized_real_pilot/protocol.json')
old = json.loads(old_path.read_text())
files = ['synchronized_ellipse_fixed.py', 'validate_synchronized_ellipse_fixed.py',
         'run_synchronized_real_pilot_fixed.py', 'run_synchronized_real_pilot.py']
protocol = {'version': 2, 'parent_protocol_commit': '46fc91b', 'parent_result_commit': '7e357da',
            'date': '2026-10-27', 'status': 'frozen before corrected radar outcomes',
            'defect': 'Original singular generalized eigensystem discarded eigenvalues and admitted infinite-eigenvalue degenerate conics; radial residual alone cannot detect unit-circle point collapse',
            'changed': 'Only ellipse candidate validation and support checks; no new fitting objective, calibration interval or selection by HR',
            'unchanged_protocol_sha256': hashlib.sha256(old_path.read_bytes()).hexdigest(),
            'members': old['members'], 'subjects': old['subjects'], 'ecg_primary_lead': old['ecg_primary_lead'],
            'unchanged': ['original record bytes', 'first30s calibration', 'score intervals', '512-frame windows and2s hops',
                          '45-150bpm band', 'FFT readout', 'main ECG leads and frozen automatic peaks', 'fixed reference offsets'],
            'failure_policy': 'No valid calibration => record failure with no HR output and no replacement; preserve all failures',
            'guards': {'finite_generalized_eigenvalue': 'homogeneous beta >64epsilon*(abs(alpha)+abs(beta))',
                       'eigenpair_backward_error_max': 1e-7, 'quadratic_norm_fraction_min': 1e-10,
                       'mapped_largest_centered_std_min': .001, 'mapped_smallest_centered_std_min': 1e-6,
                       'mapped_circular_variance_min': 1e-6, 'mapped_unwrapped_q99_q01_span_min_rad': .01,
                       'original_radial_rmse_max': .2, 'original_axis_ratio_max': 100},
            'guard_interpretation': 'Numerical degeneracy guards only, not physiological correctness or full ellipse identifiability guarantees',
            'source_hashes': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files},
            'no_other_subject_access': True, 'training': False,
            'reference_review': 'ECG-only materials; no corrected radar outputs opened before signed review decisions; reviewer had seen previous defective-run predictions, so not retrospectively blinded to old study',
            'clock_error_bound_s': None}
(root/'repair_protocol.json').write_text(json.dumps(protocol, indent=2))
review = {'date': '2026-10-27', 'scope': 'All184s saved raw primary-lead ECG per six records, including scoring boundaries',
          'allowed_inputs': ['raw primary ECG', 'native sample rate/time origin', 'original automatic R peaks', 'fixed record and window definitions'],
          'forbidden_inputs': ['corrected radar predictions', 'radar errors', 'radar spectra', 'secondary lead selection to improve outcomes'],
          'review': 'Visual technical adjudication of all10s strips, including start/end support; optional filtered ECG trace only for morphology visibility; preserve original signal',
          'decisions': 'Accept clear QRS detections; add/delete/refine only with explicit ECG morphology evidence and recorded times/reasons. Mark uncertain spans unscorable; never edit for HR smoothness or radar agreement.',
          'abnormal_beats': 'Do not remove plausible ectopic morphology solely for RR irregularity; retain existing0.3-2s scoring QC and flag ambiguous intervals',
          'comparison': 'Keep original automatic peaks and score both references on common support after review freeze',
          'limitation': 'Technical research waveform review by the assistant, not independent expert clinical annotation; previous defective-run predictions were already visible',
          'other_subjects': '27 remain untouched'}
(root/'ecg_review_protocol.json').write_text(json.dumps(review, indent=2))
print(root)
