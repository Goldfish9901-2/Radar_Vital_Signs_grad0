"""Freeze synthetic evaluation calibration; existing radar conclusions unchanged."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path('experiment_runs/ftu_evaluation_controls');ROOT.mkdir(exist_ok=True,parents=True)
SRC=Path('experiment_runs/ftu_reference_support_v2')
files=['polar_reference_values.csv','selected_support.csv','timestamp_only_selection.json','record_feasibility.csv','decision.json']
hashes={str(SRC/n).replace('\\','/'):hashlib.sha256((SRC/n).read_bytes()).hexdigest() for n in files}
hashes['audit_ftu_structure.py']=hashlib.sha256(Path('audit_ftu_structure.py').read_bytes()).hexdigest()
protocol=dict(date='2026-10-26',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    purpose='Calibrate frozen evaluation sensitivity using reference-derived synthetic signals, not radar performance',
    old_0_of_6_unchanged=True,training=False,new_radar_data=False,rule_changes=False,
    source_files=hashes,fs_hz=20,frames=1200,window_lengths=[512,768],hop_frames=20,nfft=4096,
    hann='numpy.hanning(length), same as audit_ftu_structure.spectrum',band_bpm=[45,150],
    preprocessing='Representation-level periodic real trace, per-window linear detrend, Hann, exact existing spectrum/scalar_metrics functions; no ADC/RDA or phase-extractor claim',
    reference='interpolate existing finite HR only within observed bounds; no extrapolation, no circular labels; use previous exact common windows',
    generators=dict(positive='frequency(t)=piecewise-linear reference HR(t); phase=initial_phase+2*pi*trapezoidal integral frequency/60; trace=cos(phase)',
        constant='frequency=arithmetic mean of all recorded HR samples; intentionally oracle-centered mean-only comparator, not deployable baseline',
        trend='frequency(t)=OLS linear trend of HR vs reference time; all HR residuals removed; trend comparator is not necessarily physiologically independent of labels'),
    noise_sigma=[0.0,0.25],seeds=[20261026,20261027,20261028],
    randomization='same initial phase and standard-normal noise within each record/seed across generators and matched-offset conditions; unit cosine amplitude, noise SD given above; no tuning',
    generation_modes=dict(primary='generate using radar time t aligned to reference time t, score at all0/1/2s assumptions',
        matched_secondary='positive only: separately generate frequency from HR(t+1) or HR(t+2) and score at that matching offset, to separate known time-origin mismatch from rule failure'),
    scoring='exact common-window labels and shifts -20/-15/-10/-5 from frozen timestamp-only selection; no new shift selection, no radar scores',
    metrics=['MAE','peak within5bpm','near-reference band power +/-one native bin','centered RMSE/r/std ratio','power and MAE differences for EACH shifted label','peak versus mean and Hann-energy-weighted mean HR'],
    native_bin_bpm=dict(length512=2.34375,length768=1.5625),display_grid_bpm=.29296875,
    score_rule='coverage>=.9; peak within5>=.6; near-power advantage>=.01 over each4controls; MAE strictly smaller than each4controls',
    admission='Report score-rule diagnostics separately from existing reference-admission gate. Existing0/6 admission remains, so no formal real-data pass or54record expansion.',
    calibration_acceptance='Primary512 noiseless: positive must pass score rule in>=5/6records for each seed at offset0 and in all-offset primary sensitivity; no constant/trend record may pass in any seed at offset0. Matched positive sensitivity reported separately and cannot rescue primary. Noisysigma.25 descriptive robustness. If positive fails or comparator passes, rules are not calibrated for a dynamic tracking claim.',
    secondary768='preserve0/1 common-window support; no pass decision below5windows; no shorter-window rescue',
    development='All six are previously examined development records. No replacement scoring rule is selected or validated this round; any future rule needs development then preregistered independent-record validation.',
    metadata_questions=['same-acquisition radar/Polar trigger relationship and clock mapping','whether original RR or ECG is recorded/released','HR calculation, window, filtering, smoothing, latency and export path'])
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
assert json.loads((SRC/'decision.json').read_text())['evaluable_records']==0
print('Frozen synthetic calibration protocol; previous0/6 preserved.')
