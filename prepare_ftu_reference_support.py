"""Freeze reference-only support audit; no radar arrays or scores are read."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT=Path('experiment_runs/ftu_reference_support_v2')
ROOT.mkdir(parents=True,exist_ok=True)
shutil.copyfile('experiment_runs/ftu_frequency_structure/selected_records.csv',ROOT/'selected_records.csv')
protocol=dict(
    base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    purpose='Reference-only feasibility gate before any extended signal reconstruction',
    source='experiment_runs/ftu_adc_controlled/signals',
    permitted_npz_keys=['reference_time','reference_hr'],
    records=6,fs_hz=20,frames=1200,calibration_end_frame=256,
    lengths_frames=[512,768],hop_frames=20,offsets_seconds=[0,1,2],
    candidate_shifts_seconds=[-20,-15,-10,-5,5,10,15,20],
    number_controls=4,minimum_common_windows=5,
    minimum_windows_note='Fixed feasibility floor only; overlapping windows are not independent replicates',
    interpolation='piecewise linear only within actual finite strictly increasing reference support; no circular wrap or extrapolation',
    gaps='query must be bracketed by observations separated by <=2 times median positive reference interval, unless exactly on an observation',
    window_label='mean interpolated HR at every frame query in the unchanged window',
    support='support selection uses timestamps only; each offset has pairwise true/control and selected-all-control windows; final scoring intersects all3offsets',
    control_selection='For each length choose ONE global4shift combination shared by six records. Enumerate all combinations. Rank by number of records with >=5 final common windows, minimum common count across records, total common count, minimum abs shift, sum abs shifts; break ties lexicographically ascending signed tuple. No HR values or radar scores enter selection. If <5 records have support, do not substitute fewer controls or shorten windows.',
    distinguishability='After timestamp-only selection is saved: for EACH selected control at EACH offset, >=60% of final common windows must have abs(mean HR_true-mean HR_control)>1200/length bpm. No reselection based on HR. No p-values.',
    record_evaluable='>=5 final common windows AND all4controls distinguishable at all3offsets',
    stage2_gate='At least5/6 records evaluable at PRIMARY512frames. Secondary768 cannot replace primary. Otherwise no expanded RDA reconstruction/scoring; seek fuller references/synchronization evidence.',
    downstream_gate='unchanged: same candidate >=5/6 evaluable records with coverage>=.9, peak hit>=.6, near-power advantage>=.01 over each control and MAE below each control at all3offsets; only then preregister54records',
    training=False,radar_scores=False,full_rda_reconstruction=False,
    limitation='A failure establishes insufficiency of this frozen finite candidate/control design, not mathematical impossibility of all controls or absence of radar HR information. 20Hz and offsets remain assumptions, not measured sync.')
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
source_hashes=json.loads(Path('experiment_runs/ftu_frequency_structure/input_hashes.json').read_text())
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(protocol['source']).glob('*.npz'))}
assert len(hashes)==6
for n,h in hashes.items():assert h==source_hashes['signals/'+n]
(ROOT/'source_archive_hashes.json').write_text(json.dumps(hashes,indent=2))
print('Reference-only protocol prepared; source hashes verified; no new HR or radar scores computed.')
