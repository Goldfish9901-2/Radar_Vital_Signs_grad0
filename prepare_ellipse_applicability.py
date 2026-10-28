"""Freeze diagnostic timing, outputs and exits before the I/Q applicability audit."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT=Path('experiment_runs/ellipse_applicability'); ROOT.mkdir(exist_ok=True)
previous=json.loads(Path('experiment_runs/synchronized_real_pilot/protocol.json').read_text())
timing=json.loads(Path('experiment_runs/synchronized_real_pilot/records.json').read_text())
accepted=json.loads(Path('experiment_runs/synchronized_real_pilot_fixed/records.json').read_text())
records=[]
for member in previous['members']:
    stem=Path(member['name']).stem; pid=stem.split('_')[0]; scenario='Resting' if 'Resting' in stem else 'Valsalva'
    meta=next(r for r in timing if r['subject']==pid and r['scenario']==scenario)
    records.append({'record':stem,'score_start_s':meta['score_start_s'],'score_end_s':meta['score_end_s'],
                    'intervention_pulses_s':meta['intervention_pulses_s'],
                    'previous_primary_valid':any(r['subject']==pid and r['scenario']==scenario for r in accepted)})
sources=['audit_ellipse_applicability.py','run_ellipse_applicability.py','synchronized_ellipse_fixed.py','run_synchronized_real_pilot.py']
protocol={'date':'2026-10-28','status':'frozen before new applicability audit outputs','parent_commit':'938723a',
    'records':records,'members':previous['members'],'other27_subjects':'No signal or outcome access',
    'fit_intervals':[['cal_0_30',0.,30.],['front_0_15',0.,15.],['back_15_30',15.,30.],['post_30_60',30.,60.]],
    'evaluation_intervals':[[f'cal_block{k+1}',5.*k,5.*(k+1)] for k in range(6)]+[['front',0.,15.],['back',15.,30.],['cal_whole',0.,30.],['post',30.,60.],['later',60.,90.]],
    'additional_evaluation':'Six fixed30s blocks and full180s in the already frozen scoring interval; radar-only metrics, no HR scoring',
    'transfer_pairs':[['front_0_15','back_15_30'],['cal_0_30','post_30_60']],
    'sampling':'100Hz index subsample identical to original calibration (no new filtering); save points for0..max(90,score_end). Not a new antialiased signal representation.',
    'candidate_policy':'Save all6 homogeneous eigenpairs per fit, including all finite rejected candidates. Preserve undefined ellipse diagnostics as null+reason for hyperbolae/imaginary conics, never zero error.',
    'numeric_outputs':['homogeneous alpha/beta','finite eigenvalue/vector','coefficients','center normalized/native mV','ellipse semi-axes normalized/native mV and orientation',
                       'design/scatter/quadratic/shape condition numbers','radial residual','mapped centered singular values','circular variance',
                       '95percent circular angular support arc','chronological unwrapped span','unchanged guard reasons and acceptance'],
    'production_policy':'Import immutable fixed solver only for acceptance checks. Diagnostic partial fits and rejected ellipses never replace production calibration. No HR output, band/peak/window change or guard relaxation.',
    'time_exit':{'support':'Unique proper finite ellipse at both intervals, at least6 common10degree angle bins (>=10points each), and each local95percent arc>=180degrees',
        'time_dependent_mismatch':'Both local fits pass unchanged guards, matched-angle median absolute radius change>=0.10, max cross-fit RMSE penalty>=0.10, and center shift>=0.25 early-radius units or log-shape change>=0.25',
        'stable_geometry':'On supported arcs, matched-angle change<=0.05, transfer penalty<=0.05, center shift<=0.10 and log-shape change<=0.10',
        'otherwise':'Inconclusive/mixed; limited arc and changing angular sampling can make parameter changes non-identifiable',
        'interpretation':'Descriptive diagnostic conventions, not statistical significance, physiological thresholds or new acceptance gates',
        'next':'Time-dependent evidence => study fixed-model applicability. Stable geometry but high mismatch => investigate model/solver/guard rationale, validate any proposed modification on known phase before real rerun. Mixed => retain uncertainty. Do not change HR rules or visit untouched subjects in this audit.'},
    'source_hashes':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in sources},
    'reference_inputs':'No ECG/HR decoded. Existing record timing and intervention markers only; known jump catalogue is separate and never a calibration input.'}
(ROOT/'audit_protocol.json').write_text(json.dumps(protocol,indent=2))
# Register the previously observed lower-band entry/exit only; do not read HR columns.
pred=pd.read_csv('experiment_runs/synchronized_real_pilot_fixed/window_predictions.csv',
                 usecols=['subject','scenario','start_s','end_s','center_s','prediction','reference_offset_s'])
pred=pred[(pred.subject=='GDN0002')&(pred.scenario=='Valsalva')&(pred.reference_offset_s==0)].sort_values('start_s')
near=pred.prediction.to_numpy()<=46
catalogue=[]
for k in range(1,len(pred)):
    if near[k]!=near[k-1]:
        catalogue.append({'from_window':pred.iloc[k-1][['start_s','end_s','center_s','prediction']].to_dict(),
                          'to_window':pred.iloc[k][['start_s','end_s','center_s','prediction']].to_dict(),
                          'event':'enter lower-band bins' if near[k] else 'exit lower-band bins'})
(ROOT/'future_peak_readout_objects.json').write_text(json.dumps({'record':'GDN0002_2_Valsalva','source_commit':'938723a',
    'events':catalogue,'diagnosis_this_round':False,'do_not_change':['calibration','frequency band','peak rule'],
    'boundary_flag_le46bpm_is_catalogue_only':True},indent=2))
print(ROOT,'registered boundary events:',len(catalogue))
