"""Metadata-only protocol creation; no participant arrays are opened."""
from pathlib import Path
import hashlib
import json
import random

root = Path('experiment_runs/peak_subject_validation')
root.mkdir(exist_ok=False)
pool = [f'GDN{i:04}' for i in range(1, 31) if i not in [2, 6, 7]]
subjects = sorted(random.Random(20261107).sample(pool, 6))
article = json.loads(Path('tmp/sync_readiness_kaggle_v2/sync_readiness/clinical_article.json').read_text())
archives = []
for f, first in zip([f for f in article['files'] if f['name'].endswith('.zip')], [1, 11, 21]):
    archives.append(dict(url=f['download_url'], size=f['size'], first=first, last=first+9,
                         official_archive_md5=f['computed_md5']))
channels = [2,2,2,2,2,1,2,2,2,2,2,2,2,2,1,2,2,2,2,2,1,1,2,2,2,2,2,2,2,2]
protocol = dict(date='2026-11-07', status='frozen before selected signal access',
    question='Does removal of crop-created endpoint candidates generalize across untouched subjects without systematic dynamic-error degradation?',
    parent_readout_commit='7fd0bd70d03dbdc5697bb34ff9dd6f3bbef5d3ff',
    seed=20261107, selection='Python random.Random(seed).sample(sorted27IDpool,6), sorted output. ID pool1..30 excluding2,6,7. No signal/label filtering or replacement.',
    pool=pool, subjects=subjects, reserved21=[s for s in pool if s not in subjects], archives=archives,
    records='Resting and first recovery from Valsalva for every selected subject,12intended records,78windows each; retain acquisition/interval/calibration/reference/output failures with936nominal-window denominator.',
    scoring_intervals='Resting[30,210)s; Valsalva second of exactly6intervention pulse midpoints+[30,210)s, must end before third pulse. No alternative recovery or trimming.',
    frontend='Immutable synchronized_ellipse_fixed.py: first30s calibration at100Hz, finite nondegenerate ellipse,radialRMSE<=0.2,axisratio<=100 and mapped-support guards; native P2,cumulative sum,resample_poly20Hz; low amplitude radius<0.1,>1%low per window invalid. No fallback calibration.',
    readout='Same archived original512sample/25.6s,2s hop,linear detrend,periodicHann4096rFFT,45-150bpm. Original=cropped argmax. Candidate=full-spectrum find_peaks then inband largest power; default plateau midpoint and lower-bin ties. No prominence/amplitude filtering, temporal tracking or ECG selection. Empty candidate=NaN.',
    ecg_primary_lead={s: channels[int(s[3:])-1] for s in subjects},
    ecg_primary_source='Official plotData.m ECG_CHANNEL vector frozen from metadata; source hash archived.',
    reference='Pinned neurokit2==0.2.12 original ecg_beats/window_hr,250Hz,finite runs>=10s,2s run-edge exclusion,no artifact correction,no beat edits; RR0.3-2s,complete support,no extrapolation; integrate60/RR over exact window.',
    backup_switch='Before radar processing, per-record lead support=fraction78windows with finite reference atALL3offsets. Keep official primary unless its support<0.80 AND other lead support>=0.80, then switch entire record to other lead. Never windowwise switching or radar-error consultation.',
    reference_validity='Selected lead must have finite reference atall3offsets. At each offset where both leads finite, require absolute HR disagreement<=5bpm; any violation invalidates that window forALLoffsets. Missing secondary does not invalidate an otherwise valid selected lead; count such uncorroborated windows separately. These are operational QA conditions, not clinical adjudication.',
    offsets_s=[0., -.5, .5],
    synchronization='Same released sample origins and ownfs,publishedGold-code correction,unknown residual clock-error bound. Main0s; reportALLfixed sensitivities on identical common support; never select lag.',
    coverage='All12records in denominators: calibration pass; reference anyvalid and>=80%validrecord rates; nominal78window reference support,original radar-valid outputs,candidate outputs and final common paired support; acquisition/interval errors remain in denominators. Reference processed even on calibration failures.',
    metrics='On identical common valid windows forboth readers, perrecord MAE,adjacent2s-pair deltaHRRMSE,|error|>5fraction and improved/worsened/equal counts(absdelta<=1e-10equal),missing outputs and changed endpoint counts. Never bridge gaps. Preserve each offset and all raw predictions/references/validity masks.',
    aggregation='Perrecord effects,then unweighted mean ofavailable2records with count perperson,then equal-person mean. Missing subjects/records haveNaNnot0; report counts and subject direction,never combine936overlaps as independent. Main effect=new-original; show records with both scenarios and partial-coverage separately.',
    interpretation='No significance testing or result-dependent thresholds. Adopt only if cross-person benefits and absence ofsystematic dynamic degradation can be supported by adequate common coverage; if failures block that assessment,markinconclusive and identify calibration/reference as bottleneck. Wrong genuine-local peaks quantified on common support. No tuning/replacement/retraining/further bins after results.',
    environment='KaggleCPU,rawdataabsentlocal. Local downloaded-output verification/metrics<5min withuv-managedenvironment,recordtimings. No new synthetic work or modeltraining.')
(root/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
files = ['run_peak_subject_validation.py', 'run_synchronized_real_pilot.py', 'synchronized_ellipse_fixed.py',
         'run_full_local_peak_readout.py', 'kaggle/prepare_peak_subject_validation.py', 'prepare_peak_validation_protocol.py',
         'experiment_runs/synchronized_real_pilot/protocol.json',
         'tmp/sync_readiness_kaggle_v2/sync_readiness/plotData.m']
# Packaging only Python modules; protocol/metadata keep their full source identities.
hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}
(root/'source_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
(root/'official_plotData.m.txt').write_bytes(Path(files[-1]).read_bytes())
print(json.dumps(dict(subjects=subjects, reserved21=protocol['reserved21']), indent=2))
