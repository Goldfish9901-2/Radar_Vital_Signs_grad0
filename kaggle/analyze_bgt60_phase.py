import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import argparse
import gzip

parser = argparse.ArgumentParser(description='Analyze direct-phase windows without model training')
parser.add_argument('--input-dir', type=Path, default=Path('experiment_runs/bgt60_direct_phase_2026-10-10'))
args = parser.parse_args()
p = args.input_dir
source = p/'windows.jsonl'
stream = source.open() if source.exists() else gzip.open(p/'windows.jsonl.gz', 'rt', encoding='utf-8')
with stream:
    rows = [json.loads(line) for line in stream]
records = json.loads((p/'records.json').read_text())
groups = defaultdict(list)
for r in rows:
    groups[(r['coefficient'],r['method'],r['label_offset_s'])].append(r)

def stats(rs):
    valid = [r for r in rs if r['label_bpm'] is not None and r['fft_hr_bpm'] is not None]
    if not valid: return {'rows':len(rs),'valid':0}
    y = np.array([r['label_bpm'] for r in valid]); pred = np.array([r['fft_hr_bpm'] for r in valid])
    corr = float(np.corrcoef(y,pred)[0,1]) if min(y.std(),pred.std())>.01 else None
    return {'rows':len(rs),'valid':len(valid),'mae':float(np.mean(abs(y-pred))),
            'r':corr,'label_mean':float(y.mean()),'label_std':float(y.std()),
            'prediction_mean':float(pred.mean()),'prediction_std':float(pred.std()),
            'phase_variance_median':float(np.median([r['phase_variance'] for r in valid])),
            'hr_band_ratio_median':float(np.median([r['hr_band_ratio'] for r in valid])),
            'low_edge_fraction':float(np.mean(pred==46.875)),
            'peak_counts':Counter(pred.tolist()).most_common(5)}

result = {'total_rows':len(rows),'groups':{},'paired_controls':{}}
for key,rs in groups.items():
    coefficient,method,offset=key
    # Every offset compared on same record/window keys with complete labels.
    others = [groups[(coefficient,method,o)] for o in [0,-60,60,-120,120]]
    sets = [{(r['sample'],r['window_start']) for r in g if r['label_bpm'] is not None and r['fft_hr_bpm'] is not None} for g in others]
    common = set.intersection(*sets)
    matched = [r for r in rs if (r['sample'],r['window_start']) in common]
    entry = {'all':stats(rs),'matched':stats(matched),
             'subjects':{str(s):stats([r for r in matched if r['subject']==s]) for s in range(1,9)},
             'samples':{rec['sample']:stats([r for r in matched if r['sample']==rec['sample']]) for rec in records}}
    entry['subject_macro_mae'] = float(np.mean([v['mae'] for v in entry['subjects'].values()]))
    result['groups']['/'.join(map(str,key))]=entry

for coefficient in ['first_chirp','zero_doppler','adjacent_doppler']:
 for method in ['complex_real','complex_imag','unwrap','edacm']:
    prefix=f'{coefficient}/{method}'
    zero=result['groups'][prefix+'/0']
    comparisons={}
    for offset in [-60,60,-120,120]:
        shift=result['groups'][prefix+f'/{offset}']
        diffs=np.array([shift['subjects'][str(s)]['mae']-zero['subjects'][str(s)]['mae'] for s in range(1,9)])
        rng=np.random.default_rng(42)
        ci=np.quantile(rng.choice(diffs,(10000,8),replace=True).mean(axis=1),[.025,.975])
        comparisons[str(offset)]={'shift_minus_zero_macro_mae':float(diffs.mean()),'subject_bootstrap_95ci':ci.tolist(),
                                 'subjects_zero_better':int(np.sum(diffs>0)), 'subject_differences':diffs.tolist()}
    result['paired_controls'][prefix]=comparisons

# LOSO mean constant baseline on the same zero-offset matched windows (no fitting).
base=result['groups']['zero_doppler/unwrap/0']
eligible={r['sample'] for r in records}
rs=groups[('zero_doppler','unwrap',0)]
validkeys=set.intersection(*[{(r['sample'],r['window_start']) for r in groups[('zero_doppler','unwrap',o)] if r['label_bpm'] is not None and r['fft_hr_bpm'] is not None} for o in [0,-60,60,-120,120]])
v=[r for r in rs if (r['sample'],r['window_start']) in validkeys]
folds=[]
for s in range(1,9):
    train=[r['label_bpm'] for r in v if r['subject']!=s]; test=[r['label_bpm'] for r in v if r['subject']==s]
    mean=float(np.mean(train)); folds.append({'subject':s,'prediction':mean,'mae':float(np.mean(abs(np.array(test)-mean)))})
result['matched_constant_baseline']={'folds':folds,'subject_macro_mae':float(np.mean([f['mae'] for f in folds]))}
(p/'window_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
report=['# Direct phase window-level analysis','',f"176,640 rows represent 2,944 physical windows × 12 representations × 5 reference offsets. Overlapping windows are not independent subjects.",
        '', 'All offset comparisons below use the exact intersection of valid sample/window keys. CIs resample eight subject-level paired differences; exploratory, no multiplicity correction.',
        '', '| Representation | Matched windows | Subject macro MAE | Pooled r | Lowest HR-bin fraction |', '|---|---:|---:|---:|---:|']
for key,e in result['groups'].items():
 if key.endswith('/0'):
    a=e['matched']; report.append(f"| {key} | {a['valid']} | {e['subject_macro_mae']:.3f} | {a['r']:.3f} | {a['low_edge_fraction']:.3f} |")
report += ['',f"Matched-window LOSO training-subject mean constant macro MAE: {result['matched_constant_baseline']['subject_macro_mae']:.3f} bpm.",
           '', 'Pooled correlation can reflect between-subject heart-rate differences; inspect per-record and per-subject correlations in window_analysis.json.',
           'Phase variance of complex real/imaginary amplitude is in ADC units, and cannot be compared directly with phase variance in radians.',
           'Acquisition slope, frame rate and absolute radar/reference synchronization remain unverified. Reference is a 1 Hz HR series, not raw ECG.',
           '', '| Representation | Offset seconds | Shift − zero MAE | Subject bootstrap 95% CI | Subjects zero better |', '|---|---:|---:|---|---:|']
for key, cs in result['paired_controls'].items():
 for offset,c in cs.items():
    report.append(f"| {key} | {offset} | {c['shift_minus_zero_macro_mae']:.3f} | {c['subject_bootstrap_95ci']} | {c['subjects_zero_better']} |")
(p/'WINDOW_ANALYSIS.md').write_text('\n'.join(report),encoding='utf-8')
print('\n'.join(report[:22]))
print('zero_doppler unwrap subjects',result['groups']['zero_doppler/unwrap/0']['subjects'])
print('zero_doppler unwrap controls',result['paired_controls']['zero_doppler/unwrap'])
