"""Reaudit existing eight-subject outputs; no new ADC processing or fitting."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def metric(rows):
    valid=[r for r in rows if r['label_bpm'] is not None and r['fft_hr_bpm'] is not None]
    pred=np.array([r['fft_hr_bpm'] for r in valid]);y=np.array([r['label_bpm'] for r in valid])
    return {'windows':len(rows),'valid':len(valid),'mae_bpm':float(abs(pred-y).mean()),
            'rmse_bpm':float(np.sqrt(np.mean((pred-y)**2))),
            'pearson_r':float(np.corrcoef(pred,y)[0,1]) if min(pred.std(),y.std())>.01 else None,
            'prediction_std_bpm':float(pred.std()),'reference_std_bpm':float(y.std()),
            'low_edge_fraction':float(np.mean(pred==46.875))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',type=Path,default=Path('experiment_runs/bgt60_direct_phase_2026-10-10'))
    p.add_argument('--output-dir',type=Path,default=Path('tmp/bgt60_eight_subject_baseline'))
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True);clock=time.perf_counter()
    source=args.input_dir/'windows.jsonl.gz';rows=[]
    with gzip.open(source,'rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['coefficient']=='zero_doppler' and row['method']=='unwrap' and row['label_offset_s']==0:rows.append(row)
    assert len(rows)==2944 and len({r['sample'] for r in rows})==32
    subjects={};records={}
    for subject in range(1,9):
        test=[r for r in rows if r['subject']==subject]
        subjects[str(subject)]=metric(test)
        train_y=[r['label_bpm'] for r in rows if r['subject']!=subject and r['label_bpm'] is not None]
        test_y=[r['label_bpm'] for r in test if r['label_bpm'] is not None and r['fft_hr_bpm'] is not None]
        mean=float(np.mean(train_y))
        subjects[str(subject)]['other_subject_mean_bpm']=mean
        subjects[str(subject)]['constant_mae_bpm']=float(np.mean(abs(np.array(test_y)-mean)))
    for tag in sorted({r['sample'] for r in rows}):records[tag]=metric([r for r in rows if r['sample']==tag])
    protocol={'source':'existing completed 32-record direct-phase audit; no new ADC run',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'baseline':'native 512 FFT, fast-time DC mean; strongest local nonDC peak; whole-record energy RX; fixed per record; zero Doppler unwrap; 384-frame Hann FFT; 30Hz assumed',
        'evaluation':'full reference-valid windows; subject macro MAE; constant uses reference means of other seven subjects only',
        'limitations':'all eight subjects appeared in earlier exploratory diagnostics; no untouched independent cohort; shared start/range calibration unverified',
        'selection_changes':False,'training':False}
    result={'pooled':metric(rows),'subjects':subjects,'records':records,
        'subject_macro_mae_bpm':float(np.mean([s['mae_bpm'] for s in subjects.values()])),
        'subject_macro_constant_mae_bpm':float(np.mean([s['constant_mae_bpm'] for s in subjects.values()])),
        'record_r_arithmetic_mean':float(np.mean([s['pearson_r'] for s in records.values() if s['pearson_r'] is not None]))}
    for name,data in [('protocol',protocol),('summary',result)]:
        (out/f'{name}.json').write_text(json.dumps(data,indent=2,allow_nan=False))
    with (out/'baseline_windows.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    report=['# Eight-subject frozen direct-phase baseline reaudit','',
        'This reuses existing actual ADC-derived outputs, not a new experiment or independent cohort. No algorithm/ROI changes, no training.',
        '', '| Subject | Full valid/total | FFT MAE | Other-seven-subject mean MAE | Pearson r |','|---|---:|---:|---:|---:|']
    for subject,s in subjects.items():report.append(f"| {subject} | {s['valid']}/{s['windows']} | {s['mae_bpm']:.3f} | {s['constant_mae_bpm']:.3f} | {s['pearson_r']:.3f} |")
    report+=['',f"Subject macro FFT MAE: {result['subject_macro_mae_bpm']:.3f} bpm; constant: {result['subject_macro_constant_mae_bpm']:.3f} bpm.",
             f"Mean within-record r (descriptive arithmetic mean): {result['record_r_arithmetic_mean']:.3f}.",
             'Overlapping windows are not independent subjects. p03 is exploratory; all others were also examined previously. Absolute synchronization and frame rate remain unverified.']
    (out/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    (out/'timing.json').write_text(json.dumps({'seconds':time.perf_counter()-clock,'execution':'local uv','training':False},indent=2))
    print('\n'.join(report))


if __name__=='__main__':main()
