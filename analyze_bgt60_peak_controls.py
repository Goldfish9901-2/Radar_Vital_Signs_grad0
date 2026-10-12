"""Candidate-peak attribution controls; no label-assisted deployed estimator."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from analyze_bgt60_spectral_attribution import DF


def statistics(rows):
    result={'windows':len(rows)}
    for key in ['prediction_near_harmonic','prediction_near_shifted_comb','reference_hr_in_top3','reference_hr_near_harmonic']:
        values=[r[key] for r in rows if r[key] is not None]
        result[key+'_fraction']=float(np.mean(values)) if values else None
    result['wrong_peak_harmonic_and_reference_harmonic']=sum(bool(r['prediction_near_harmonic'] and r['reference_hr_near_harmonic']) for r in rows)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--audit-dir',type=Path,default=Path('tmp/bgt60_p03_harmonic_audit'))
    p.add_argument('--trace-dir',type=Path,default=Path('tmp/bgt60_p03_pair_v2_results/pair_audit'))
    p.add_argument('--output-dir',type=Path,default=Path('tmp/bgt60_peak_controls'))
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True);clock=time.perf_counter()
    protocol={'thresholds_unchanged':True,'tolerance_hz':DF,'offsets_s':[0,-60,60,-120,120],
        'coverage':'exact same windows for all offsets, complete labels',
        'nonoverlap':'window_start mod 384 == 0; still temporally dependent',
        'rr_agrees':'reference RR error <= one FFT bin in bpm; reference-assisted attribution stratum only',
        'stratification':'HR error >5 bpm based on zero-offset prediction',
        'status':'post hoc mechanism diagnostic; no bin/peak selection or threshold fitting',
        'training':False,'no_new_estimator':True}
    result={};all_rows={}
    for distance in ['0.3m','0.6m']:
        source=json.loads((args.audit_dir/f'p03_{distance}_windows.json').read_text())
        valid=[r for r in source if r['reference_hr_bpm'] is not None and r['prediction_bpm'] is not None]
        strata={'all_wrong':[r for r in valid if r['absolute_error_bpm']>5],
                'nonoverlap_wrong':[r for r in valid if r['absolute_error_bpm']>5 and r['window_start']%384==0],
                'rr_agrees_wrong':[r for r in valid if r['absolute_error_bpm']>5 and r['reference_rr_error_bpm'] is not None and r['reference_rr_error_bpm']<=DF*60]}
        result[distance]={'strata':{k:statistics(v) for k,v in strata.items()}}
        with np.load(args.trace_dir/f'p03_{distance}_traces.npz') as d:ref=d['reference']
        evaluated=[]
        for r in valid:
            labels={}
            for offset in protocol['offsets_s']:
                y=np.interp(np.arange(r['window_start'],r['window_start']+384)/30+offset,ref[:,0],ref[:,1],left=np.nan,right=np.nan)
                labels[offset]=float(y.mean()) if np.isfinite(y).all() else None
            if any(y is None for y in labels.values()):continue
            hits={str(offset):bool(any(abs(peak['bpm']-y)<=DF*60+1e-10 for peak in r['top_hr_peaks'])) for offset,y in labels.items()}
            evaluated.append({'window_start':r['window_start'],'wrong_zero_offset':r['absolute_error_bpm']>5,
                              'top3_hits':hits,'reference_bpm':labels})
        result[distance]['candidate_controls']={}
        for group,rs in [('all_matched',evaluated),('wrong_matched',[r for r in evaluated if r['wrong_zero_offset']]),
                          ('nonoverlap_matched',[r for r in evaluated if r['window_start']%384==0])]:
            rates={str(o):float(np.mean([r['top3_hits'][str(o)] for r in rs])) if rs else None for o in protocol['offsets_s']}
            result[distance]['candidate_controls'][group]={'windows':len(rs),'hit_rates':rates}
        all_rows[distance]=evaluated
    for name,data in [('protocol',protocol),('summary',result),('matched_windows',all_rows)]:
        (out/f'{name}.json').write_text(json.dumps(data,indent=2,allow_nan=False))
    report=['# Candidate-peak controls (local follow-up)', '',
        'This is diagnostic use of reference labels, not a practical peak selector or oracle MAE. No estimation rule is changed.',
        'Top3 proximity tolerance stays one FFT bin; exact common windows across offsets. Negative offsets are fixed, not optimized.',
        '', '| Record | Group | Windows | Hit at 0 | -60 | +60 | -120 | +120 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for distance,entry in result.items():
        for group,s in entry['candidate_controls'].items():
            report.append(f"| {distance} | {group} | {s['windows']} | "+' | '.join(f"{s['hit_rates'][str(o)]:.3f}" if s['hit_rates'][str(o)] is not None else 'NA' for o in protocol['offsets_s'])+' |')
    report+=['','Reference-assisted RR agreement strata do not authorize discarding windows in the algorithm. Nonoverlap reduces shared samples but is not independent-subject validation.',
             'Slow HR, coarse frequency bins and multiple candidate peaks can yield high recall without synchronous HR tracking. Harmonic and half-order masks have different edge coverage; coincidences do not prove source.',
             'Do not choose new thresholds or a best time shift using these post hoc p03 labels.']
    (out/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    (out/'timing.json').write_text(json.dumps({'seconds':time.perf_counter()-clock,'execution':'local uv','training':False},indent=2))
    print('\n'.join(report));print(json.dumps({d:r['strata'] for d,r in result.items()},indent=2))


if __name__=='__main__':main()
