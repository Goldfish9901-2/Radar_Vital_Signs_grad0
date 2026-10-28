"""Kaggle CPU: acquire the same six MAT members, decode I/Q only, no HR."""
from pathlib import Path
import ast
import hashlib
import io
import json
import struct
import time
import urllib.request
import zipfile
import zlib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from audit_ellipse_applicability import enumerate_fit, map_points, point_metrics, transfer_pair

HERE = Path(__file__).parent
CONFIG = json.loads((HERE/'audit_protocol.json').read_text())
OUT = Path('/kaggle/working/ellipse_applicability')
RAW = Path('/kaggle/temp/ellipse_applicability_raw')


def clean(value):
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer): return int(value)
    if isinstance(value,np.bool_): return bool(value)
    return value


def main():
    began=time.perf_counter(); OUT.mkdir(exist_ok=True); RAW.mkdir(parents=True,exist_ok=True)
    for name,digest in CONFIG['source_hashes'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    # Reuse only the original bounded ZIP acquisition functions, not ECG/HR code.
    tree=ast.parse((HERE/'run_synchronized_real_pilot.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['get_range','ArchiveDirectory','acquire']]
    namespace=dict(globals(),PROTO=json.loads((HERE/'protocol.json').read_text()),OUT=OUT,RAW=RAW)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original acquisition only>','exec'),namespace)
    namespace['acquire']()
    all_candidates=[]; evaluations=[]; transfers=[]; record_rows=[]
    (OUT/'clouds').mkdir(exist_ok=True)
    for record in CONFIG['records']:
        tag=record['record']; first=time.perf_counter()
        data=loadmat(RAW/(tag+'.mat'),squeeze_me=True,variable_names=['radar_i','radar_q','fs_radar'])
        assert set(data)-{'__header__','__version__','__globals__'}=={'radar_i','radar_q','fs_radar'}
        fs=float(data['fs_radar']); step=round(fs/100)
        assert abs(fs/step-100)<1e-12
        stop=max(90.,record['score_end_s'])
        xy=np.column_stack([data['radar_i'][:int(np.ceil(stop*fs)):step],data['radar_q'][:int(np.ceil(stop*fs)):step]]).astype(float)
        t=np.arange(len(xy))/100
        assert np.isfinite(xy).all() and t[-1]>=stop-.02
        np.savez_compressed(OUT/'clouds'/f'{tag}.npz',iq=xy,time_s=t,native_fs=fs,subsample_step=step)
        def segment(a,b): return xy[np.searchsorted(t,a):np.searchsorted(t,b)]
        fits=[enumerate_fit(segment(a,b),name,a,b) for name,a,b in CONFIG['fit_intervals']]
        primary=next(f for f in fits if f['fit']=='cal_0_30')
        assert primary['production_valid']==record['previous_primary_valid'], 'Original acceptance changed'
        all_candidates.append({'record':tag,'fits':fits})
        targets=list(CONFIG['evaluation_intervals'])
        start=record['score_start_s']
        targets += [(f'score_block{k+1}',start+30*k,start+30*(k+1)) for k in range(6)]
        targets += [('score_whole',start,record['score_end_s'])]
        for fit in fits:
            for candidate in fit['candidates']:
                if not candidate['geometry_defined']: continue
                for name,a,b in targets:
                    points=segment(a,b)
                    result=point_metrics(map_points(points,candidate))
                    evaluations.append(dict(record=tag,fit=fit['fit'],candidate=candidate['candidate'],
                        model_production_accepted=fit['production_valid'],candidate_passes_guards=candidate.get('passes_unchanged_guards',False),
                        evaluation=name,first_s=a,last_s=b,points=len(points),**result))
        fit_by_name={f['fit']:f for f in fits}
        for a,b in CONFIG['transfer_pairs']:
            fa,fb=fit_by_name[a],fit_by_name[b]
            result=transfer_pair(fa,fb,segment(fa['first_s'],fa['last_s']),segment(fb['first_s'],fb['last_s']),a+'->'+b)
            transfers.append(dict(record=tag,**result))
        record_rows.append({'record':tag,'native_fs':fs,'diagnostic_sample_rate':100.,'points_saved':len(xy),
                            'primary_accepted_unchanged':primary['production_valid'],'elapsed_seconds':time.perf_counter()-first})
        print('AUDITED',tag,'primary accepted:',primary['production_valid'],flush=True)
    (OUT/'candidate_diagnostics.json').write_text(json.dumps(clean(all_candidates),indent=2,allow_nan=False))
    (OUT/'block_metrics.json').write_text(json.dumps(clean(evaluations),indent=2,allow_nan=False))
    pd.DataFrame(clean(evaluations)).to_csv(OUT/'block_metrics.csv',index=False)
    pd.DataFrame(clean(transfers)).to_csv(OUT/'transfer_metrics.csv',index=False)
    (OUT/'transfer_metrics.json').write_text(json.dumps(clean(transfers),indent=2,allow_nan=False))
    pd.DataFrame(record_rows).to_csv(OUT/'record_execution.csv',index=False)
    (OUT/'execution.json').write_text(json.dumps({'elapsed_seconds':time.perf_counter()-began,'records':len(record_rows),
        'decoded_mat_variables':['radar_i','radar_q','fs_radar'],'ecg_or_hr_arrays_decoded':False,
        'hr_predictions_generated':False,'production_code_changed':False,'other27_subjects_read':False},indent=2))


if __name__=='__main__': main()
