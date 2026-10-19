"""Two-stage reference-only feasibility audit. Never reads radar data or scores."""
import argparse
import hashlib
import itertools
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('experiment_runs/ftu_reference_support_v2')
P=json.loads((ROOT/'protocol.json').read_text())
SRC=Path(P['source'])
SHIFTS=P['candidate_shifts_seconds']


def valid_queries(t,q):
    k=np.searchsorted(t,q,side='left')
    upper=np.clip(k,0,len(t)-1);lower=np.clip(k-1,0,len(t)-1)
    exact=np.isclose(q,t[upper],rtol=0,atol=1e-9)
    inside=(q>=t[0])&(q<=t[-1])
    gap=t[upper]-t[lower]
    return inside&(exact|((k>0)&(k<len(t))&(gap<=2*np.median(np.diff(t)))))


def query(start,length,off,shift):
    return np.arange(start,start+length)/20+off+shift


def hashfile(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def support():
    begin=time.perf_counter(); details=[]; inventories=[]; masks={}; times=[]
    for name,h in json.loads((ROOT/'source_archive_hashes.json').read_text()).items():
        path=SRC/name;assert hashfile(path)==h
        # Only reference_time is accessed in this stage. np.load is lazy.
        with np.load(path) as d:t=d['reference_time'].astype(float)
        assert len(t)>=2 and np.isfinite(t).all() and (np.diff(t)>0).all()
        tag=path.stem
        inventories.append(dict(record=tag,points=len(t),first_second=t[0],last_second=t[-1],
            median_step=float(np.median(np.diff(t))),maximum_gap=float(np.diff(t).max())))
        times.extend(dict(record=tag,time_second=float(v)) for v in t)
        for length in P['lengths_frames']:
            starts=np.arange(0,1200-length+1,20); post=starts>=256
            use={}
            for off in P['offsets_seconds']:
                for shift in [0]+SHIFTS:
                    use[off,shift]=np.array([valid_queries(t,query(s,length,off,shift)).all() for s in starts])
                for shift in SHIFTS:
                    for j,s in enumerate(starts):
                        details.append(dict(record=tag,length=length,offset=off,shift=shift,
                            start_frame=int(s),center_second=(s+length/2)/20,end_second=(s+length-1)/20,
                            post_calibration=bool(post[j]),true_supported=bool(use[off,0][j]),
                            control_supported=bool(use[off,shift][j]),
                            pair_common=bool(post[j]&use[off,0][j]&use[off,shift][j])))
            masks[tag,length]=(starts,post,use)
    pd.DataFrame(inventories).to_csv(ROOT/'reference_inventory.csv',index=False)
    pd.DataFrame(times).to_csv(ROOT/'reference_timestamps.csv',index=False)
    details=pd.DataFrame(details);details.to_csv(ROOT/'window_support.csv',index=False)
    pair=[]
    for (tag,length,off,shift),g in details.groupby(['record','length','offset','shift']):
        a=g[g.pair_common]
        pair.append(dict(record=tag,length=length,offset=off,shift=shift,common_windows=len(a),
            start_frames=json.dumps(a.start_frame.tolist()),centers_seconds=json.dumps(a.center_second.tolist())))
    pd.DataFrame(pair).to_csv(ROOT/'pair_support_summary.csv',index=False)
    selection={};combination_rows=[];final=[]
    for length in P['lengths_frames']:
        options=[]
        for shifts in itertools.combinations(SHIFTS,4):
            counts=[]
            for tag in sorted(x['record'] for x in inventories):
                starts,post,use=masks[tag,length];common=post.copy()
                for off in range(3):
                    for shift in [0]+list(shifts):common &= use[off,shift]
                counts.append(int(common.sum()))
            score=(sum(n>=5 for n in counts),min(counts),sum(counts),min(abs(s) for s in shifts),sum(abs(s) for s in shifts))
            options.append((score,shifts,counts))
            combination_rows.append(dict(length=length,shifts=json.dumps(shifts),support_records=score[0],
                minimum_windows=score[1],total_windows=score[2],record_counts=json.dumps(counts)))
        # Fixed lexicographically ascending signed-tuple tie break.
        best=sorted(options,key=lambda x:tuple(-v for v in x[0])+x[1])[0]
        selection[str(length)]=dict(shifts=list(best[1]),support_records=best[0][0],record_counts=best[2],
            sorted_records=sorted(x['record'] for x in inventories),ranking_score=list(best[0]))
        for tag in sorted(x['record'] for x in inventories):
            starts,post,use=masks[tag,length]; common_all=post.copy()
            for off in range(3):
                for shift in [0]+list(best[1]):common_all &= use[off,shift]
            for off in range(3):
                per_off=post.copy()
                for shift in [0]+list(best[1]):per_off &= use[off,shift]
                final.append(dict(record=tag,length=length,offset=off,shifts=json.dumps(best[1]),
                    offset_common_count=int(per_off.sum()),offset_common_starts=json.dumps(starts[per_off].tolist()),
                    all_offsets_common_count=int(common_all.sum()),all_offsets_common_starts=json.dumps(starts[common_all].tolist()),
                    support_evaluable=bool(common_all.sum()>=5)))
    pd.DataFrame(combination_rows).to_csv(ROOT/'candidate_combination_support.csv',index=False)
    pd.DataFrame(final).to_csv(ROOT/'selected_support.csv',index=False)
    (ROOT/'timestamp_only_selection.json').write_text(json.dumps(selection,indent=2))
    (ROOT/'support_execution.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-begin,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        reference_hr_accessed=False,radar_arrays_accessed=False,radar_scores_accessed=False,
        selection_sha256=hashfile(ROOT/'timestamp_only_selection.json'),command='uv run --locked --no-sync python audit_ftu_reference_support.py support'),indent=2))
    print(json.dumps(selection,indent=2))


def interpolate(t,y,q):
    a=np.interp(q,t,y,left=np.nan,right=np.nan)
    a[~valid_queries(t,q)]=np.nan
    return a


def labels():
    begin=time.perf_counter();sel=json.loads((ROOT/'timestamp_only_selection.json').read_text())
    execution=json.loads((ROOT/'support_execution.json').read_text())
    assert execution['selection_sha256']==hashfile(ROOT/'timestamp_only_selection.json')
    supported=pd.read_csv(ROOT/'selected_support.csv');window_rows=[];scores=[];reference=[]
    for name,h in json.loads((ROOT/'source_archive_hashes.json').read_text()).items():
        path=SRC/name;assert hashfile(path)==h
        with np.load(path) as d:t=d['reference_time'].astype(float);y=d['reference_hr'].astype(float)
        assert t.shape==y.shape
        tag=path.stem
        reference.extend(dict(record=tag,time_second=float(a),hr_bpm=float(b)) for a,b in zip(t,y))
        for length in P['lengths_frames']:
            for off in range(3):
                row=supported[(supported.record==tag)&(supported.length==length)&(supported.offset==off)].iloc[0]
                common=np.array(json.loads(row.all_offsets_common_starts),dtype=int)
                for shift in SHIFTS:
                    # Selected controls use identical final support; others only describe their own pair support.
                    selected=shift in sel[str(length)]['shifts']
                    starts=common if selected else np.array([s for s in np.arange(0,1200-length+1,20)
                        if s>=256 and valid_queries(t,query(s,length,off,0)).all() and valid_queries(t,query(s,length,off,shift)).all()])
                    deltas=[]
                    for start in starts:
                        true=interpolate(t,y,query(start,length,off,0)).mean()
                        wrong=interpolate(t,y,query(start,length,off,shift)).mean()
                        delta=abs(true-wrong);deltas.append(delta)
                        window_rows.append(dict(record=tag,length=length,offset=off,shift=shift,selected_control=selected,
                            support_kind='selected_all_offsets_all_controls' if selected else 'unselected_pair_only',
                            start_frame=int(start),true_hr=true,control_hr=wrong,absolute_difference=delta,native_bin_bpm=1200/length))
                    diff=np.array(deltas)
                    finite=bool(np.isfinite(diff).all())
                    fraction=float((diff>1200/length).mean()) if len(diff) and finite else np.nan
                    scores.append(dict(record=tag,length=length,offset=off,shift=shift,selected_control=selected,
                        common_windows=len(diff),labels_finite=finite,
                        mean_absolute_difference=float(diff.mean()) if len(diff) else np.nan,
                        median_absolute_difference=float(np.median(diff)) if len(diff) else np.nan,
                        maximum_absolute_difference=float(diff.max()) if len(diff) else np.nan,
                        fraction_above_native=fraction,native_bin_bpm=1200/length,
                        distinguishable=bool(len(diff)>=5 and finite and fraction>=.6)))
    pd.DataFrame(reference).to_csv(ROOT/'polar_reference_values.csv',index=False)
    pd.DataFrame(window_rows).to_csv(ROOT/'label_window_differences.csv',index=False)
    df=pd.DataFrame(scores);df.to_csv(ROOT/'control_distinguishability.csv',index=False)
    decision=[]
    for (tag,length),group in df[df.selected_control].groupby(['record','length']):
        n=int(group.common_windows.min());distinct=bool(group.distinguishable.all())
        decision.append(dict(record=tag,length=length,common_windows=n,support_evaluable=n>=5,
            distinguishable_all_offsets_all_controls=distinct,evaluable=bool(n>=5 and distinct),
            reason='evaluable' if n>=5 and distinct else 'insufficient_common_support' if n<5 else 'insufficient_label_separation'))
    decision=pd.DataFrame(decision);decision.to_csv(ROOT/'record_feasibility.csv',index=False)
    count=int(decision[(decision.length==512)&decision.evaluable].shape[0])
    gate=dict(primary_length=512,evaluable_records=count,required_records=5,
              stage2_authorized_by_data_gate=count>=5,stage2_executed=False,radar_scores_computed=False,
              decision='prepare frozen extended-signal protocol' if count>=5 else 'do not reconstruct/score extended RDA; investigate reference coverage and synchronization provenance',
              selection_sha256=hashfile(ROOT/'timestamp_only_selection.json'))
    (ROOT/'decision.json').write_text(json.dumps(gate,indent=2))
    (ROOT/'label_execution.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-begin,
        command='uv run --locked --no-sync python audit_ftu_reference_support.py labels',
        local=True,radar_arrays_accessed=False,radar_scores_accessed=False,numpy=np.__version__,pandas=pd.__version__),indent=2))
    print(decision.to_string(index=False));print(json.dumps(gate,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['support','labels']);args=parser.parse_args()
    {'support':support,'labels':labels}[args.stage]()
