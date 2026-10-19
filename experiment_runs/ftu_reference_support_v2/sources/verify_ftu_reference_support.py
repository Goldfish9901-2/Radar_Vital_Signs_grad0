"""Independent support/label verification plus original Polar CSV provenance."""
import csv
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT=Path('experiment_runs/ftu_reference_support_v2')


def main():
    begin=time.perf_counter();rows=pd.read_csv(ROOT/'window_support.csv')
    refs=pd.read_csv(ROOT/'polar_reference_values.csv')
    original=json.loads(Path('tmp/ftu_p0_result/ftu_p0/reference_times.json').read_text())
    records=pd.read_csv(ROOT/'selected_records.csv');provenance=[];checked=0
    allrefs={};rawroot=Path('tmp/ftu_p0_result/ftu_p0/raw_time_evidence')
    for _,r in records.iterrows():
        tag=r.sample_tag;pid=int(tag[1:3]);matches=[]
        for item in original:
            if f'/P{pid}_HR_refs/' not in item['source']:continue
            header=list(csv.reader(item['first_three_lines']))
            if header[1][2]==r.date and header[1][3]==r.hr_start_time:matches.append(item)
        assert len(matches)==1,(tag,len(matches))
        item=matches[0];path=rawroot/item['archived_file'];raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==item['sha256']
        data=pd.read_csv(path,skiprows=3,header=None)
        t=pd.to_timedelta(data.iloc[:,1]).dt.total_seconds().to_numpy()
        y=pd.to_numeric(data.iloc[:,2]).to_numpy()
        saved=refs[refs.record==tag]
        np.testing.assert_array_equal(t,saved.time_second)
        np.testing.assert_array_equal(y,saved.hr_bpm)
        assert (np.diff(t)==1).all() and np.isfinite(y).all()
        allrefs[tag]=(t,y)
        header=list(csv.reader(item['first_three_lines']))
        provenance.append(dict(record=tag,original_source=item['source'],archived_csv_sha256=item['sha256'],
            original_points=len(t),original_first=t[0],original_last=t[-1],declared_duration=header[1][4],
            original_equals_archived_reference=True,date=header[1][2],polar_start=header[1][3],
            measured_radar_polar_sync=False))
        subset=rows[rows.record==tag]
        for _,x in subset.iterrows():
            s=int(x.start_frame);length=int(x.length);off=x.offset;shift=x['shift']
            true=(s/20+off>=t[0]) and ((s+length-1)/20+off<=t[-1])
            control=(s/20+off+shift>=t[0]) and ((s+length-1)/20+off+shift<=t[-1])
            assert true==x.true_supported and control==x.control_supported
            assert bool(s>=256 and true and control)==x.pair_common
            checked+=1
    pd.DataFrame(provenance).to_csv(ROOT/'original_csv_provenance.csv',index=False)
    # Recompute selected intersection and global timestamp-only combination ranking.
    selection=json.loads((ROOT/'timestamp_only_selection.json').read_text())
    candidate=[-20,-15,-10,-5,5,10,15,20];selected=pd.read_csv(ROOT/'selected_support.csv')
    for length in [512,768]:
        options=[]
        for shifts in itertools.combinations(candidate,4):
            counts=[]
            for tag,(t,y) in allrefs.items():
                valid=[]
                for start in range(0,1200-length+1,20):
                    # Endpoints suffice only because the independently checked references are contiguous1Hz.
                    good=start>=256
                    for off in range(3):
                        for shift in [0]+list(shifts):
                            good &= start/20+off+shift>=t[0] and (start+length-1)/20+off+shift<=t[-1]
                    if good:valid.append(start)
                counts.append(len(valid))
                if list(shifts)==selection[str(length)]['shifts']:
                    for off in range(3):
                        row=selected[(selected.record==tag)&(selected.length==length)&(selected.offset==off)].iloc[0]
                        assert valid==json.loads(row.all_offsets_common_starts)
            score=(sum(n>=5 for n in counts),min(counts),sum(counts),min(abs(s) for s in shifts),sum(abs(s) for s in shifts))
            options.append((score,shifts))
        best=sorted(options,key=lambda v:tuple(-x for x in v[0])+v[1])[0]
        assert list(best[1])==selection[str(length)]['shifts']
    saved=pd.read_csv(ROOT/'label_window_differences.csv');maxerr=0.
    for _,r in saved.iterrows():
        t,y=allrefs[r.record];f=interp1d(t,y,bounds_error=False,fill_value=np.nan)
        q=np.arange(int(r.length))*.05+r.start_frame/20+r.offset
        a=f(q).mean();b=f(q+r['shift']).mean()
        maxerr=max(maxerr,abs(a-r.true_hr),abs(b-r.control_hr))
        np.testing.assert_allclose([a,b,abs(a-b)],[r.true_hr,r.control_hr,r.absolute_difference],atol=1e-10)
    dist=pd.read_csv(ROOT/'control_distinguishability.csv')
    for _,r in dist.iterrows():
        w=saved[(saved.record==r.record)&(saved.length==r.length)&(saved.offset==r.offset)&(saved['shift']==r['shift'])]
        assert len(w)==r.common_windows
        expected=bool(len(w)>=5 and (w.absolute_difference>1200/r.length).mean()>=.6)
        assert expected==r.distinguishable
    result=dict(elapsed_seconds=time.perf_counter()-begin,original_csv_matches=len(provenance),
        support_window_rows_checked=checked,label_window_rows_checked=len(saved),
        maximum_label_error=maxerr,control_selection_independently_verified=True,
        radar_scores_accessed=False,original_source_index_sha256=hashlib.sha256(Path('tmp/ftu_p0_result/ftu_p0/reference_times.json').read_bytes()).hexdigest())
    (ROOT/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
