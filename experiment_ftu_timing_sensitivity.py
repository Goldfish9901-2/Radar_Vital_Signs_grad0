"""Prespecified offset sensitivity; never select a lag by test HR or prediction error."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from src.data.ftu_timebase import frame_times, labels_at_frames
from src.data.audit_ftu_p0 import digest, write_json


def validate_mapping():
    t=frame_times(5,2.,radar_start_on_reference_clock_s=-.5)
    assert np.array_equal(t,[-.5,0,.5,1,1.5])
    y=labels_at_frames([0,1],[60,80],t)
    assert np.isnan(y[[0,4]]).all() and np.array_equal(y[1:4],[60,70,80])
    for bad in ([0,0],[1,0]):
        try: labels_at_frames(bad,[60,80],t)
        except ValueError: pass
        else: raise AssertionError('Invalid timestamps accepted')
    try: frame_times(256,20.,radar_start_on_reference_clock_s=np.nan)
    except ValueError: pass
    else: raise AssertionError('Missing offset accepted')


def main():
    started=time.perf_counter(); validate_mapping()
    out=Path('experiment_runs/ftu_sktime_baseline_2026-10-16');out.mkdir(exist_ok=True,parents=True)
    archive=Path('experiment_runs/ftu_source_audit_2026-10-13')
    mp=archive/'e2a_evidence/training_exports/manifest.csv'
    manifest=pd.read_csv(mp)
    tp=Path('experiment_runs/ftu_p0_audit_2026-10-14/raw_timing_audit.csv')
    timing=pd.read_csv(tp)
    rawroot=Path('tmp/ftu_p0_result/ftu_p0')
    refs={r['source']:r for r in json.loads((rawroot/'reference_times.json').read_text())}
    rows=[]
    for record in timing[timing.saved_window_count.notna()].itertuples():
        r=refs[record.hr_source]; path=rawroot/'raw_time_evidence'/r['archived_file']
        assert digest(path)==r['sha256']
        df=pd.read_csv(path,skiprows=3,header=None)
        rt=pd.to_timedelta(df.iloc[:,1]).dt.total_seconds().to_numpy()
        hr=pd.to_numeric(df.iloc[:,2]).to_numpy()
        part=manifest[manifest.sample_tag==record.sample_tag]
        offsets={f'fixed_{v:+d}s':float(v) for v in [-4,-2,-1,0,1,2,4]}
        # HR header minus radar start has opposite sign to radar on HR clock.
        if np.isfinite(record.header_hr_minus_radar_start_s):
            h=-record.header_hr_minus_radar_start_s
            offsets.update(header=h,header_minus_1s=h-1,header_plus_1s=h+1)
        for scenario,offset in offsets.items():
            f=frame_times(1200,20.,radar_start_on_reference_clock_s=offset)
            y=labels_at_frames(rt,hr,f)
            for w in part.itertuples():
                values=y[int(w.window_start):int(w.window_end)]
                coverage=float(np.isfinite(values).mean())
                new=float(np.nanmean(values))
                rows.append(dict(sample_tag=record.sample_tag,group_key=w.group_key,split=w.split,
                    window_start=int(w.window_start),scenario=scenario,offset_s=offset,coverage=coverage,
                    passes_80pct_coverage=coverage>=.8,full_coverage=coverage==1.,
                    historical_label=float(w.label_heart_rate),candidate_label=new,
                    label_delta_bpm=new-w.label_heart_rate,abs_label_delta_bpm=abs(new-w.label_heart_rate),
                    frame_rate_basis=record.frame_rate_basis))
    data=pd.DataFrame(rows);data.to_csv(out/'timing_sensitivity_windows.csv',index=False)
    summary=[]
    for scenario,d in data.groupby('scenario',sort=True):
        for subset,mask in [('all',np.ones(len(d),dtype=bool)),('full_coverage',d.full_coverage.to_numpy())]:
            selected=d[mask]
            summary.append(dict(scenario=scenario,subset=subset,windows=len(selected),records=selected.sample_tag.nunique(),
                pass_80pct=int(selected.passes_80pct_coverage.sum()),
                min_coverage=float(selected.coverage.min()),mean_abs_delta=float(selected.abs_label_delta_bpm.mean()),
                p95_abs_delta=float(selected.abs_label_delta_bpm.quantile(.95)),max_abs_delta=float(selected.abs_label_delta_bpm.max())))
    pd.DataFrame(summary).to_csv(out/'timing_sensitivity_summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=pd.DataFrame(summary);s=s[s.subset=='full_coverage']
    fig,ax=plt.subplots(figsize=(11,4))
    ax.bar(s.scenario,s.mean_abs_delta,label='mean absolute label change')
    ax.scatter(s.scenario,s.p95_abs_delta,color='orange',label='95th percentile')
    ax.set(ylabel='Change from historical label (bpm)',title='Prespecified timing assumptions; fully covered windows only')
    ax.tick_params(axis='x',rotation=35);ax.legend();fig.tight_layout()
    fig.savefig(out/'timing_sensitivity.png',dpi=160);plt.close(fig)
    write_json(out/'timing_summary.json',dict(elapsed_sec=time.perf_counter()-started,rows=len(data),
        records=440,windows=3520,source_sha256=digest(__file__),manifest_sha256=digest(mp),
        timing_audit_sha256=digest(tp),mapping_contract_checks_passed=True,
        scenarios='fixed offsets -4,-2,-1,0,+1,+2,+4 seconds; negative header difference and +/-1 second',
        caveat='Offsets are sensitivity assumptions, not measured synchronization. Header scenarios cover 412 records; fixed scenarios cover 440.',
        selected_best_offset=None,synchronization_verified=False,retraining_allowed=False))


if __name__=='__main__': main()
