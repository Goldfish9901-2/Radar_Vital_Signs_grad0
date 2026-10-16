"""Validate downloaded P0 artifacts and compare raw FTU clock sidecars to labels."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np
import pandas as pd

from src.data.audit_ftu_p0 import digest, write_json


def sample_tag(source):
    parts = source.split('/')
    ix = next(i for i, s in enumerate(parts) if re.match(r'P\d+_', s))
    pid = int(re.match(r'P(\d+)_', parts[ix])[1])
    scenarios = {'1. Distance Scenario':'Distance', '2. Orientation Scenario':'Orientation',
                 '3. Angle Scenario':'Angle', '4. Elevated':'Elevated'}
    scenario = scenarios[parts[ix+1]]
    distance = '80 cm' if '_HR_refs' in parts[ix] and scenario=='Elevated' else parts[ix+2]
    for token in (' ', '/', '\\', '-', '.'): distance = distance.replace(token, '_')
    repeat = re.search(r'\d+', Path(parts[-1]).stem)[0] if '_HR_refs' in parts[ix] else parts[ix+3]
    return f'p{pid:02d}_{scenario}_{distance}_r{repeat}'


def run(remote, out, archive):
    started=time.perf_counter()
    summary=json.loads((remote/'remote_summary.json').read_text())
    manifest=pd.read_csv(archive/'e2a_evidence/training_exports/manifest.csv')
    modes=pd.read_csv(remote/'exported_window_modes.csv')
    assert summary['manifest_sha256']==digest(archive/'e2a_evidence/training_exports/manifest.csv')
    keys=['sample_tag','window_start','split']
    assert not modes.duplicated(keys).any()
    merged=manifest.merge(modes, on=keys, how='outer', indicator=True)
    assert (merged['_merge']=='both').all() and len(merged)==len(manifest)==summary['windows']
    assert int(modes.all_arrays_exactly_equal.sum()) == summary['exactly_duplicate_windows']
    assert int((modes.numerical_rank == 1).sum()) == summary['rank1_windows']
    assert int(((modes.min_weight == 1) & (modes.max_weight == 1)).sum()) == summary['weights_all_one_windows']
    actual_checks=[]
    with np.load(remote/'exported_examples.npz') as z:
        for key in z.files:
            x=z[key]
            identical=all(np.array_equal(x[0], row) for row in x[1:])
            actual_checks.append(dict(key=key, all_channels_identical=identical, shape=list(x.shape)))
    out.mkdir(exist_ok=True, parents=True)
    for filename in ('remote_summary.json','exported_window_modes.csv','exported_examples.npz','exported_example_metadata.json'):
        shutil.copyfile(remote/filename, out/filename)
    logs=json.loads((remote/'radar_logs.json').read_text())
    refs=json.loads((remote/'reference_times.json').read_text())
    by_tag={}; failures=[]
    for rec in logs:
        try: tag=sample_tag(rec['source'])
        except Exception as exc:
            failures.append(dict(source=rec['source'], error=repr(exc))); continue
        row=by_tag.setdefault(tag, dict(sample_tag=tag))
        rawpath=remote/'raw_time_evidence'/rec['archived_file']
        assert digest(rawpath)==rec['sha256']
        if rec['kind']=='radar_config_log':
            row['radar_log_sha256']=rec['sha256']
            row['radar_source']=rec['source']
            row['frame_config_count']=len(rec['frame_configs'])
            row['sensor_start_count']=len(rec['sensor_starts'])
            if rec['frame_configs']:
                fields=rec['frame_configs'][-1].split('API:FrameConfig,')[1].split(',')
                row['configured_frames']=int(fields[2]); row['chirp_loops']=int(fields[3])
                row['frame_period_ticks']=int(fields[4])
                row['configured_frame_period_s']=int(fields[4])*5e-9
                row['configured_fs_hz']=1/row['configured_frame_period_s']
                row['last_frame_config']=rec['frame_configs'][-1]
            if rec['sensor_starts']:
                row['last_sensor_start']=rec['sensor_starts'][-1].split(': API:')[0]
        else:
            row['capture_log_sha256']=rec['sha256']; row['capture_log_text']=rec['text']
            for label, key in [('Out of sequence count', 'out_of_sequence_packets'),
                               ('Number of received packets', 'received_packets'),
                               ('Number of zero filled packets', 'zero_filled_packets'),
                               ('Number of zero filled bytes', 'zero_filled_bytes')]:
                match=re.search(re.escape(label)+r'\s*-\s*(\d+)',rec['text'])
                if match: row[key]=int(match[1])
    for rec in refs:
        try: tag=sample_tag(rec['source'])
        except Exception as exc:
            failures.append(dict(source=rec['source'], error=repr(exc))); continue
        row=by_tag.setdefault(tag, dict(sample_tag=tag))
        rawpath=remote/'raw_time_evidence'/rec['archived_file']
        assert digest(rawpath)==rec['sha256']
        row['hr_source']=rec['source']; row['hr_sha256']=rec['sha256']
        if 'error' in rec:
            row['reference_error']=rec['error']; continue
        row.update({k:rec[k] for k in ('n','start_s','end_s','span_s','min_step_s','max_step_s','nonpositive_steps','nonfinite_labels')})
        # Keep only timing fields from the Polar export header, not participant names or demographics.
        header,values=list(csv.reader(rec['first_three_lines'][:2]))
        hdr=dict(zip(header, values))
        row['hr_date']=hdr.get('Date'); row['hr_start_time']=hdr.get('Start time'); row['hr_duration']=hdr.get('Duration')
        if row.get('last_sensor_start') and row['hr_date'] and row['hr_start_time']:
            radar_start=datetime.strptime(row['last_sensor_start'], '%d-%b-%Y %H:%M:%S')
            hr_start=datetime.strptime(row['hr_date']+' '+row['hr_start_time'], '%d-%m-%Y %H:%M:%S')
            row['header_hr_minus_radar_start_s']=(hr_start-radar_start).total_seconds()
        # Compare archived model labels to raw CSV values, with historical and nominal time maps.
        pred=manifest.loc[manifest.sample_tag==tag]
        if len(pred):
            df=pd.read_csv(rawpath, skiprows=3, header=None)
            t=pd.to_timedelta(df.iloc[:,1]).dt.total_seconds().to_numpy(dtype=np.float32)
            y=pd.to_numeric(df.iloc[:,2]).to_numpy(dtype=np.float32)
            n=row.get('configured_frames', 1200); fs=row.get('configured_fs_hz', 20.)
            row['frame_count_basis']='raw_configuration' if 'configured_frames' in row else 'historical_loader_constant_unverified'
            row['frame_rate_basis']='raw_configuration' if 'configured_fs_hz' in row else 'historical_pipeline_constant_unverified'
            old_t=np.linspace(t.min(),t.max(),n,dtype=np.float32)
            old=np.interp(old_t,t,y,left=np.nan,right=np.nan).astype(np.float32)
            nominal=t[0]+np.arange(n)/fs
            same=np.interp(nominal,t,y,left=np.nan,right=np.nan)
            differences=[]; reproductions=[]; coverages=[]
            for window in pred.itertuples():
                s,e=int(window.window_start),int(window.window_end)
                reproductions.append(abs(float(np.nanmean(old[s:e]))-window.label_heart_rate))
                differences.append(abs(float(np.nanmean(same[s:e]))-window.label_heart_rate))
                coverages.append(float(np.isfinite(same[s:e]).mean()))
            row['saved_window_count']=len(pred)
            row['raw_label_reproduction_max_bpm']=max(reproductions)
            row['nominal_same_start_label_max_delta_bpm']=max(differences)
            row['nominal_same_start_min_window_coverage']=min(coverages)
            row['label_mapping_implied_fs_hz']=(n-1)/(t[-1]-t[0])
            row['time_stretch_end_error_s']=float((t[-1]-t[0])-(n-1)/fs)
    timing=pd.DataFrame(by_tag.values()).sort_values('sample_tag')
    timing.to_csv(out/'raw_timing_audit.csv',index=False)
    write_json(out/'timing_parse_failures.json',failures)
    validated=dict(elapsed_sec=time.perf_counter()-started,
        manifest_rows=len(manifest), mode_rows=len(modes), identical_channels_examples=actual_checks,
        record_count=len(timing), unmatched_sources=failures,
        raw_label_reproduction_max_bpm=float(timing.raw_label_reproduction_max_bpm.max()),
        raw_label_matched_records=int(timing.raw_label_reproduction_max_bpm.notna().sum()),
        raw_label_mismatch_records=int((timing.raw_label_reproduction_max_bpm>1e-4).sum()),
        model_records_without_raw_config=int((timing.saved_window_count.notna() & timing.configured_fs_hz.isna()).sum()),
        capture_logs_with_zero_filling=int((timing.zero_filled_packets>0).sum()),
        capture_logs_out_of_sequence=int((timing.out_of_sequence_packets>0).sum()),
        checked_raw_sidecars=len(logs)+len(refs),
        configured_fs_counts=timing.configured_fs_hz.value_counts().to_dict(),
        reference_span_counts=timing.span_s.value_counts().to_dict(),
        header_offset_counts=timing.header_hr_minus_radar_start_s.value_counts().to_dict(),
        synchronization_verified=False, retraining_allowed=False,
        frame_period_documentation='https://www.ti.com/content/dam/videos/external-videos/en-us/2/3816841626001/5427375477001.mp4/subassets/mmWave_Radar_Programming_Model.pdf',
        documentation_rule='framePeriodicity LSB = 5 ns; configured cadence is not measured per-frame timing',
        header_offset_caveat='Different clocks and integer-second headers: diagnostic offsets, not calibrated synchronization',
        source_sha256=digest(__file__))
    write_json(out/'validation.json',validated)
    print(json.dumps({k:v for k,v in validated.items() if k!='identical_channels_examples'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--remote',type=Path,default=Path('tmp/ftu_p0_result/ftu_p0'))
    p.add_argument('--output',type=Path,default=Path('experiment_runs/ftu_p0_audit_2026-10-14'))
    p.add_argument('--archive',type=Path,default=Path('experiment_runs/ftu_source_audit_2026-10-13'))
    a=p.parse_args();run(a.remote,a.output,a.archive)
