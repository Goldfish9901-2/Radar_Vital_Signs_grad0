"""Freeze capture-independent within-subject splits without inspecting HR values."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from src.training.ftu_position_protocol import load_arrays

start=time.perf_counter()
out=Path('experiment_runs/ftu_record_learnability');out.mkdir(parents=True,exist_ok=True)
manifest,xt,xf,_=load_arrays('experiment_runs/ftu_timebase_edacm_v2/full_artifacts.zip')
timing=pd.read_csv('experiment_runs/ftu_p0_audit_2026-10-14/raw_timing_audit.csv')
timing=timing[timing.saved_window_count.notna()].set_index('sample_tag')
selected=manifest[manifest.split=='train']
rows=[]
for tag,g in selected.groupby('sample_tag',sort=True):
    idx=g.sort_values('window_start').index.to_numpy();r=timing.loc[tag]
    subject=g.group_key.iloc[0]
    capture=f'{subject}/{r.hr_date}/{r.hr_start_time}'
    feature_hash=hashlib.sha256(xt[idx].tobytes()+xf[idx].tobytes()).hexdigest()
    condition=tag.split('_',1)[1].rsplit('_r',1)[0]
    if condition in ['Distance_80_cm','Angle_0_deg__80_cm','Orientation_80_cm__Front_side']:
        condition='Front_80_cm'
    rows.append(dict(sample_tag=tag,group_key=subject,condition=condition,capture_key=capture,
        date=r.hr_date,hr_start_time=r.hr_start_time,radar_start=r.last_sensor_start,
        frame_count=r.configured_frames,feature_sha256=feature_hash,windows=len(idx)))
records=pd.DataFrame(rows)
assert len(records)==264
for key,g in records.groupby('capture_key'):
    assert g.feature_sha256.nunique()==1,('timestamp aliases have different inputs',key)
    assert g.condition.nunique()==1
# Also require exact feature duplicates to map to the same capture key.
assert records.groupby('feature_sha256').capture_key.nunique().max()==1
canonical=records.sort_values('sample_tag').groupby('capture_key',sort=True).first().reset_index()
assert len(canonical)==216
canonical['split']=''
for (subject,condition),g in canonical.groupby(['group_key','condition']):
    assert len(g)==4
    # Label-independent stable hash ranking; two train / one validation / one evaluation.
    ids=sorted(g.index,key=lambda i:hashlib.sha256(('20261017|'+canonical.loc[i,'capture_key']).encode()).hexdigest())
    canonical.loc[ids,'split']=['train','train','val','evaluation']
assert canonical.groupby('group_key').split.nunique().eq(3).all()
records=records.merge(canonical[['capture_key','sample_tag','split']].rename(columns={'sample_tag':'canonical_tag'}),on='capture_key',validate='many_to_one')
records['kept']=records.sample_tag==records.canonical_tag
records.to_csv(out/'capture_audit.csv',index=False)
canonical.to_csv(out/'record_split.csv',index=False)
gaps=[]
for subject,g in canonical.groupby('group_key'):
    starts=pd.to_datetime(g.date+' '+g.hr_start_time,format='%d-%m-%Y %H:%M:%S').sort_values()
    gaps.append(dict(subject=subject,unique_dates=int(g.date.nunique()),min_start_gap_s=float(starts.diff().dt.total_seconds().min()),
                     unique_captures=len(g),missing_radar_start=int(g.radar_start.isna().sum())))
protocol=dict(model='MLP 129->64->32->1; GELU; dropout .15',seeds=[42,43,44],shuffle_seeds=[101,202,303],
    epochs=60,patience=12,batch_size=64,lr=.0003,weight_decay=.0001,optimizer='AdamW',
    loss='SmoothL1 beta=.5',clip_grad_norm=1.,scheduler='ReduceLROnPlateau factor=.5 patience=4 on validation centered RMSE',
    feature_processing='unchanged exported x_freq; per-bin StandardScaler fitted on new training captures only',
    selection='minimum validation record-centered RMSE using correct labels in all arms',
    shuffle='fixed per-record permutation of 8 training labels; same label multiset; validation/evaluation always correct',
    capture_group='same subject/date/reference start plus exact time+frequency feature equality; canonical lexicographic alias only',
    split='stable SHA256 rank 20261017|capture_key within subject/condition; 2 train, 1 val, 1 evaluation',
    subjects=[2,3,4,5,7,9],exclusions='original validation/test subjects never enter this experiment',
    scope='same-subject same-visit cross-triggered-capture diagnostic; NOT cross-visit/session generalization',
    continuous_capture_evidence='Separate finite 1200-frame sensor starts where logs exist; no overlapping unique start intervals; p03 partial missing radar logs noted',
    synthetic=dict(train_records=96,val_records=24,evaluation_records=24,windows_per_record=8,
        train_grid_hz='0.75 + .05*k, k=0..35',val_grid_hz='0.775 + .05*k, k=0..34',
        evaluation_grid_hz='0.7625 or 0.7875 + .05*k, k=0..34',
        phase='uniform 0..2pi',amplitude='uniform .5..2',noise='Gaussian std uniform 0...3 times amplitude',
        feature='same zscore_1d -> frequency_features pipeline',
        gate='all three seeds: validation AND evaluation centered RMSE <=4.6875 bpm AND centered r >=.9',
        groups='independent synthetic groups of eight signals, not physiological continuous recordings'),
    inference='seeds are optimization replicates, not independent subjects; report subject-level paired differences; no window-level significance claims',
    interpretation=['synthetic failure: stop before real-data training',
        'synthetic pass with real train-only fit: investigate frontend/labels/cross-record stability',
        'consistent held-out-capture advantage over shuffles: then investigate cross-subject transfer',
        'true and shuffled similar: no reproducible dynamic signal evidence in this representation'])
(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
summary=dict(named_records=264,unique_captures=216,duplicate_aliases_removed=48,
    windows=1728,split_records=canonical.split.value_counts().to_dict(),subjects=gaps,
    elapsed_sec=time.perf_counter()-start,uses_labels_for_split=False)
(out/'split_validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
