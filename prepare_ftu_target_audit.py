"""Label-blind selection of one training capture per subject/condition."""
import hashlib,json,time
from pathlib import Path
import pandas as pd
start=time.perf_counter();out=Path('experiment_runs/ftu_target_audit');out.mkdir(exist_ok=True)
s=pd.read_csv('experiment_runs/ftu_record_learnability/record_split.csv')
s=s[s.split=='train'].copy();assert len(s)==108
s['selection_hash']=s.capture_key.map(lambda k:hashlib.sha256(('ftu-target-audit-20261018|'+k).encode()).hexdigest())
selected=s.sort_values('selection_hash').groupby(['group_key','condition'],sort=True).head(1).sort_values(['group_key','condition'])
assert len(selected)==54 and selected.groupby('group_key').size().eq(9).all()
selected.to_csv(out/'selected_records.csv',index=False)
protocol=dict(selection='SHA256(ftu-target-audit-20261018|capture_key) minimum within each original-training subject x canonical condition; no HR/error use',
    records=54,fs_hz=20,window=256,stride=128,anchor=[0,256],
    A='original window-local top3 selector and original weights',
    B='top3 targets and weights selected once on frames[0:256], then frozen; same window EDACM/detrend/zscore/spectrum',
    evaluation='only original starts>=256; both arms same windows and labels; >=80percent finite reference coverage; no overlap with anchor',
    time_mapping='radar_time=frame_index/20 +0s on original reference time axis; assumed_unverified; no offset search in this experiment',
    target_selection='unchanged original rules; internal scoring HR band .75-2.5Hz; no reference HR used',
    readout='maximum spectrum bin in .5-2.75Hz for fused and all3 ranked singles; no label-selected peak or best single',
    overlap='adjacent windows; compare aligned increments excluding artificial first delta; phase segments demeaned to remove arbitrary origin; no HR-based alignment',
    overlap_spectrum='same127 samples of each window processed phase, demean only then Hann/rFFT normalized power total-variation distance; preserves window detrend effects',
    fusion='retain complex/amplitude/raw EDACM delta/cumulative/detrended per-target phase; weighted phase sum before and after zscore; band power ratio coherent sum over sum individual weighted powers',
    temporal_diagnostics='set turnover, target identity weight L1 change in union coordinates; overlap raw-delta and detrended-delta nRMSE; post-window spectrum disagreement',
    outcomes='raw MAE and record-centered RMSE/R2/r/std ratio; pooled and six subject equal weight; all plots, no error-ranked examples',
    boundaries='selected indices are positions in exported RDA, not calibrated native range/angle; no ADC rerun; no learned models; same-visit original-training-only exploration',
    decision=['B consistency AND correct-label correspondence improve across subjects: basis for target-selection revision, then one fixed re-export/retrain',
              'B only stability improves: inspect phase extraction/fusion/reference time axis',
              'neither has stable structure: stop network search and audit original complex data and documentation'])
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
(out/'selection_validation.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-start,records=54,per_subject=9,
    training_only=True,selection_uses_labels=False,source_split_sha256=hashlib.sha256(Path('experiment_runs/ftu_record_learnability/record_split.csv').read_bytes()).hexdigest()),indent=2))
print(selected[['group_key','condition','sample_tag']].to_string(index=False))
