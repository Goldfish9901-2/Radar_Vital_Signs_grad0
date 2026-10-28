"""Record the completed 24-page technical visual review, without radar inputs."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd

root = Path('experiment_runs/synchronized_real_pilot_fixed')
raw = Path('tmp/sync_real_results/sync_real_pilot')
packet = json.loads((root/'ecg_only_packet_manifest.json').read_text())
reviewed = root/'reviewed_beats'; reviewed.mkdir(exist_ok=True)
pages = root/'review_pages'; pages.mkdir(exist_ok=True)
records = []
for stem in dict.fromkeys(p['record'] for p in packet['pages']):
    sheets = [p for p in packet['pages'] if p['record'] == stem]
    assert len(sheets) == 4
    first, last = min(p['start_s'] for p in sheets), max(p['end_s'] for p in sheets)
    lead = sheets[0]['primary_lead']
    source = raw/f'{stem}_lead{lead}_beats.csv'
    frame = pd.read_csv(source)
    selected = frame[(frame.beat_time_s >= first)&(frame.beat_time_s < last)]
    selected.to_csv(reviewed/f'{stem}.csv', index=False)
    for page in sheets:
        path = Path(page['file'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == page['sha256']
        shutil.copyfile(path, pages/path.name)
    record = {'record': stem, 'primary_lead': lead, 'review_start_s': first, 'review_end_s': last,
              'all_four_pages_visually_inspected': True, 'technical_quality': 'QRS visible; no clear missed or extra automatic detections found',
              'added_peaks_s': [], 'deleted_peaks_s': [], 'uncertain_intervals_s': [],
              'reviewed_peak_count': len(selected), 'original_auto_peaks_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'raw_primary_container_sha256': hashlib.sha256((raw/f'{stem}_reference_ecg.npz').read_bytes()).hexdigest()}
    if stem == 'GDN0002_1_Resting':
        record['notes'] = 'Baseline drift and transients around188-208s; sharp QRS remain identifiable and coincide with auto markers. No smoothing, beat removal, or exclusion.'
    elif stem.startswith('GDN0007'):
        record['notes'] = 'Primary ECG2 has distinguishable narrow QRS and later broad T waves; markers follow QRS in reviewed strips. No replacement by ECG1.'
    else:
        record['notes'] = 'Primary QRS morphology and marker correspondence remain clear across the full saved interval, including boundaries.'
    records.append(record)
decision = {'reviewer': 'Codex technical visual waveform review; not independent expert clinical adjudication',
            'review_method': 'All24 ECG-only pages inspected with image tools,184s per record; no detector rerun or parameter tuning',
            'corrected_radar_predictions_opened_before_review': False,
            'previous_defective_run_predictions_previously_seen': True,
            'meaning': 'Accept original main-lead automatic peak locations for this limited window-HR analysis; not a certified beat-time ground truth or ECG diagnosis',
            'records': records, 'total_seconds_reviewed': sum(r['review_end_s']-r['review_start_s'] for r in records),
            'total_peak_edits': 0, 'other27_subjects_accessed': False}
(root/'ecg_review_decisions.json').write_text(json.dumps(decision, indent=2))
pd.DataFrame([{k: r[k] for k in ['record','primary_lead','review_start_s','review_end_s','reviewed_peak_count','technical_quality','notes']} for r in records]).to_csv(root/'ecg_review_summary.csv', index=False)
print('Frozen visual review:', len(records), 'records;', decision['total_seconds_reviewed'], 'seconds; no peak edits')
