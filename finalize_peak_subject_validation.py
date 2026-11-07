"""Curate completed validation evidence without changing scores or inputs."""
from pathlib import Path
import hashlib
import json
import shutil

root = Path('experiment_runs/peak_subject_validation')
download = Path('tmp/peak_subject_validation_results')
data = download/'peak_subject_validation'
for name in ['acquisition.json', 'records.json', 'execution.json']:
    shutil.copyfile(data/name, root/name)
(root/'raw').mkdir(exist_ok=True)
shutil.copyfile(data/'windows.csv', root/'raw/windows.csv')
shutil.copyfile(download/'frozen-peak-subject-validation.log', root/'kaggle_v2_log.json')
(root/'signals').mkdir(exist_ok=True)
for pattern in ['*_phase20.npz', '*_spectra.npz', '*_ecg_beats.json']:
    for path in data.glob(pattern):
        shutil.copyfile(path, root/'signals'/path.name)
(root/'source').mkdir(exist_ok=True)
for path in (download/'frozen_validation_code').iterdir():
    if path.is_file():
        shutil.copyfile(path, root/'source'/path.name)
description = dict(kernel='https://www.kaggle.com/code/goldfish9901/frozen-peak-subject-validation',
    version=2, protocol_commit='b7eaec2', final_packaging_commit='577bca9',
    kaggle_code='source/',
    retained='All selected phase20 and spectra (8calibration-passing records),12lead-pairbeatdiagnostics,raw2808offsetwindowrows,12recordmetadata,CRC32/SHA256rawacquisitioninventory.',
    ecg_waveforms='All12rawscoringECGarchives remain inKaggleversion2outputs. Not duplicated inGit. Local fulloutput download was interrupted; only complete requiredphase/spectrum/beat/CSVfiles were used forverification. No newmanualECGadjudication.',
    independent_local_scorer='analyze_peak_subject_validation.py',
    source_archives='OfficialGUARDIANfigshareDOI10.6084/m9.figshare.12186516.v2; acquisition.json maps exact members and original hashes.')
(root/'provenance.json').write_text(json.dumps(description, indent=2), encoding='utf-8')
manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'}
(root/'artifact_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('Curated artifacts', len(manifest))
