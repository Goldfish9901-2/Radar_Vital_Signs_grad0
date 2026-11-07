"""Prospective six-subject paired validation, frozen before signal access."""
from pathlib import Path
import hashlib
import json
import struct
import time
import zipfile
import zlib

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
import neurokit2 as nk

import run_synchronized_real_pilot as base
from synchronized_ellipse_fixed import ellipse_phase
from run_full_local_peak_readout import select

CODE = Path(__file__).parent
PROTO = json.loads((CODE/'validation_protocol.json').read_text())
OUT = Path('/kaggle/working/peak_subject_validation')
RAW = Path('/kaggle/temp/peak_subject_validation_raw')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def acquire():
    inventory = []
    for archive in PROTO['archives']:
        subjects = [s for s in PROTO['subjects'] if archive['first'] <= int(s[3:]) <= archive['last']]
        if not subjects:
            continue
        with zipfile.ZipFile(base.ArchiveDirectory(archive['url'], archive['size'])) as directory:
            infos = {x.filename: x for x in directory.infolist()}
        for subject in subjects:
            for number, scenario in [(1, 'Resting'), (2, 'Valsalva')]:
                name = f'{subject}/{subject}_{number}_{scenario}.mat'
                entry = dict(subject=subject, scenario=scenario, name=name, archive=archive['url'])
                try:
                    info = infos[name]
                    assert info.compress_type == zipfile.ZIP_STORED
                    header = base.get_range(archive['url'], info.header_offset, 30)
                    assert header[:4] == b'PK\x03\x04'
                    name_len, extra_len = struct.unpack_from('<HH', header, 26)
                    path = RAW/Path(name).name
                    started = time.perf_counter()
                    base.get_range(archive['url'], info.header_offset+30+name_len+extra_len, info.compress_size, path)
                    content = path.read_bytes()
                    assert len(content) == info.file_size and zlib.crc32(content) == info.CRC
                    entry.update(ok=True, size=info.file_size, crc32=info.CRC,
                        sha256=hashlib.sha256(content).hexdigest(), elapsed_seconds=time.perf_counter()-started)
                except Exception as exc:
                    entry.update(ok=False, error=repr(exc))
                inventory.append(entry)
                print('ACQUIRE', entry, flush=True)
                write_json(OUT/'acquisition.json', inventory)
    return inventory


def process(path):
    started = time.perf_counter()
    data = loadmat(path, squeeze_me=True)
    subject = path.stem.split('_')[0]
    scenario = 'Resting' if 'Resting' in path.stem else 'Valsalva'
    fs, ef = base.scalar(data, 'fs_radar'), base.scalar(data, 'fs_ecg')
    start, end, pulses = (30., 210., []) if scenario == 'Resting' else base.recovery_interval(data)
    starts = np.arange(start, end-25.6+1e-6, 2.)
    assert len(starts) == 78
    meta = dict(subject=subject, scenario=scenario, record=path.stem, interval_valid=True,
                score_start_s=start, score_end_s=end, pulses=pulses, fs_radar=fs, fs_ecg=ef)
    # Reference processing and deterministic lead selection precede radar processing.
    leads = [np.asarray(data[f'tfm_ecg{k}']).reshape(-1) for k in [1, 2]]
    detected = []
    reference = np.full((2, len(starts), 3), np.nan)
    for index, values in enumerate(leads):
        try:
            beats, runs = base.ecg_beats(values, ef)
            detected.append(dict(lead=index+1, beats_s=beats.tolist(), valid_runs_s=runs))
            for j, s in enumerate(starts):
                for k, offset in enumerate(PROTO['offsets_s']):
                    reference[index, j, k] = base.window_hr(beats, s+offset, s+25.6+offset, runs)
        except Exception as exc:
            detected.append(dict(lead=index+1, error=repr(exc), beats_s=[], valid_runs_s=[]))
    lead_valid = np.isfinite(reference).all(axis=2)
    primary = PROTO['ecg_primary_lead'][subject]-1
    selected = primary
    coverage = lead_valid.mean(axis=1)
    switched = bool(coverage[primary] < .8 and coverage[1-primary] >= .8)
    if switched:
        selected = 1-primary
    both_finite = np.isfinite(reference).all(axis=0)
    disagreement = abs(reference[0]-reference[1])
    concordant = (~both_finite | (disagreement <= 5)).all(axis=1)
    reference_valid = lead_valid[selected] & concordant
    meta.update(primary_lead=primary+1, selected_lead=selected+1, backup_switched=switched,
        lead_common_offset_coverage=coverage.tolist(), reference_valid_windows=int(reference_valid.sum()),
        reference_disagreement_windows=int((~concordant).sum()), reference_any_valid=bool(reference_valid.any()),
        reference_80pct_valid=bool(reference_valid.mean() >= .8))
    write_json(OUT/f'{path.stem}_ecg_beats.json', detected)
    a, b = round((start-2)*ef), round((end+2)*ef)
    np.savez_compressed(OUT/f'{path.stem}_ecg.npz', ecg1=leads[0][a:b], ecg2=leads[1][a:b], fs=ef, start_s=a/ef)
    phase20 = None
    low20 = None
    try:
        i, q = [np.asarray(data[key], dtype=float).reshape(-1) for key in ['radar_i', 'radar_q']]
        assert len(i) == len(q) and len(i)/fs >= end+1
        phase, low, calibration = ellipse_phase(i, q, fs)
        ratio = round(fs/20)
        assert abs(fs/ratio-20) < 1e-6
        phase20 = signal.resample_poly(phase, 1, ratio)
        low20 = signal.resample_poly(low.astype(float), 1, ratio) > .01
        meta.update(calibration_passed=True, calibration=calibration)
        np.savez_compressed(OUT/f'{path.stem}_phase20.npz', phase=phase20, low_amplitude=low20, fs=20.)
    except Exception as exc:
        meta.update(calibration_passed=False, calibration_error=repr(exc))
    rows, spectra = [], []
    bpm = np.fft.rfftfreq(4096, 1/20)*60
    band = (bpm >= 45) & (bpm <= 150)
    for j, s in enumerate(starts):
        original, candidate = np.nan, np.nan
        low_fraction, radar_valid, count = np.nan, False, 0
        if phase20 is not None:
            first = round(s*20)
            trace = phase20[first:first+512]
            low_fraction = float(low20[first:first+512].mean())
            radar_valid = len(trace) == 512 and low_fraction <= .01 and np.isfinite(trace).all()
            power = abs(np.fft.rfft(signal.detrend(trace)*signal.windows.hann(512, sym=False), n=4096))**2
            original = float(bpm[np.flatnonzero(band)[np.argmax(power[band])]])
            candidate, diag = select(power, bpm)
            count = diag['candidate_count']
            spectra.append(power)
        for k, offset in enumerate(PROTO['offsets_s']):
            rows.append(dict(subject=subject, scenario=scenario, record=path.stem, start_s=s,
                end_s=s+25.6, offset_s=offset, original=original, candidate=candidate,
                reference=reference[selected, j, k], lead1_reference=reference[0, j, k],
                lead2_reference=reference[1, j, k], primary_lead=primary+1, selected_lead=selected+1,
                reference_valid=bool(reference_valid[j]), radar_valid=bool(radar_valid),
                low_amplitude_fraction=low_fraction, candidate_count=count,
                common_valid=bool(reference_valid[j] and radar_valid and np.isfinite(original) and np.isfinite(candidate))))
    if spectra:
        np.savez_compressed(OUT/f'{path.stem}_spectra.npz', power=np.array(spectra), bpm=bpm, starts_s=starts)
    meta['elapsed_seconds'] = time.perf_counter()-started
    return rows, meta


def main():
    started = time.perf_counter()
    assert nk.__version__ == '0.2.12'
    OUT.mkdir(exist_ok=False)
    RAW.mkdir(parents=True, exist_ok=True)
    inventory = acquire()
    all_rows, records = [], []
    for item in inventory:
        try:
            if not item['ok']:
                raise ValueError('Acquisition failed: '+item['error'])
            rows, meta = process(RAW/Path(item['name']).name)
            all_rows.extend(rows)
        except Exception as exc:
            meta = dict(subject=item['subject'], scenario=item['scenario'], record=Path(item['name']).stem,
                interval_valid=False, calibration_passed=False, reference_any_valid=False,
                reference_80pct_valid=False, reference_valid_windows=0, error=repr(exc))
        records.append(meta)
        write_json(OUT/'records.json', records)
        print('RECORD', meta, flush=True)
    pd.DataFrame(all_rows).to_csv(OUT/'windows.csv', index=False)
    write_json(OUT/'execution.json', dict(elapsed_seconds=time.perf_counter()-started,
        subjects=PROTO['subjects'], intended_records=12, intended_windows=936, actual_rows=len(all_rows),
        neurokit2=nk.__version__, training=False, remaining21_signals_accessed=False))


if __name__ == '__main__':
    main()
