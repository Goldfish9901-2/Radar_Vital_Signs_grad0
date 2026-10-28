"""One frozen CW phase/FFT pilot; official synchronized raw ECG, no learning."""
from pathlib import Path
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
from scipy import linalg, signal
from scipy.io import loadmat
import neurokit2 as nk

OUT = Path('/kaggle/working/sync_real_pilot')
RAW = Path('/kaggle/temp/sync_real_raw')
PROTO = json.loads(Path(__file__).with_name('protocol.json').read_text())


def get_range(url, start, count, destination=None):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{start+count-1}'})
    with urllib.request.urlopen(req, timeout=90) as response:
        expected = f'bytes {start}-{start+count-1}/'
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith(expected):
            raise ValueError('Server did not honor exact byte range')
        if destination is None:
            data = response.read(count+1)
            if len(data) != count: raise ValueError('Range size mismatch')
            return data
        with destination.open('wb') as target:
            remaining = count
            while remaining:
                chunk = response.read(min(4*1024**2, remaining))
                if not chunk: raise ValueError('Truncated range download')
                target.write(chunk)
                remaining -= len(chunk)


class ArchiveDirectory(io.RawIOBase):
    def __init__(self, url, size): self.url, self.size, self.pos = url, size, 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset+(0 if whence == 0 else self.pos if whence == 1 else self.size)
        return self.pos
    def read(self, n=-1):
        n = self.size-self.pos if n < 0 else min(n, self.size-self.pos)
        if not n: return b''
        if n > 5_000_000: raise ValueError('ZIP directory budget exceeded')
        data = get_range(self.url, self.pos, n)
        self.pos += n
        return data


def acquire():
    archive = PROTO['archive']
    remote = ArchiveDirectory(archive['url'], archive['size'])
    with zipfile.ZipFile(remote) as directory:
        infos = {x.filename: x for x in directory.infolist()}
    inventory = []
    for item in PROTO['members']:
        name = item['name']
        info = infos[name]
        if info.file_size != item['size'] or info.CRC != item['crc32']:
            raise ValueError('Frozen member metadata mismatch')
        if info.compress_type != zipfile.ZIP_STORED: raise ValueError('Unexpected outer ZIP compression')
        header = get_range(archive['url'], info.header_offset, 30)
        if header[:4] != b'PK\x03\x04': raise ValueError('Invalid local ZIP header')
        name_len, extra_len = struct.unpack_from('<HH', header, 26)
        start = info.header_offset+30+name_len+extra_len
        path = RAW/Path(name).name
        began = time.perf_counter()
        get_range(archive['url'], start, info.compress_size, path)
        data = path.read_bytes()
        if zlib.crc32(data) != info.CRC: raise ValueError('ZIP member CRC mismatch')
        inventory.append(dict(item, sha256=hashlib.sha256(data).hexdigest(), download_seconds=time.perf_counter()-began))
        print('ACQUIRED', name, len(data), flush=True)
    (OUT/'acquisition.json').write_text(json.dumps(inventory, indent=2))


def scalar(data, key): return float(np.asarray(data[key]).squeeze())


def ellipse_phase(i, q, fs):
    """Direct least-squares ellipse on first 30 s only; no label or fallback."""
    xy = np.column_stack([i, q])
    calibration = xy[:round(30*fs):max(1, round(fs/100))]
    origin, scale = calibration.mean(0), calibration.std(0)
    if np.any(scale < 1e-12): raise ValueError('Degenerate I/Q calibration')
    norm = (calibration-origin)/scale
    x, y = norm.T
    design = np.column_stack([x*x, x*y, y*y, x, y, np.ones(len(x))])
    constraint = np.zeros((6, 6)); constraint[0, 2] = constraint[2, 0] = 2; constraint[1, 1] = -1
    _, vectors = linalg.eig(design.T@design, constraint)
    candidates = []
    for v in vectors.T:
        if np.max(np.abs(v.imag)) > 1e-8: continue
        v = v.real
        a, b, c, d, e, f = v
        if 4*a*c-b*b <= 0: continue
        quad = np.array([[a, b/2], [b/2, c]])
        if np.trace(quad) < 0: quad, v = -quad, -v
        center = -.5*np.linalg.solve(quad, v[3:5])
        level = center@quad@center-v[5]
        eig, basis = np.linalg.eigh(quad/level)
        if level <= 0 or np.any(eig <= 0): continue
        transform = (basis*np.sqrt(eig))@basis.T
        corrected = (norm-center)@transform
        residual = float(np.sqrt(np.mean((np.linalg.norm(corrected, axis=1)-1)**2)))
        candidates.append((residual, center, transform, eig))
    if not candidates: raise ValueError('No valid calibration ellipse')
    residual, center, transform, eig = min(candidates, key=lambda z: z[0])
    if residual > .2 or np.sqrt(eig.max()/eig.min()) > 100:
        raise ValueError('Frozen ellipse reliability criterion failed')
    corrected = ((xy-origin)/scale-center)@transform
    z = corrected[:, 0]+1j*corrected[:, 1]
    increments = np.angle(z[1:]*z[:-1].conj())
    phase = np.r_[0., np.cumsum(increments)]
    low = np.abs(z) < .1
    return phase, low, {'origin': origin.tolist(), 'scale': scale.tolist(), 'ellipse_center': center.tolist(),
                        'ellipse_transform': transform.tolist(), 'calibration_radial_rmse': residual,
                        'low_amplitude_fraction': float(low.mean()), 'phase_increment_q999_rad': float(np.quantile(np.abs(increments), .999))}


def ecg_beats(values, fs):
    """Pinned detector; process finite contiguous runs, never fill missing ECG."""
    ratio = round(fs/250)
    if abs(fs/ratio-250) > 1e-6: raise ValueError('ECG sampling rate not integer multiple of 250 Hz')
    finite = np.isfinite(values)
    edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    beats, valid_runs = [], []
    for first, last in edges.reshape(-1, 2):
        if last-first < 10*fs: continue
        reduced = signal.resample_poly(values[first:last], 1, ratio)
        cleaned = nk.ecg_clean(reduced, sampling_rate=250, method='neurokit')
        _, peaks = nk.ecg_peaks(cleaned, sampling_rate=250, method='neurokit', correct_artifacts=False)
        times = first/fs+np.asarray(peaks['ECG_R_Peaks'])/250
        times = times[(times >= first/fs+2)&(times <= last/fs-2)]
        beats.extend(times.tolist()); valid_runs.append([first/fs+2, last/fs-2])
    return np.asarray(beats), valid_runs


def window_hr(beats, start, end, valid_runs):
    if not any(a <= start and b >= end for a, b in valid_runs): return np.nan
    left, right = beats[:-1], beats[1:]
    overlap = np.maximum(0, np.minimum(right, end)-np.maximum(left, start))
    use = overlap > 0
    rr = right-left
    if not use.any() or abs(overlap.sum()-(end-start)) > 1e-6: return np.nan
    if np.any((rr[use] < .3)|(rr[use] > 2)): return np.nan
    return float(np.sum(overlap[use]*60/rr[use])/(end-start))


def recovery_interval(data):
    values = np.asarray(data['tfm_intervention']).reshape(-1)
    fs = scalar(data, 'fs_intervention')
    if not np.isfinite(values).all() or values.min() > 2: raise ValueError('Missing intervention support')
    down = values < (values.min()+values.max())/2
    edges = np.flatnonzero(np.diff(np.r_[False, down, False]))
    pulses = [(a+b)/(2*fs) for a, b in edges.reshape(-1, 2) if b-a >= 3]
    if len(pulses) != 6: raise ValueError(f'Expected 6 intervention pulses, got {len(pulses)}; no posthoc replacement')
    start, end = pulses[1]+30, pulses[1]+210
    if end > pulses[2]: raise ValueError('Recovery would overlap next maneuver')
    return start, end, pulses


def process(path):
    began = time.perf_counter()
    data = loadmat(path, squeeze_me=True)
    pid = path.name.split('_')[0]
    scenario = 'Resting' if 'Resting' in path.name else 'Valsalva'
    fs = scalar(data, 'fs_radar'); ef = scalar(data, 'fs_ecg')
    i, q = (np.asarray(data[key], dtype=float).reshape(-1) for key in ['radar_i', 'radar_q'])
    if len(i) != len(q) or not np.isfinite(i).all() or not np.isfinite(q).all(): raise ValueError('Invalid I/Q arrays')
    phase, low, meta = ellipse_phase(i, q, fs)
    ratio = round(fs/20)
    if abs(fs/ratio-20) > 1e-6: raise ValueError('Radar rate not integer multiple of 20 Hz')
    phase20 = signal.resample_poly(phase, 1, ratio)
    low20 = signal.resample_poly(low.astype(float), 1, ratio) > .01
    leads = [np.asarray(data['tfm_ecg1']).reshape(-1), np.asarray(data['tfm_ecg2']).reshape(-1)]
    detected = [ecg_beats(x, ef) for x in leads]
    lead = PROTO['ecg_primary_lead'][pid]-1
    start, end, pulses = (30., 210., []) if scenario == 'Resting' else recovery_interval(data)
    if min(len(i)/fs, len(leads[0])/ef, len(leads[1])/ef) < end+1:
        raise ValueError('Selected continuous scoring interval lacks support')
    rows = []
    for s in np.arange(start, end-25.6+1e-6, 2.):
        first = round(s*20); trace = phase20[first:first+512]
        valid_radar = len(trace) == 512 and np.mean(low20[first:first+512]) <= .01
        spectrum = np.abs(np.fft.rfft(signal.detrend(trace)*signal.windows.hann(512, sym=False), n=4096))**2
        freq = np.fft.rfftfreq(4096, 1/20)*60
        band = (freq >= 45)&(freq <= 150)
        prediction = float(freq[np.flatnonzero(band)[np.argmax(spectrum[band])]])
        for offset in [0., -.5, .5]:
            refs = [window_hr(b, s+offset, s+25.6+offset, runs) for b, runs in detected]
            rows.append({'subject': pid, 'scenario': scenario, 'start_s': s, 'end_s': s+25.6,
                         'center_s': s+12.8, 'reference_offset_s': offset, 'prediction': prediction,
                         'reference': refs[lead], 'secondary_lead_reference': refs[1-lead],
                         'radar_valid': valid_radar, 'low_amplitude_fraction': float(low20[first:first+512].mean())})
    for index, (beats, runs) in enumerate(detected):
        pd.DataFrame({'beat_time_s': beats}).to_csv(OUT/f'{path.stem}_lead{index+1}_beats.csv', index=False)
    ecg_first, ecg_last = round((start-2)*ef), round((end+2)*ef)
    np.savez_compressed(OUT/f'{path.stem}_reference_ecg.npz',
                        ecg1=leads[0][ecg_first:ecg_last], ecg2=leads[1][ecg_first:ecg_last],
                        fs=ef, start_s=ecg_first/ef)
    np.savez_compressed(OUT/f'{path.stem}_phase20.npz', phase=phase20, low_amplitude=low20, fs=20.)
    # Keep raw synchronized I/Q and ECG files in Kaggle temporary storage, not GitHub.
    meta.update(subject=pid, scenario=scenario, fs_radar=fs, fs_ecg=ef, score_start_s=start, score_end_s=end,
                intervention_pulses_s=pulses, clock_error_bound_s=None,
                synchronization='Published Gold-code resampling and lag correction; no retained sync sequence for independent residual bound',
                ecg_reference='NeuroKit2 0.2.12 uncorrected R peaks; not manually adjudicated',
                elapsed_seconds=time.perf_counter()-began)
    return rows, meta


def metrics(rows):
    results = []
    for (pid, scenario, offset), group in rows.groupby(['subject', 'scenario', 'reference_offset_s']):
        valid = group.radar_valid & np.isfinite(group.reference)
        frame = group.loc[valid].sort_values('start_s')
        entry = {'subject': pid, 'scenario': scenario, 'reference_offset_s': offset,
                 'windows_total': len(group), 'windows_scored': len(frame)}
        if len(frame) >= 3:
            t, y, p = frame.center_s.to_numpy(), frame.reference.to_numpy(), frame.prediction.to_numpy()
            design = np.column_stack([np.ones(len(t)), t-t.mean()])
            trend = design@np.linalg.lstsq(design, y, rcond=None)[0]
            yr = y-trend
            pr = p-design@np.linalg.lstsq(design, p, rcond=None)[0]
            adjacent = np.isclose(np.diff(t), 2.)
            mean, median = np.full(len(y), y.mean()), np.full(len(y), np.median(y))
            secondary = frame.secondary_lead_reference.to_numpy()
            entry.update(mae=float(np.mean(np.abs(p-y))), mean_oracle_mae=float(np.mean(np.abs(mean-y))),
                         median_oracle_mae=float(np.mean(np.abs(median-y))), trend_oracle_mae=float(np.mean(np.abs(trend-y))),
                         change_rmse=float(np.sqrt(np.mean((np.diff(p)[adjacent]-np.diff(y)[adjacent])**2))) if adjacent.any() else None,
                         constant_change_rmse=float(np.sqrt(np.mean(np.diff(y)[adjacent]**2))) if adjacent.any() else None,
                         trend_change_rmse=float(np.sqrt(np.mean((np.diff(trend)[adjacent]-np.diff(y)[adjacent])**2))) if adjacent.any() else None,
                         detrended_rmse=float(np.sqrt(np.mean((pr-yr)**2))), trend_detrended_rmse=float(np.sqrt(np.mean(yr**2))),
                         detrended_r2=float(1-np.sum((pr-yr)**2)/np.sum(yr**2)) if np.sum(yr**2)>1e-12 else None,
                         detrended_r=float(np.corrcoef(pr, yr)[0, 1]) if min(np.std(pr), np.std(yr))>1e-12 else None,
                         reference_std=float(np.std(y)), reference_residual_std=float(np.std(yr)),
                         prediction_std=float(np.std(p)),
                         ecg_lead_disagreement_rmse=float(np.sqrt(np.nanmean((y-secondary)**2))) if np.isfinite(secondary).any() else None)
        results.append(entry)
    return results


def main():
    began = time.perf_counter(); OUT.mkdir(exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    acquire()
    all_rows, metadata, errors = [], [], []
    for item in PROTO['members']:
        path = RAW/Path(item['name']).name
        try:
            rows, meta = process(path); all_rows.extend(rows); metadata.append(meta)
            print('PROCESSED', path.name, len(rows), flush=True)
        except Exception as exc:
            errors.append({'record': path.name, 'error': repr(exc)})
            print('RECORD ERROR', path.name, repr(exc), flush=True)
    frame = pd.DataFrame(all_rows)
    if len(frame):
        frame['common_support'] = frame.groupby(['subject', 'scenario', 'start_s']).reference.transform(lambda v: np.isfinite(v).all())
        frame['radar_valid'] &= frame.common_support
    frame.to_csv(OUT/'window_predictions.csv', index=False)
    if len(frame): pd.DataFrame(metrics(frame)).to_csv(OUT/'record_metrics.csv', index=False)
    (OUT/'records.json').write_text(json.dumps(metadata, indent=2))
    (OUT/'errors.json').write_text(json.dumps(errors, indent=2))
    (OUT/'execution.json').write_text(json.dumps({'elapsed_seconds': time.perf_counter()-began,
        'training': False, 'new_ftu_synthetic_calibration': False, 'records_completed': len(metadata),
        'records_failed': len(errors), 'clock_error_bound_measured': False, 'reference_manually_adjudicated': False}, indent=2))


if __name__ == '__main__': main()
