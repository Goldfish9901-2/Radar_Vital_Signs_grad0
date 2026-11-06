"""Stage1 radar-only catalogue. No ECG reference columns are decoded."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy import signal

from run_ellipse_readout_controls import original_reader


ROOT = Path('experiment_runs/real_band_truncation')
INPUT = Path('experiment_runs/synchronized_real_pilot_fixed')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    began = time.perf_counter()
    config = read(ROOT/'protocol.json')
    for name, expected in read(ROOT/'source_hashes.json').items():
        assert digest(Path(name)) == expected, name
    radar = ROOT/'radar'
    assert not radar.exists(), 'Preserve the completed radar freeze'
    radar.mkdir()
    windows = pd.read_csv(INPUT/'window_predictions.csv', usecols=config['radar_only_columns'])
    windows = windows[windows.reference_offset_s == 0].copy()
    reader, source = original_reader()
    (radar/'extracted_reader.py.txt').write_text(source, encoding='utf-8')
    all_windows = []
    all_peaks = []
    gain = signal.windows.hann(512, sym=False).sum()
    for record in config['records']:
        pid, number, scenario = record.split('_')
        table = windows[(windows.subject == pid) & (windows.scenario == scenario)].sort_values('start_s')
        assert len(table) == config['windows_per_record']
        phase_file = INPUT/'signals'/f'{record}_phase20.npz'
        data = np.load(phase_file)
        assert float(data['fs']) == 20
        powers = []
        for row in table.itertuples(index=False):
            first = round(row.start_s*20)
            trace = data['phase'][first:first+512]
            power, bpm, band, prediction = reader(trace)
            assert prediction == row.prediction
            powers.append(power)
            amplitude = 2*np.sqrt(power)/gain
            indices, properties = signal.find_peaks(power, prominence=(None, None),
                                                    width=(None, None), plateau_size=(None, None))
            genuine = set(indices.tolist())
            info = {int(index): dict(full_local=True, kind='full_local',
                                    prominence_power=float(properties['prominences'][k]),
                                    width_bins=float(properties['widths'][k]),
                                    plateau_bins=int(properties['plateau_sizes'][k]))
                    for k, index in enumerate(indices)}
            for index, neighbor in [(0, 1), (len(power)-1, len(power)-2)]:
                if power[index] >= power[neighbor]:
                    info[index] = dict(full_local=False, kind='full_endpoint',
                                       prominence_power=None, width_bins=None, plateau_bins=None)
            band_indices = np.flatnonzero(band)
            left, right = int(band_indices[0]), int(band_indices[-1])
            for index, neighbor in [(left, left+1), (right, right-1)]:
                if power[index] >= power[neighbor] and index not in info:
                    info[index] = dict(full_local=False, kind='crop_endpoint_only',
                                       prominence_power=None, width_bins=None, plateau_bins=None)
            full_peaks = sorted([i for i, p in info.items() if p['kind'] != 'crop_endpoint_only'], key=lambda i: (-power[i], i))
            band_genuine = sorted([i for i in genuine if band[i]], key=lambda i: (-power[i], i))
            crop_peaks = sorted([i for i in info if band[i]], key=lambda i: (-power[i], i))
            winner = int(band_indices[np.argmax(power[band])])
            assert crop_peaks[0] == winner
            full_ranks = {i: k+1 for k, i in enumerate(full_peaks)}
            band_ranks = {i: k+1 for k, i in enumerate(band_genuine)}
            crop_ranks = {i: k+1 for k, i in enumerate(crop_peaks)}
            median_power = float(np.median(power[band]))
            for index, entry in sorted(info.items()):
                prom = entry['prominence_power']
                relative_amplitude = float(amplitude[index]/amplitude[winner])
                prominence_fraction = float(prom/power[index]) if prom is not None and power[index] > 0 else None
                floor_ratio = float(power[index]/median_power) if median_power > 0 else None
                qualified = bool(index in genuine and band[index] and relative_amplitude >= .1 and
                                 prominence_fraction >= .1 and floor_ratio is not None and floor_ratio >= 3)
                all_peaks.append(dict(record=record, start_s=row.start_s, bin=index, bpm=float(bpm[index]),
                                      power=float(power[index]), amplitude_rad=float(amplitude[index]),
                                      in_band=bool(band[index]), full_rank=full_ranks.get(index),
                                      genuine_band_rank=band_ranks.get(index), crop_rank=crop_ranks.get(index),
                                      relative_amplitude=relative_amplitude, prominence_fraction=prominence_fraction,
                                      power_over_band_median=floor_ratio, radar_qualified=qualified, **entry))
            boundary = bool(bpm[winner] <= 47.34375 or bpm[winner] >= 147.65625)
            kind = 'genuine_full_local_peak' if winner in genuine else 'nonpeak_other'
            parent = None
            monotone = None
            if winner == left and winner not in genuine and power[winner-1] > power[winner] >= power[winner+1]:
                kind = 'lower_cut_descending_flank'
                outside = [i for i in full_peaks if i < winner]
                if outside:
                    parent = max(outside)
                    monotone = bool(np.all(np.diff(power[parent:winner+1]) <= 0))
            if winner == right and winner not in genuine and power[winner+1] > power[winner] >= power[winner-1]:
                kind = 'upper_cut_ascending_flank'
                outside = [i for i in full_peaks if i > winner]
                if outside:
                    parent = min(outside)
                    monotone = bool(np.all(np.diff(power[winner:parent+1]) >= 0))
            def ratio(first_index, second_index, values):
                return float(values[first_index]/values[second_index]) if values[second_index] > 0 else None
            second_crop = crop_peaks[1] if len(crop_peaks) > 1 else None
            all_windows.append(dict(record=record, **row._asdict(), selected_bin=winner,
                selected_amplitude_rad=float(amplitude[winner]), selected_is_full_local=winner in genuine,
                selected_boundary_region=boundary, selected_exact_endpoint=winner in (left, right),
                winner_kind=kind, outside_parent_bpm=float(bpm[parent]) if parent is not None else None,
                outside_parent_amplitude_rad=float(amplitude[parent]) if parent is not None else None,
                monotone_parent_to_boundary=monotone, full_global_bpm=float(bpm[np.argmax(power)]),
                full_local_main_bpm=float(bpm[full_peaks[0]]), full_local_second_bpm=float(bpm[full_peaks[1]]),
                full_main_second_amplitude_ratio=ratio(full_peaks[0], full_peaks[1], amplitude),
                full_main_second_power_ratio=ratio(full_peaks[0], full_peaks[1], power),
                crop_second_bpm=float(bpm[second_crop]) if second_crop is not None else None,
                crop_main_second_amplitude_ratio=ratio(winner, second_crop, amplitude) if second_crop is not None else None,
                crop_main_second_power_ratio=ratio(winner, second_crop, power) if second_crop is not None else None,
                band_median_power=median_power, full_local_count=len(genuine),
                genuine_band_count=len(band_genuine), crop_peak_count=len(crop_peaks)))
        np.savez_compressed(radar/f'{record}_spectra.npz', power=np.array(powers), bpm=bpm,
                            window_starts_s=table.start_s.to_numpy())
    pd.DataFrame(all_windows).to_csv(radar/'windows.csv', index=False)
    pd.DataFrame(all_peaks).to_csv(radar/'peaks.csv', index=False)
    execution = dict(elapsed_seconds=time.perf_counter()-began, windows=len(all_windows),
                     peaks=len(all_peaks), decoded_csv_columns=config['radar_only_columns'],
                     ecg_reference_columns_decoded=False, original_predictions_exact=True)
    (radar/'execution.json').write_text(json.dumps(execution, indent=2), encoding='utf-8')
    manifest = {p.name: digest(p) for p in sorted(radar.iterdir()) if p.is_file()}
    (ROOT/'radar_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(execution, indent=2))
    print(pd.DataFrame(all_windows).groupby(['record', 'winner_kind']).size().to_string())


if __name__ == '__main__':
    main()
