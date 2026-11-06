"""Frozen known-heartbeat task readout using archived A/C/D synthetic inputs."""
from pathlib import Path
import ast
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy import signal

from synchronized_ellipse_fixed import ellipse_phase, fit_calibration


ROOT = Path('experiment_runs/ellipse_readout_controls')
PARENT = Path('experiment_runs/ellipse_phase_controls')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def original_reader():
    tree = ast.parse(Path('run_synchronized_real_pilot.py').read_text(encoding='utf-8'))
    process = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'process')
    loop = next(node for node in process.body if isinstance(node, ast.For))
    names = ['spectrum', 'freq', 'band', 'prediction']
    assignments = [node for node in loop.body if isinstance(node, ast.Assign)
                   and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names]
    assert [node.targets[0].id for node in assignments] == names
    namespace = {'np': np, 'signal': signal}
    code = compile(ast.Module(body=assignments, type_ignores=[]), '<frozen original FFT reader>', 'exec')

    def reader(trace):
        local = {'trace': trace}
        exec(code, namespace, local)
        return [local[name] for name in names]

    return reader, '\n'.join(ast.unparse(node) for node in assignments)


def absent_observation(data, case, parent_protocol):
    t = data['time_s']
    phase = data['true_phase_rad'] - .08*np.sin(2*np.pi*1.2*t)
    radius = np.ones_like(phase)
    angle = phase.copy()
    if case.startswith('D_'):
        radius += .12*np.cos(4*phase)
    if case == 'D_warp':
        angle += .6*np.sin(2*phase)
    circle = radius[:, None]*np.c_[np.cos(angle), np.sin(angle)]
    xy = (circle+data['true_center_shift_circle'])@np.array(parent_protocol['affine'])
    xy += parent_protocol['center_native']
    xy += data['noise_native']
    return xy, phase


def peak_diagnostics(trace, power, bpm, band, protocol):
    weighted = signal.detrend(trace)*signal.windows.hann(512, sym=False)
    coherent_gain = signal.windows.hann(512, sym=False).sum()
    amplitude = 2*np.sqrt(power)/coherent_gain
    target = protocol['target_bpm']
    native = protocol['native_resolution_bpm']
    band_indices = np.flatnonzero(band)
    band_power = power[band]
    target_mask = abs(bpm-target) <= native/2
    low_region = band & (bpm <= 45+native)
    high_region = band & (bpm >= 150-native)
    nearest = int(np.argmin(abs(bpm-target)))
    exact = abs(np.dot(weighted, np.exp(-2j*np.pi*1.2*np.arange(512)/20)))*2/coherent_gain
    local_peaks = signal.find_peaks(band_power)[0].tolist()
    if band_power[0] >= band_power[1]:
        local_peaks.append(0)
    if band_power[-1] >= band_power[-2]:
        local_peaks.append(len(band_power)-1)
    peaks = sorted((int(band_indices[index]) for index in set(local_peaks)),
                   key=lambda index: (-power[index], index))
    competition = [index for index in peaks if abs(bpm[index]-target) > native]
    competitor = competition[0] if competition else None
    selected = int(band_indices[np.argmax(band_power)])
    low_bin = int(np.flatnonzero(low_region)[np.argmax(power[low_region])])
    high_bin = int(np.flatnonzero(high_region)[np.argmax(power[high_region])])
    total_power = power[band].sum()
    result = {
        'target_exact_amplitude_rad': float(exact),
        'target_nearest_bin_bpm': float(bpm[nearest]),
        'target_nearest_bin_amplitude_rad': float(amplitude[nearest]),
        'target_region_max_amplitude_rad': float(amplitude[target_mask].max()),
        'target_region_power': float(power[target_mask].sum()),
        'target_region_power_fraction': float(power[target_mask].sum()/total_power) if total_power > 0 else None,
        'selected_amplitude_rad': float(amplitude[selected]),
        'competitor_bpm': float(bpm[competitor]) if competitor is not None else None,
        'competitor_amplitude_rad': float(amplitude[competitor]) if competitor is not None else None,
        'lower_endpoint_amplitude_rad': float(amplitude[band_indices[0]]),
        'upper_endpoint_amplitude_rad': float(amplitude[band_indices[-1]]),
        'lower_region_peak_bpm': float(bpm[low_bin]),
        'lower_region_peak_amplitude_rad': float(amplitude[low_bin]),
        'upper_region_peak_bpm': float(bpm[high_bin]),
        'upper_region_peak_amplitude_rad': float(amplitude[high_bin]),
        'selected_band_endpoint': bool(selected in [band_indices[0], band_indices[-1]]),
        'selected_boundary_region': bool(low_region[selected] or high_region[selected]),
        'within_72_reference': bool(abs(bpm[selected]-target) <= native/2),
        'spectrum_finite': bool(np.isfinite(power).all()),
    }
    peak_rows = [dict(rank=rank+1, bpm=float(bpm[index]), amplitude_rad=float(amplitude[index]),
                      power=float(power[index]), endpoint=bool(index in [band_indices[0], band_indices[-1]]))
                 for rank, index in enumerate(peaks)]
    return result, peak_rows


def main():
    started = time.perf_counter()
    protocol = read(ROOT/'protocol.json')
    hashes = read(ROOT/'source_hashes.json')
    for name, digest in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    for name, digest in read(PARENT/'artifact_manifest.json').items():
        assert hashlib.sha256((PARENT/name).read_bytes()).hexdigest() == digest, name
    assert not (ROOT/'windows.csv').exists(), 'Frozen readout already executed'
    parent_protocol = read(PARENT/'protocol.json')
    reader, reader_text = original_reader()
    (ROOT/'extracted_reader.py.txt').write_text(reader_text, encoding='utf-8')
    (ROOT/'signals').mkdir(exist_ok=True)
    windows = []
    peaks = []
    calibrations = []
    for case in protocol['cases']:
        for seed in protocol['seeds']:
            for sigma in protocol['noise_std_native']:
                parent_tag = f'{case}_seed{seed}_noise{sigma:g}'
                data = dict(np.load(PARENT/'signals'/f'{parent_tag}.npz'))
                for heartbeat in protocol['heartbeat_conditions']:
                    tag = parent_tag+'_'+heartbeat
                    common = dict(tag=tag, case=case, seed=seed, noise_std=sigma, heartbeat=heartbeat)
                    if heartbeat == 'present':
                        xy, truth, recovered = data['iq'], data['true_phase_rad'], data['raw_recovered_phase_rad']
                        try:
                            origin, scale, center, transform, meta = fit_calibration(xy[:3000])
                            mapped = ((xy-origin)/scale-center)@transform
                            low = np.linalg.norm(mapped, axis=1) < .1
                            accepted = True
                        except ValueError as exc:
                            raise AssertionError('Archived accepted fit changed') from exc
                    else:
                        xy, truth = absent_observation(data, case, parent_protocol)
                        try:
                            recovered, low, meta = ellipse_phase(xy[:, 0], xy[:, 1], 100)
                            accepted = True
                        except ValueError as exc:
                            accepted = False
                            recovered = np.full_like(truth, np.nan)
                            low = np.ones_like(truth, dtype=bool)
                            meta = {'rejection': str(exc)}
                    calibrations.append(dict(**common, accepted=accepted, **meta))
                    phase20 = np.stack([signal.resample_poly(truth, 1, 5), signal.resample_poly(recovered, 1, 5)])
                    low20 = signal.resample_poly(low.astype(float), 1, 5) > .01
                    powers = []
                    for source_index, source in enumerate(protocol['phase_sources']):
                        source_power = []
                        for start in protocol['window_starts_s']:
                            first = round(start*20)
                            trace = phase20[source_index, first:first+512]
                            row = dict(**common, source=source, start_s=start, end_s=start+25.6,
                                       calibration_accepted=accepted,
                                       low_amplitude_fraction=float(low20[first:first+512].mean()))
                            if source == 'recovered' and not accepted:
                                row['readout_available'] = False
                                source_power.append(np.full(2049, np.nan))
                            else:
                                power, bpm, band, prediction = reader(trace)
                                row.update(readout_available=True, prediction_bpm=prediction)
                                extra, local_peaks = peak_diagnostics(trace, power, bpm, band, protocol)
                                row.update(extra)
                                for peak in local_peaks:
                                    peaks.append(dict(**common, source=source, start_s=start, **peak))
                                source_power.append(power)
                            windows.append(row)
                        powers.append(source_power)
                    np.savez_compressed(ROOT/'signals'/f'{tag}.npz', time_s=data['time_s'], iq=xy,
                                        true_phase_rad=truth, recovered_phase_rad=recovered,
                                        phase20=phase20, low_amplitude20=low20, power=np.array(powers))
    pd.DataFrame(windows).to_csv(ROOT/'windows.csv', index=False)
    pd.DataFrame(peaks).to_csv(ROOT/'peaks.csv', index=False)
    pd.DataFrame(calibrations).to_json(ROOT/'calibrations.json', orient='records', indent=2)
    result = dict(elapsed_seconds=time.perf_counter()-started, paired_inputs=30,
                  input_conditions=60, sources=2, windows=len(windows), real_signal_arrays_read=False,
                  environment='local uv-managed .venv', source_hashes=hashes)
    (ROOT/'execution.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    frame = pd.DataFrame(windows)
    print(frame.groupby(['case', 'heartbeat', 'source'])['prediction_bpm'].agg(['min', 'max']).to_string())


if __name__ == '__main__':
    main()
