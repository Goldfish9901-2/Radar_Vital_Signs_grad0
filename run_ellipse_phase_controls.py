"""Run frozen, paired known-phase synthetic controls in the local uv environment."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd

from audit_ellipse_applicability import enumerate_fit, map_points, point_metrics, transfer_pair
from synchronized_ellipse_fixed import ellipse_phase


ROOT = Path('experiment_runs/ellipse_phase_controls')


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding='utf-8')


def base_phase(t):
    return 0.2 + 2*np.pi*0.17*t + 0.08*np.sin(2*np.pi*1.2*t)


def generate(case_id, t, noise, protocol):
    phase = base_phase(t)
    if case_id == 'B_dwell':
        phase = phase + 0.9*np.sin(phase)
    if case_id == 'B_arc':
        phase = np.where(t < 15, 0.2 + 0.75*np.sin(2*np.pi*0.2*t), base_phase(t-15))
    radius = np.ones(len(t))
    angle = phase.copy()
    if case_id.startswith('D_'):
        radius += 0.12*np.cos(4*phase)
    if case_id == 'D_warp':
        angle += 0.6*np.sin(2*phase)
    circle = radius[:, None]*np.column_stack([np.cos(angle), np.sin(angle)])
    center_shift = np.zeros_like(circle)
    if case_id.startswith('C_'):
        strength = float(case_id.split('_')[1])
        center_shift = strength*(t[:, None]/protocol['duration_s'])*np.array([0.8, 0.6])
    points = (circle+center_shift)@np.array(protocol['affine']) + protocol['center_native'] + noise
    return points, phase, center_shift


def phase_metrics(recovered, truth, cal, test, quality):
    candidates = []
    for sign in [-1, 1]:
        offset = np.mean(truth[cal]-sign*recovered[cal])
        error = sign*recovered+offset-truth
        candidates.append((np.mean(error[cal]**2), sign, offset, error))
    _, sign, offset, error = min(candidates, key=lambda row: row[0])
    aligned = sign*recovered+offset
    pair_mask = test[1:] & test[:-1]
    increment_error = np.diff(aligned)-np.diff(truth)
    cycle = np.rint(error/(2*np.pi)).astype(int)
    transition_mask = test[1:]
    phase_rmse = float(np.sqrt(np.mean(error[test]**2)))
    increment_rmse = float(np.sqrt(np.mean(increment_error[pair_mask]**2)))
    cycle_fraction = float(np.mean(cycle[test] != 0))
    result = {
        'alignment_sign': sign,
        'alignment_offset_rad': float(offset),
        'cal_phase_rmse_rad': float(np.sqrt(np.mean(error[cal]**2))),
        'test_phase_rmse_rad': phase_rmse,
        'test_increment_rmse_rad_per_sample': increment_rmse,
        'test_max_abs_phase_error_rad': float(np.max(abs(error[test]))),
        'test_cycle_index_mismatch_fraction': cycle_fraction,
        'test_cycle_index_transitions': int(np.sum((np.diff(cycle) != 0) & transition_mask)),
        'test_cycle_index_final': int(cycle[-1]),
        'meets_phase_reference': bool(phase_rmse <= quality['test_phase_rmse_max_rad'] and
                                      increment_rmse <= quality['test_increment_rmse_max_rad_per_sample'] and
                                      cycle_fraction <= quality['test_cycle_index_mismatch_max_fraction']),
    }
    return result, aligned, error, cycle


def main():
    started = time.perf_counter()
    protocol = json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    frozen_hashes = json.loads((ROOT/'source_hashes.json').read_text(encoding='utf-8'))
    for name, expected in frozen_hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    assert not (ROOT/'metrics.csv').exists(), 'Frozen experiment already executed; use a new output for a justified rerun'
    signals = ROOT/'signals'
    signals.mkdir(exist_ok=True)
    t = np.arange(round(protocol['duration_s']*protocol['fs_hz']))/protocol['fs_hz']
    cal = t < 30
    test = t >= 30
    metrics = []
    diagnostics = []
    transfers = []
    blocks = []
    for seed in protocol['seeds']:
        normal_noise = np.random.default_rng(seed).normal(size=(len(t), 2))
        for sigma in protocol['noise_std_native']:
            noise = sigma*normal_noise
            for case in protocol['cases']:
                case_id = case['id']
                tag = f'{case_id}_seed{seed}_noise{sigma:g}'
                points, truth, center_shift = generate(case_id, t, noise, protocol)
                assert np.max(abs(np.diff(truth))) < np.pi
                fits = {}
                for name, first, last in protocol['fit_intervals']:
                    selection = (t >= first) & (t < last)
                    fits[name] = enumerate_fit(points[selection], name, first, last)
                primary = fits['cal_0_30']
                common = dict(tag=tag, case=case_id, group=case['group'], seed=seed, noise_std=sigma)
                diagnostics.append(dict(**common, fits=list(fits.values())))
                for early, late in protocol['transfer_pairs']:
                    a, b = fits[early], fits[late]
                    xa = points[(t >= a['first_s']) & (t < a['last_s'])]
                    xb = points[(t >= b['first_s']) & (t < b['last_s'])]
                    transfers.append(dict(**common, **transfer_pair(a, b, xa, xb, early+'->'+late)))
                proper = [c for c in primary['candidates'] if c['geometry_defined']]
                row = dict(**common, production_accepted=primary['production_valid'],
                           rejection_reason=primary.get('production_error'),
                           diagnostic_only=not primary['production_valid'])
                arrays = dict(time_s=t, iq=points, true_phase_rad=truth, noise_native=noise,
                              true_center_shift_circle=center_shift)
                if primary['production_valid']:
                    recovered, low, meta = ellipse_phase(points[:, 0], points[:, 1], protocol['fs_hz'])
                    candidate = next(c for c in proper if c.get('production_selected'))
                    row['low_amplitude_test_fraction'] = float(low[test].mean())
                elif len(proper) == 1:
                    candidate = proper[0]
                    mapped = map_points(points, candidate)
                    z = mapped[:, 0] + 1j*mapped[:, 1]
                    recovered = np.r_[0.0, np.cumsum(np.angle(z[1:]*z[:-1].conj()))]
                else:
                    row['phase_unavailable_reason'] = 'No unique proper ellipse'
                    metrics.append(row)
                    np.savez_compressed(signals/f'{tag}.npz', **arrays)
                    continue
                mapped = map_points(points, candidate)
                z = mapped[:, 0] + 1j*mapped[:, 1]
                unwrapped = np.unwrap(np.angle(z))
                row['unwrap_implementation_max_diff_rad'] = float(np.max(abs(recovered-(unwrapped-unwrapped[0]))))
                row['cal_radial_rmse'] = candidate['radial_rmse']
                row['test_radial_rmse'] = point_metrics(mapped[test])['radial_rmse']
                result, aligned, error, cycle = phase_metrics(recovered, truth, cal, test, protocol['phase_quality_reference'])
                row.update(result)
                row['accepted_but_phase_reference_failed'] = bool(primary['production_valid'] and not result['meets_phase_reference'])
                for k in range(6):
                    selection = (t >= 5*k) & (t < 5*(k+1))
                    blocks.append(dict(**common, block=k+1, **point_metrics(mapped[selection])))
                arrays.update(raw_recovered_phase_rad=recovered, aligned_phase_rad=aligned,
                              phase_error_rad=error, cycle_index=cycle)
                np.savez_compressed(signals/f'{tag}.npz', **arrays)
                metrics.append(row)
    pd.DataFrame(metrics).to_csv(ROOT/'metrics.csv', index=False)
    pd.DataFrame(transfers).to_csv(ROOT/'transfer_metrics.csv', index=False)
    write_json(ROOT/'candidate_diagnostics.json', diagnostics)
    write_json(ROOT/'transfer_metrics.json', transfers)
    write_json(ROOT/'block_metrics.json', blocks)
    write_json(ROOT/'execution.json', dict(elapsed_seconds=time.perf_counter()-started,
               runs=len(metrics), transfer_pairs=len(transfers), environment='local uv-managed .venv',
               real_signal_arrays_read=False, hr_predictions_generated=False,
               source_hashes=frozen_hashes, numpy_version=np.__version__))
    print(pd.DataFrame(metrics).groupby('case')[['production_accepted', 'test_phase_rmse_rad', 'accepted_but_phase_reference_failed']].mean().to_string())


if __name__ == '__main__':
    main()
