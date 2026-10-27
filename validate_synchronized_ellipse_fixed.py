"""Analytic known-phase and degeneracy regression checks, without real HR."""
from pathlib import Path
import ast
import json
import time
import numpy as np
from scipy import linalg
from synchronized_ellipse_fixed import ellipse_phase, fit_calibration, mapped_support


def main():
    started = time.perf_counter(); rows = []
    fs = 100.; t = np.arange(6000)/fs
    for name, phase in [('full_ellipse', 2*np.pi*1.17*t+.3),
                        ('noisy_ellipse', 2*np.pi*1.17*t+.3),
                        ('partial_breathing_arc', 1.4*np.sin(2*np.pi*.23*t)+.03*np.sin(2*np.pi*1.17*t))]:
        rotation = np.array([[np.cos(.6), -np.sin(.6)], [np.sin(.6), np.cos(.6)]])
        affine = rotation@np.diag([3., .7])@rotation.T
        xy = np.column_stack([np.cos(phase), np.sin(phase)])@affine+np.array([5., -8.])
        if name == 'noisy_ellipse': xy += np.random.default_rng(20261027).normal(0, .002, xy.shape)
        recovered, _, meta = ellipse_phase(xy[:, 0], xy[:, 1], fs)
        error = np.sqrt(np.mean((recovered-(phase-phase[0]))**2))
        assert error < (.02 if name == 'noisy_ellipse' else 1e-6), (name, error)
        rows.append({'case': name, 'phase_rmse_rad': float(error), 'passed': True,
                     'circular_variance': meta['mapped_circular_variance'], 'rejections': meta['rejected_candidates']})
        if name == 'noisy_ellipse':
            tree = ast.parse(Path('run_synchronized_real_pilot.py').read_text(encoding='utf-8'))
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'ellipse_phase')
            namespace = {'np': np, 'linalg': linalg}
            exec(compile(ast.Module(body=[function], type_ignores=[]), '<archived original ellipse>', 'exec'), namespace)
            old_phase, _, old_meta = namespace['ellipse_phase'](xy[:, 0], xy[:, 1], fs)
            old_error = float(np.sqrt(np.mean((old_phase-(phase-phase[0]))**2)))
            rows[-1].update(original_phase_rmse_rad=old_error, original_radial_rmse=old_meta['calibration_radial_rmse'])
    bad = {'constant': np.ones((3000, 2)), 'line': np.column_stack([t[:3000], 2*t[:3000]+1]),
           'nonfinite': np.full((3000, 2), np.nan)}
    for name, points in bad.items():
        try: fit_calibration(points)
        except ValueError as exc: rows.append({'case': name, 'passed': True, 'rejected': str(exc)})
        else: raise AssertionError('Accepted '+name)
    # Radial residual is exactly zero for this collapsed point on the unit circle.
    try: mapped_support(np.tile([1., 0.], (3000, 1)))
    except ValueError as exc: rows.append({'case': 'unit_circle_point_collapse', 'radial_rmse': 0., 'passed': True, 'rejected': str(exc)})
    else: raise AssertionError('Accepted collapsed calibrated points')
    out = Path('experiment_runs/synchronized_real_pilot_fixed'); out.mkdir(parents=True, exist_ok=True)
    result = {'elapsed_seconds': time.perf_counter()-started, 'real_radar_or_hr_read': False, 'cases': rows}
    (out/'analytic_validation.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
