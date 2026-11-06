"""Verify saved synthetic controls and plot phase error against geometry diagnostics."""
from pathlib import Path
import hashlib
import json
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg

from synchronized_ellipse_fixed import fit_calibration


ROOT = Path('experiment_runs/ellipse_phase_controls')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def circle_map(points, candidate):
    normalized = (points-np.array(candidate['origin']))/candidate['scale']
    return (normalized-candidate['center_normalized'])@candidate['transform']


def radius_rmse(points, candidate):
    mapped = circle_map(points, candidate)
    radius = np.hypot(mapped[:, 0], mapped[:, 1])
    return float(np.sqrt(np.mean((radius-1)**2)))


def support_arc95(mapped):
    angles = np.sort(np.mod(np.arctan2(mapped[:, 1], mapped[:, 0]), 2*np.pi))
    count = int(np.ceil(.95*len(angles)))
    doubled = np.r_[angles, angles+2*np.pi]
    widths = doubled[count-1:count-1+len(angles)]-angles
    return float(np.min(widths)*180/np.pi)


def main():
    started = time.perf_counter()
    protocol = read(ROOT/'protocol.json')
    frozen = read(ROOT/'source_hashes.json')
    assert all(sha256(Path(name)) == digest for name, digest in frozen.items())
    metrics = pd.read_csv(ROOT/'metrics.csv')
    geometry = {row['tag']: row for row in read(ROOT/'candidate_diagnostics.json')}
    transfers = read(ROOT/'transfer_metrics.json')
    errors = []
    saved = {}
    all_pairs = finite = proper = 0
    for row in metrics.to_dict('records'):
        data = dict(np.load(ROOT/'signals'/f"{row['tag']}.npz"))
        saved[row['tag']] = data
        t = data['time_s']
        points = data['iq']
        truth = data['true_phase_rad']
        base = .2 + (2*np.pi*.17)*t + .08*np.sin((2*np.pi*1.2)*t)
        expected_truth = base.copy()
        if row['case'] == 'B_dwell':
            expected_truth += .9*np.sin(base)
        if row['case'] == 'B_arc':
            shifted = t-15
            later = .2 + 2*np.pi*.17*shifted + .08*np.sin(2*np.pi*1.2*shifted)
            expected_truth = np.where(t < 15, .2+.75*np.sin(2*np.pi*.2*t), later)
        errors.append(float(np.max(abs(truth-expected_truth))))
        expected_noise = row['noise_std']*np.random.default_rng(row['seed']).normal(size=(len(t), 2))
        assert np.array_equal(expected_noise, data['noise_native'])
        calibration = t < 30
        test = t >= 30
        assert len(t) == 6000 and np.max(abs(np.diff(truth))) < np.pi
        fits = geometry[row['tag']]['fits']
        for fit in fits:
            selection = (t >= fit['first_s']) & (t < fit['last_s'])
            candidates = fit['candidates']
            assert len(candidates) == 6
            all_pairs += 6
            finite += sum(c['finite_generalized_eigenvalue'] for c in candidates)
            for candidate in candidates:
                if not candidate['geometry_defined']:
                    continue
                proper += 1
                errors.append(abs(radius_rmse(points[selection], candidate)-candidate['radial_rmse']))
                local_map = circle_map(points[selection], candidate)
                errors.append(abs(support_arc95(local_map)-candidate['mapped_metrics']['angular_95percent_arc_deg']))
        primary = fits[0]
        try:
            origin, scale, center, transform, metadata = fit_calibration(points[calibration])
            accepted = True
        except ValueError:
            accepted = False
        assert accepted == row['production_accepted'] == primary['production_valid']
        # Recover using arctan2 + unwrap, independently of cumulative conjugate products.
        candidate = next(c for c in primary['candidates'] if c['geometry_defined'])
        mapped = circle_map(points, candidate)
        raw = np.unwrap(np.arctan2(mapped[:, 1], mapped[:, 0]))
        raw -= raw[0]
        errors.append(float(np.max(abs(raw-data['raw_recovered_phase_rad']))))
        alternatives = []
        for sign in (-1, 1):
            offset = np.average(truth[calibration]-sign*raw[calibration])
            aligned = sign*raw+offset
            alternatives.append((np.linalg.norm(aligned[calibration]-truth[calibration]), sign, offset, aligned))
        _, sign, offset, aligned = min(alternatives, key=lambda item: item[0])
        assert sign == row['alignment_sign']
        errors.append(abs(offset-row['alignment_offset_rad']))
        phase_error = aligned-truth
        phase_rmse = np.linalg.norm(phase_error[test])/np.sqrt(test.sum())
        increment_error = np.diff(aligned[test])-np.diff(truth[test])
        increment_rmse = np.linalg.norm(increment_error)/np.sqrt(len(increment_error))
        errors.extend([abs(phase_rmse-row['test_phase_rmse_rad']),
                       abs(increment_rmse-row['test_increment_rmse_rad_per_sample']),
                       float(np.max(abs(phase_error-data['phase_error_rad'])))])
        cycle = np.floor(phase_error/(2*np.pi)+0.5).astype(int)
        assert np.array_equal(cycle, data['cycle_index'])
        assert np.mean(cycle[test] != 0) == row['test_cycle_index_mismatch_fraction']
        assert np.sum((np.diff(cycle) != 0) & test[1:]) == row['test_cycle_index_transitions']
        quality = protocol['phase_quality_reference']
        meets = (phase_rmse <= quality['test_phase_rmse_max_rad'] and
                 increment_rmse <= quality['test_increment_rmse_max_rad_per_sample'] and
                 not np.any(cycle[test]))
        assert meets == row['meets_phase_reference']
        assert bool(accepted and not meets) == row['accepted_but_phase_reference_failed']
        # Independently reconstruct the frozen observation model from stored true phase.
        rho = np.ones_like(truth)
        observed = truth.copy()
        if row['group'] == 'D':
            rho += 0.12*np.cos(4*truth)
        if row['case'] == 'D_warp':
            observed += 0.6*np.sin(2*truth)
        latent = rho[:, None]*np.c_[np.cos(observed), np.sin(observed)]
        shift = data['true_center_shift_circle']
        expected_shift = np.zeros_like(shift)
        if row['group'] == 'C':
            strength = float(row['case'].split('_')[1])
            expected_shift = strength*(t[:, None]/60)*np.array([.8, .6])
        assert np.array_equal(shift, expected_shift)
        expected = (latent+shift)@np.array(protocol['affine']) + protocol['center_native'] + data['noise_native']
        errors.append(float(np.max(abs(expected-points))))
        if row['group'] in ('C', 'D'):
            a_tag = f"A_seed{row['seed']}_noise{row['noise_std']:g}"
            assert np.array_equal(truth, saved[a_tag]['true_phase_rad'])
            assert np.array_equal(data['noise_native'], saved[a_tag]['noise_native'])
    for row in transfers:
        data = saved[row['tag']]
        t, points = data['time_s'], data['iq']
        fits = {f['fit']: f for f in geometry[row['tag']]['fits']}
        early, late = row['pair'].split('->')
        fa, fb = fits[early], fits[late]
        a = next(c for c in fa['candidates'] if c['geometry_defined'])
        b = next(c for c in fb['candidates'] if c['geometry_defined'])
        xa = points[(t >= fa['first_s']) & (t < fa['last_s'])]
        xb = points[(t >= fb['first_s']) & (t < fb['last_s'])]
        penalty = max(radius_rmse(xb, a)-radius_rmse(xb, b), radius_rmse(xa, b)-radius_rmse(xa, a))
        delta = np.array(b['center_native'])-a['center_native']
        center_change = float(np.sqrt(delta@np.array(a['native_shape_matrix'])@delta))
        shape_change = float(np.linalg.norm(np.log(linalg.eigvalsh(b['native_shape_matrix'], a['native_shape_matrix']))))
        errors.extend([abs(penalty-row['max_transfer_penalty']),
                       abs(center_change-row['center_shift_in_early_radius_units']),
                       abs(shape_change-row['log_shape_change'])])
        ma, mb = circle_map(xa, a), circle_map(xb, a)
        ia = (np.mod(np.arctan2(ma[:, 1], ma[:, 0]), 2*np.pi)*36/(2*np.pi)).astype(int)
        ib = (np.mod(np.arctan2(mb[:, 1], mb[:, 0]), 2*np.pi)*36/(2*np.pi)).astype(int)
        ra, rb = np.hypot(ma[:, 0], ma[:, 1]), np.hypot(mb[:, 0], mb[:, 1])
        differences = [abs(np.median(ra[ia == k])-np.median(rb[ib == k])) for k in range(36)
                       if np.count_nonzero(ia == k) >= 10 and np.count_nonzero(ib == k) >= 10]
        assert len(differences) == row['matched_angular_bins']
        matched = float(np.median(differences))
        errors.append(abs(matched-row['matched_angle_radial_change']))
        coverage = min(support_arc95(circle_map(xa, a)), support_arc95(circle_map(xb, b)))
        errors.append(abs(coverage-row['minimum_arc95_deg']))
        support = len(differences) >= 6 and coverage >= 180
        drift = (support and fa['production_valid'] and fb['production_valid'] and matched >= .1 and
                 penalty >= .1 and (center_change >= .25 or shape_change >= .25))
        stable = support and matched <= .05 and penalty <= .05 and center_change <= .1 and shape_change <= .1
        expected = 'inconclusive'
        if drift:
            expected = 'time_dependent_mismatch_supported'
        elif stable:
            expected = 'stable_geometry_supported'
        assert row['classification'] == expected
    blocks = read(ROOT/'block_metrics.json')
    for row in blocks:
        data = saved[row['tag']]
        t = data['time_s']
        selection = (t >= (row['block']-1)*5) & (t < row['block']*5)
        fit = geometry[row['tag']]['fits'][0]
        candidate = next(c for c in fit['candidates'] if c['geometry_defined'])
        mapped = circle_map(data['iq'][selection], candidate)
        errors.append(abs(radius_rmse(data['iq'][selection], candidate)-row['radial_rmse']))
        errors.append(abs(support_arc95(mapped)-row['angular_95percent_arc_deg']))
    assert len(metrics) == 42 and len(transfers) == 84 and all_pairs == 1008
    assert max(errors) < 1e-9, max(errors)
    old_artifacts = {}
    for folder in ['synchronized_real_pilot', 'synchronized_real_pilot_fixed', 'ellipse_applicability']:
        root = ROOT.parent/folder
        manifest = read(root/'artifact_manifest.json')
        assert all(sha256(root/name) == digest for name, digest in manifest.items())
        old_artifacts[folder] = len(manifest)
    summary = metrics.groupby('case', sort=False).agg(
        runs=('tag', 'size'), accepted=('production_accepted', 'sum'),
        cal_radial_rmse=('cal_radial_rmse', 'mean'), test_radial_rmse=('test_radial_rmse', 'mean'),
        test_phase_rmse_rad=('test_phase_rmse_rad', 'mean'),
        test_increment_rmse_rad_per_sample=('test_increment_rmse_rad_per_sample', 'mean'),
        false_accepts=('accepted_but_phase_reference_failed', 'sum'),
        cycle_mismatch_runs=('test_cycle_index_mismatch_fraction', lambda x: int((x > 0).sum())))
    summary.to_csv(ROOT/'summary.csv')
    metrics.groupby(['case', 'noise_std']).agg(
        runs=('tag', 'size'), cal_radial_rmse=('cal_radial_rmse', 'mean'),
        test_radial_rmse=('test_radial_rmse', 'mean'), test_phase_rmse_rad=('test_phase_rmse_rad', 'mean'),
        test_increment_rmse_rad_per_sample=('test_increment_rmse_rad_per_sample', 'mean'),
        max_abs_phase_error_rad=('test_max_abs_phase_error_rad', 'max'),
        false_accepts=('accepted_but_phase_reference_failed', 'sum')).to_csv(ROOT/'noise_summary.csv')
    counts = pd.crosstab([pd.Series([r['case'] for r in transfers], name='case'),
                         pd.Series([r['pair'] for r in transfers], name='pair')],
                        pd.Series([r['classification'] for r in transfers], name='classification'))
    counts.to_csv(ROOT/'classification_counts.csv')
    plots = ROOT/'plots'
    plots.mkdir(exist_ok=True)
    for case in protocol['cases']:
        # Display rule uses first declared seed and higher declared noise, independent of errors.
        tag = f"{case['id']}_seed{protocol['seeds'][0]}_noise{protocol['noise_std_native'][-1]:g}"
        data = saved[tag]
        t = data['time_s']
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        axes[0, 0].scatter(*data['iq'].T, c=t, s=2, cmap='viridis')
        axes[0, 0].set(xlabel='I (synthetic units)', ylabel='Q (synthetic units)', title='Time-coloured I/Q, 0-60s')
        axes[0, 0].set_aspect('equal', adjustable='box')
        axes[0, 1].plot(t, data['true_phase_rad'], label='Known truth')
        axes[0, 1].plot(t, data['aligned_phase_rad'], label='Calibration-aligned reconstruction', lw=.8)
        axes[0, 1].set(xlabel='Time (s)', ylabel='Phase (rad)')
        axes[0, 1].legend(fontsize=8)
        axes[1, 0].plot(t, data['phase_error_rad'], lw=.7)
        axes[1, 0].axvline(30, color='red', ls='--')
        axes[1, 0].set(xlabel='Time (s)', ylabel='Phase error (rad)', title='Fixed sign and offset from calibration')
        dphi = np.diff(data['aligned_phase_rad'])-np.diff(data['true_phase_rad'])
        axes[1, 1].plot(t[1:], dphi, lw=.5)
        axes[1, 1].axvline(30, color='red', ls='--')
        axes[1, 1].set(xlabel='Time (s)', ylabel='Increment error (rad/sample)')
        fig.suptitle(tag)
        fig.tight_layout()
        fig.savefig(plots/f"{case['id']}.png", dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, group in metrics.groupby('case', sort=False):
        axes[0].scatter(group['cal_radial_rmse'], group['test_phase_rmse_rad'], label=name, alpha=.7)
        axes[1].scatter(group['test_radial_rmse'], group['test_phase_rmse_rad'], label=name, alpha=.7)
    axes[0].axvline(.2, color='black', ls='--', label='Original radial gate')
    for ax in axes:
        ax.axhline(.05, color='red', ls=':', label='Declared phase reference')
        ax.set(ylabel='Test phase RMSE (rad)')
    axes[0].set_xlabel('Calibration radial RMSE')
    axes[1].set_xlabel('Test radial RMSE')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots/'geometry_vs_phase.png', dpi=150)
    plt.close(fig)
    source = ROOT/'source'
    source.mkdir(exist_ok=True)
    for name in ['run_ellipse_phase_controls.py', 'verify_ellipse_phase_controls.py',
                 'audit_ellipse_applicability.py', 'synchronized_ellipse_fixed.py']:
        shutil.copyfile(name, source/name)
    result = dict(status='passed', elapsed_seconds=time.perf_counter()-started, runs=42,
                  fits=168, eigenpairs=all_pairs, finite_candidates=finite, proper_ellipses=proper,
                  transfers=84, blocks=len(blocks), max_absolute_numeric_difference=max(errors),
                  paired_truth_and_noise_verified=True, calibration_only_alignment_verified=True,
                  original_artifacts_unchanged=old_artifacts,
                  scope='Saved synthetic observation laws, alternate unwrap/scoring and diagnostic checks')
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(summary.to_string())


if __name__ == '__main__':
    main()
