"""Finite, nondegenerate algebraic I/Q ellipse calibration; no HR labels."""
import numpy as np
from scipy import linalg, signal


def mapped_support(points):
    if not np.isfinite(points).all(): raise ValueError('Nonfinite mapped calibration')
    radius = np.linalg.norm(points, axis=1)
    if np.any(radius <= 1e-12): raise ValueError('Undefined mapped phase')
    centered = points-points.mean(axis=0)
    singular = np.linalg.svd(centered/np.sqrt(len(points)), compute_uv=False)
    unit = points/radius[:, None]
    circular_variance = float(1-np.linalg.norm(unit.mean(axis=0)))
    angle = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    span = float(np.quantile(angle, .99)-np.quantile(angle, .01))
    # Numerical support guards, not evidence of physiological correctness.
    if singular[0] < 1e-3 or singular[1] < 1e-6 or circular_variance < 1e-6 or span < .01:
        raise ValueError('Mapped calibration collapses to a point or degenerate arc')
    return {'mapped_centered_singular_values': singular.tolist(), 'mapped_circular_variance': circular_variance,
            'mapped_phase_q99_q01_span_rad': span, 'calibration_radial_rmse': float(np.sqrt(np.mean((radius-1)**2)))}


def fit_calibration(calibration):
    calibration = np.asarray(calibration, dtype=float)
    if calibration.ndim != 2 or calibration.shape[1] != 2 or len(calibration) < 12 or not np.isfinite(calibration).all():
        raise ValueError('Invalid calibration input')
    origin, scale = calibration.mean(0), calibration.std(0)
    if np.any(scale < 1e-12): raise ValueError('Constant I/Q calibration')
    norm = (calibration-origin)/scale
    if np.linalg.svd(norm, compute_uv=False)[1]/np.linalg.svd(norm, compute_uv=False)[0] < 1e-6:
        raise ValueError('Collinear I/Q calibration')
    x, y = norm.T
    design = np.column_stack([x*x, x*y, y*y, x, y, np.ones(len(x))])
    scatter = design.T@design
    constraint = np.zeros((6, 6)); constraint[0, 2] = constraint[2, 0] = 2; constraint[1, 1] = -1
    values, vectors = linalg.eig(scatter, constraint, homogeneous_eigvals=True)
    candidates, rejected = [], []
    for index, (alpha, beta, vector) in enumerate(zip(values[0], values[1], vectors.T)):
        try:
            if not np.isfinite([alpha, beta]).all() or abs(beta) <= 64*np.finfo(float).eps*(abs(alpha)+abs(beta)):
                raise ValueError('Nonfinite generalized eigenvalue')
            value = alpha/beta
            if not np.isfinite(value) or abs(value.imag) > 1e-8*max(1, abs(value.real)) or np.max(abs(vector.imag)) > 1e-8:
                raise ValueError('Complex/nonfinite eigenpair')
            v = vector.real
            quadratic_norm = np.linalg.norm(v[:3])
            if quadratic_norm < 1e-10*np.linalg.norm(v): raise ValueError('Vanishing quadratic coefficients')
            v = v/quadratic_norm
            residual = np.linalg.norm(scatter@v-value.real*(constraint@v))/max((np.linalg.norm(scatter)+abs(value)*np.linalg.norm(constraint))*np.linalg.norm(v), 1e-12)
            if residual > 1e-7: raise ValueError('Invalid eigenpair residual')
            a, b, c = v[:3]
            if 4*a*c-b*b <= 1e-12: raise ValueError('Not an ellipse')
            quad = np.array([[a, b/2], [b/2, c]])
            if np.trace(quad) < 0: quad, v = -quad, -v
            if np.linalg.eigvalsh(quad).min() <= 0: raise ValueError('Nonpositive quadratic form')
            center = -.5*np.linalg.solve(quad, v[3:5])
            level = center@quad@center-v[5]
            if not np.isfinite(level) or level <= 1e-12*max(abs(center@quad@center), abs(v[5]), 1):
                raise ValueError('Degenerate ellipse level')
            eig, basis = np.linalg.eigh(quad/level)
            if not np.isfinite(eig).all() or eig.min() <= 0 or np.sqrt(eig.max()/eig.min()) > 100:
                raise ValueError('Invalid ellipse axes')
            transform = (basis*np.sqrt(eig))@basis.T
            support = mapped_support((norm-center)@transform)
            if support['calibration_radial_rmse'] > .2: raise ValueError('Radial residual exceeds original limit')
            candidates.append((support['calibration_radial_rmse'], center, transform, dict(support,
                generalized_eigenvalue=float(value.real), generalized_eigenpair_relative_residual=float(residual))))
        except (ValueError, np.linalg.LinAlgError) as exc:
            rejected.append({'candidate': index, 'reason': str(exc)})
    if not candidates: raise ValueError('No valid calibration ellipse: '+str(rejected))
    _, center, transform, support = min(candidates, key=lambda item: item[0])
    return origin, scale, center, transform, dict(support, rejected_candidates=rejected)


def ellipse_phase(i, q, fs):
    xy = np.column_stack([i, q])
    if not np.isfinite(xy).all(): raise ValueError('Nonfinite I/Q')
    calibration = xy[:round(30*fs):max(1, round(fs/100))]
    origin, scale, center, transform, meta = fit_calibration(calibration)
    mapped = ((xy-origin)/scale-center)@transform
    z = mapped[:, 0]+1j*mapped[:, 1]
    if not np.isfinite(z).all(): raise ValueError('Nonfinite corrected I/Q')
    increments = np.angle(z[1:]*z[:-1].conj())
    phase = np.r_[0., np.cumsum(increments)]
    low = abs(z) < .1
    meta.update(origin=origin.tolist(), scale=scale.tolist(), ellipse_center=center.tolist(), ellipse_transform=transform.tolist(),
                low_amplitude_fraction=float(low.mean()), phase_increment_q999_rad=float(np.quantile(abs(increments), .999)))
    return phase, low, meta
