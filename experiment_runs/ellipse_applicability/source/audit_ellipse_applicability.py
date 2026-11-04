"""Diagnostics only: all generalized candidates and frozen-model time transfer."""
import numpy as np
from scipy import linalg
from synchronized_ellipse_fixed import fit_calibration, mapped_support


def point_metrics(points):
    radius = np.linalg.norm(points, axis=1)
    centered = points-points.mean(0)
    singular = np.linalg.svd(centered/np.sqrt(len(points)), compute_uv=False)
    reliable = radius > 1e-12
    angles = np.angle(points[reliable, 0]+1j*points[reliable, 1])
    sorted_angle = np.sort(np.mod(angles, 2*np.pi))
    arc95 = np.nan
    if len(sorted_angle):
        extended = np.r_[sorted_angle, sorted_angle+2*np.pi]
        count = max(1, int(np.ceil(.95*len(sorted_angle))))
        arc95 = np.min(extended[np.arange(len(sorted_angle))+count-1]-extended[:len(sorted_angle)])*180/np.pi
    return {'radial_rmse': float(np.sqrt(np.mean((radius-1)**2))),
            'radial_median': float(np.median(radius)), 'radial_q05': float(np.quantile(radius, .05)),
            'radial_q95': float(np.quantile(radius, .95)), 'mapped_mean': points.mean(0).tolist(),
            'mapped_centered_singular_values': singular.tolist(),
            'mapped_covariance_condition': float(singular[0]**2/singular[1]**2) if singular[1]>0 else None,
            'circular_variance': float(1-abs(np.mean(np.exp(1j*angles)))) if len(angles) else None,
            'angular_95percent_arc_deg': float(arc95),
            'chronological_unwrapped_q99_q01_span_rad': float(np.ptp(np.quantile(np.unwrap(angles), [.01,.99]))) if len(angles) else None,
            'undefined_phase_fraction': float(1-np.mean(reliable))}


def map_points(points, model):
    return ((points-np.array(model['origin']))/np.array(model['scale'])-np.array(model['center_normalized']))@np.array(model['transform'])


def enumerate_fit(points, name, first, last):
    points = np.asarray(points, dtype=float)
    origin, scale = points.mean(0), points.std(0)
    result = {'fit': name, 'first_s': first, 'last_s': last, 'points': len(points), 'origin': origin.tolist(),
              'scale': scale.tolist(), 'candidates': [], 'production_valid': False}
    if not np.isfinite(points).all() or np.any(scale<1e-12):
        result['production_error'] = 'Invalid/constant diagnostic input'; return result
    norm = (points-origin)/scale
    x,y = norm.T
    design = np.column_stack([x*x,x*y,y*y,x,y,np.ones(len(x))])
    scatter = design.T@design
    constraint = np.zeros((6,6)); constraint[0,2]=constraint[2,0]=2; constraint[1,1]=-1
    homogeneous, vectors = linalg.eig(scatter, constraint, homogeneous_eigvals=True)
    result['design_condition'] = float(np.linalg.cond(design))
    result['scatter_condition'] = float(np.linalg.cond(scatter))
    try:
        _,_,center,transform,meta = fit_calibration(points)
        result.update(production_valid=True, production_center_normalized=center.tolist(), production_transform=transform.tolist(), production_radial_rmse=meta['calibration_radial_rmse'])
    except ValueError as exc: result['production_error'] = str(exc)
    for index,(alpha,beta,v) in enumerate(zip(homogeneous[0],homogeneous[1],vectors.T)):
        candidate = {'candidate': index, 'homogeneous_alpha': [float(alpha.real),float(alpha.imag)],
                     'homogeneous_beta': [float(beta.real),float(beta.imag)], 'guard_failures': [],
                     'radial_rmse': None, 'center_native': None, 'axes_native': None,
                     'quadratic_condition': None, 'mapped_metrics': None, 'geometry_defined': False}
        result['candidates'].append(candidate)
        finite = np.isfinite([alpha,beta]).all() and abs(beta)>64*np.finfo(float).eps*(abs(alpha)+abs(beta))
        candidate['finite_generalized_eigenvalue'] = bool(finite)
        if not finite:
            candidate['guard_failures'].append('Nonfinite generalized eigenvalue'); continue
        value = alpha/beta
        candidate['eigenvalue'] = [float(value.real),float(value.imag)]
        candidate['eigenvector_real'] = v.real.tolist(); candidate['eigenvector_imag'] = v.imag.tolist()
        if not np.isfinite(v).all() or abs(value.imag)>1e-8*max(1,abs(value.real)) or np.max(abs(v.imag))>1e-8:
            candidate['guard_failures'].append('Complex/nonfinite eigenpair'); continue
        v = v.real
        qnorm = np.linalg.norm(v[:3])
        if qnorm<1e-10*np.linalg.norm(v):
            candidate['guard_failures'].append('Vanishing quadratic coefficients'); continue
        v = v/qnorm
        a,b,c = v[:3]
        candidate['discriminant_4ac_minus_b2'] = float(4*a*c-b*b)
        if 4*a*c-b*b<=1e-12: candidate['guard_failures'].append('Not an ellipse')
        backward = np.linalg.norm(scatter@v-value.real*(constraint@v))/max((np.linalg.norm(scatter)+abs(value)*np.linalg.norm(constraint))*np.linalg.norm(v),1e-12)
        candidate['eigenpair_backward_error'] = float(backward)
        if backward>1e-7: candidate['guard_failures'].append('Invalid eigenpair residual')
        quad = np.array([[a,b/2],[b/2,c]])
        if np.trace(quad)<0: quad,v = -quad,-v
        candidate['conic_coefficients_normalized'] = v.tolist()
        candidate['quadratic_condition'] = float(np.linalg.cond(quad))
        candidate['quadratic_eigenvalues'] = np.linalg.eigvalsh(quad).tolist()
        try:
            center = -.5*np.linalg.solve(quad,v[3:5])
            candidate['center_normalized'] = center.tolist()
            candidate['center_native'] = (origin+scale*center).tolist()
            level = center@quad@center-v[5]
            candidate['level'] = float(level)
            if np.linalg.eigvalsh(quad).min()<=0: candidate['guard_failures'].append('Nonpositive quadratic form')
            if level<=1e-12*max(abs(center@quad@center),abs(v[5]),1): candidate['guard_failures'].append('Degenerate ellipse level')
            if not np.isfinite(level) or level==0: raise ValueError('No finite canonical shape')
            canonical = quad/level
            values,basis = np.linalg.eigh(canonical)
            candidate['canonical_eigenvalue_signs'] = np.sign(values).tolist()
            candidate['canonical_axis_magnitudes_normalized'] = (1/np.sqrt(abs(values))).tolist()
            if values.min()<=0 or level<=0:
                candidate['geometry_note'] = 'Indefinite/imaginary conic: real ellipse axes, circle mapping, radial residual and angle coverage are undefined; null is not zero error'
                continue
            transform = (basis*np.sqrt(values))@basis.T
            native_shape = canonical/np.outer(scale,scale)
            native_values,native_basis = np.linalg.eigh(native_shape)
            candidate.update(geometry_defined=True, axes_normalized=(1/np.sqrt(values)).tolist(),
                             axes_native=(1/np.sqrt(native_values)).tolist(), native_shape_matrix=native_shape.tolist(),
                             native_axis_basis=native_basis.tolist(), native_shape_condition=float(np.linalg.cond(native_shape)),
                             transform=transform.tolist(), origin=origin.tolist(), scale=scale.tolist())
            candidate['mapped_metrics'] = point_metrics(map_points(points,candidate))
            candidate['radial_rmse'] = candidate['mapped_metrics']['radial_rmse']
            if np.sqrt(values.max()/values.min())>100: candidate['guard_failures'].append('Invalid ellipse axes')
            try: mapped_support(map_points(points,candidate))
            except ValueError as exc: candidate['guard_failures'].append(str(exc))
            if candidate['radial_rmse']>.2: candidate['guard_failures'].append('Radial residual exceeds original limit')
        except (ValueError,np.linalg.LinAlgError) as exc: candidate['geometry_note'] = str(exc)
        candidate['passes_unchanged_guards'] = not candidate['guard_failures']
        if result['production_valid'] and candidate.get('passes_unchanged_guards'):
            candidate['production_selected'] = bool(np.allclose(candidate['center_normalized'],result['production_center_normalized'],rtol=1e-8,atol=1e-8) and np.allclose(candidate['transform'],result['production_transform'],rtol=1e-8,atol=1e-8))
    passing = [c for c in result['candidates'] if c.get('passes_unchanged_guards')]
    assert bool(passing)==result['production_valid'], 'Diagnostic instrumentation changed acceptance'
    return result


def transfer_pair(early,late,points_early,points_late,pair_name):
    a = [c for c in early['candidates'] if c['geometry_defined']]
    b = [c for c in late['candidates'] if c['geometry_defined']]
    row = {'pair': pair_name, 'classification': 'inconclusive', 'reason': 'Needs unique proper finite ellipse at both times'}
    if len(a)!=1 or len(b)!=1: return row
    a,b = a[0],b[0]
    ea,eb = map_points(points_early,a),map_points(points_late,a)
    ba,bb = map_points(points_early,b),map_points(points_late,b)
    ma,mb,mab,mba = point_metrics(ea),point_metrics(bb),point_metrics(eb),point_metrics(ba)
    center_shift = np.linalg.norm((np.array(b['center_native'])-a['center_native'])/np.array(a['scale'])@np.array(a['transform']))
    shape_change = np.linalg.norm(np.log(linalg.eigvalsh(np.array(b['native_shape_matrix']),np.array(a['native_shape_matrix']))))
    medians=[]
    def bins(points):
        phase=np.mod(np.angle(points[:,0]+1j*points[:,1]),2*np.pi)
        return np.minimum((phase/(2*np.pi)*36).astype(int),35),np.linalg.norm(points,axis=1)
    ia,ra=bins(ea); ib,rb=bins(eb)
    for index in range(36):
        if np.sum(ia==index)>=10 and np.sum(ib==index)>=10:
            medians.append(abs(np.median(ra[ia==index])-np.median(rb[ib==index])))
    matched = float(np.median(medians)) if medians else None
    penalty = max(mab['radial_rmse']-mb['radial_rmse'],mba['radial_rmse']-ma['radial_rmse'])
    coverage = min(ma['angular_95percent_arc_deg'],mb['angular_95percent_arc_deg'])
    row.update(early_candidate=a['candidate'],late_candidate=b['candidate'],center_shift_in_early_radius_units=float(center_shift),
               log_shape_change=float(shape_change),matched_angular_bins=len(medians),matched_angle_radial_change=matched,
               early_self_rmse=ma['radial_rmse'],late_self_rmse=mb['radial_rmse'],early_to_late_rmse=mab['radial_rmse'],
               late_to_early_rmse=mba['radial_rmse'],max_transfer_penalty=float(penalty),minimum_arc95_deg=coverage,
               both_production_accepted=bool(early['production_valid'] and late['production_valid']))
    if len(medians)<6 or coverage<180:
        row['reason']='Insufficient shared angle/arc support; parameter change may reflect sampling/identifiability'
    elif matched>=.1 and penalty>=.1 and (center_shift>=.25 or shape_change>=.25) and early['production_valid'] and late['production_valid']:
        row.update(classification='time_dependent_mismatch_supported',reason='Matched-angle radius, geometric parameters and transfer penalty agree; not a causal attribution to hardware or motion')
    elif matched<=.05 and penalty<=.05 and center_shift<=.1 and shape_change<=.1:
        row.update(classification='stable_geometry_supported',reason='Stable geometry on supported arcs; high remaining fit residual does not by itself justify relaxing acceptance')
    else: row['reason']='Mixed geometry/transfer evidence under frozen descriptive thresholds'
    return row
