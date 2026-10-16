"""Read-only P0 audit of archived FTU modes and time mapping; never trains models.

Local: python -m src.data.audit_ftu_p0
Kaggle: python -m src.data.audit_ftu_p0 --kaggle --output /kaggle/working/ftu_p0
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import hr_adavmd as vmd
from src.features.edacm import zscore_1d


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def mode_metrics(modes, fs=20.):
    x = np.asarray(modes, dtype=np.float64)
    assert x.ndim == 2 and x.shape[0] == 7 and np.isfinite(x).all()
    centered = x-x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    corr = centered @ centered.T / np.maximum(norms[:, None]*norms[None, :], 1e-30)
    singular = np.linalg.svd(centered, compute_uv=False)
    power = abs(np.fft.rfft(centered*np.hanning(x.shape[-1]), axis=-1))**2
    f = np.fft.rfftfreq(x.shape[-1], 1/fs)
    center = (power*f).sum(axis=1)/np.maximum(power.sum(axis=1), 1e-30)
    peak = f[power.argmax(axis=1)]
    pairs = np.triu_indices(7, 1)
    summary = dict(min_pair_correlation=float(corr[pairs].min()),
                   max_pair_correlation=float(corr[pairs].max()),
                   max_array_difference=float(np.max(abs(x-x[0]))),
                   rank1_energy_fraction=float(singular[0]**2/max(float((singular**2).sum()), 1e-30)),
                   numerical_rank=int(np.sum(singular > max(singular[0]*1e-6, 1e-12))),
                   nonzero_modes=int(np.count_nonzero(norms > 1e-6)),
                   all_arrays_exactly_equal=bool(np.array_equal(x, np.broadcast_to(x[0], x.shape))))
    rows = [dict(mode=i, std=float(x[i].std()), spectral_centroid_hz=float(center[i]),
                 peak_hz=float(peak[i])) for i in range(7)]
    return summary, rows, corr, power, f


def instrumented_vmd(signal, *, sequential_sum=False):
    """Instrument the actual implementation, with an optional ONE-LINE diagnostic change.

    The changed sum is an isolated numerical experiment, not a validated VMD replacement.
    """
    source = inspect.getsource(vmd.vmd_decompose)
    source = source.replace('    for _ in range(max_iter):', '    trace = []\n    for _ in range(max_iter):\n        clips = 0\n        peak_update = 0.0')
    source = source.replace('            update = np.clip(update.real',
        '            clips += int(np.count_nonzero((abs(update.real)>1e6) | (abs(update.imag)>1e6)))\n'
        '            peak_update = max(peak_update, float(np.max(abs(update))))\n'
        '            update = np.clip(update.real')
    if sequential_sum:
        source = source.replace('            u_hat[mode_idx] = update.astype(np.complex64)',
            '            sum_modes += update.astype(np.complex64) - u_hat[mode_idx]\n'
            '            u_hat[mode_idx] = update.astype(np.complex64)')
    source = source.replace('        if not np.isfinite(diff):',
        '        trace.append(dict(iteration=_+1, clip_count=clips, peak_update=peak_update, '
        'relative_update=float(diff), spectral_norm=float(np.linalg.norm(u_hat)), '
        'centers_hz=(omega*20).tolist()))\n        if not np.isfinite(diff):')
    source = source.replace('    return np.stack([zscore_1d(mode) for mode in modes], axis=0).astype(np.float32)',
        '    return np.stack([zscore_1d(mode) for mode in modes], axis=0).astype(np.float32), trace, modes')
    ns = dict(vars(vmd))
    exec(compile(source, '<instrumented archived VMD>', 'exec'), ns)
    return ns['vmd_decompose'](signal, init_omega=vmd.hr_adavmd_initial_omega(7))


def historical_frame_time(t, n):
    # Exact E2a mapping for the finite, nonempty reference arrays audited here.
    t = np.asarray(t, dtype=np.float32)
    return t if len(t) == n else np.linspace(t.min(), t.max(), n, dtype=np.float32)


def timeline_case(z, fs=20.):
    t = np.asarray(z['reference_time']).reshape(-1)
    y = np.asarray(z['reference_hr']).reshape(-1)
    n = z['edacm_fused'].size
    historical = historical_frame_time(t, n)
    # This is a sensitivity calculation, NOT a recovered radar/HR start offset.
    nominal = float(t[0])+np.arange(n)/fs
    old = np.interp(historical, t, y, left=np.nan, right=np.nan)
    new = np.interp(nominal, t, y, left=np.nan, right=np.nan)
    rows = []
    for i, s in enumerate(z['window_starts']):
        s = int(s)
        rows.append(dict(window_start=s, historical_label=float(old[s:s+256].mean()),
            saved_label=float(z['reference_window_hr'][i]),
            nominal_same_start_label=float(np.nanmean(new[s:s+256])),
            nominal_coverage=float(np.isfinite(new[s:s+256]).mean()),
            label_delta_bpm=float(np.nanmean(new[s:s+256])-old[s:s+256].mean())))
    return dict(n_frames=n, reference_n=len(t), reference_start=float(t[0]),
        reference_end=float(t[-1]), reference_span_s=float(t[-1]-t[0]),
        nonpositive_reference_steps=int(np.count_nonzero(np.diff(t)<=0)),
        min_reference_step_s=float(np.diff(t).min()), max_reference_step_s=float(np.diff(t).max()),
        nominal_frame_span_s=(n-1)/fs, implied_label_frame_rate_hz=(n-1)/float(t[-1]-t[0]),
        max_same_start_time_difference_s=float(abs(historical-nominal).max()),
        saved_label_reproduction_max_bpm=max(abs(r['historical_label']-r['saved_label']) for r in rows),
        clock_synchronization_verified=False), rows


def local_audit(archive, out):
    start = time.perf_counter()
    tree = ast.parse((archive/'e2a_evidence/source_notebook.py').read_text(encoding='utf-8'))
    files = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'FILES' for t in n.targets))
    for rel in ('src/features/hr_adavmd.py', 'src/features/edacm.py'):
        assert files[rel] == Path(rel).read_text(encoding='utf-8'), f'Archive differs: {rel}'
    inputs = {}; summaries = []; mode_rows = []; pair_rows = []; traces = []; timelines = []; label_rows = []
    series = {}
    for path in sorted((archive/'e2a_continuous').glob('*_signals.npz')):
        category = path.stem.removesuffix('_signals')
        inputs[str(path)] = digest(path)
        with np.load(path) as z:
            series[category] = z['edacm_fused'].copy()
            m, meta = vmd.hr_adavmd_decompose(z['edacm_fused'])
            assert np.allclose(m, z['vmd_modes'], atol=1e-6, rtol=1e-6)
            timing, rows = timeline_case(z)
            timelines.append(dict(case=category, **timing))
            label_rows.extend(dict(case=category, **r) for r in rows)
    t = np.arange(1200)/20
    series['synthetic_three_tone'] = (2*np.sin(2*np.pi*.25*t)+np.sin(2*np.pi*1.2*t)+.4*np.sin(2*np.pi*2.1*t)).astype(np.float32)
    arrays = {}
    for name, signal in series.items():
        for variant in ('historical', 'sequential_sum_diagnostic'):
            modes, trace, raw = instrumented_vmd(signal, sequential_sum=variant != 'historical')
            if variant == 'historical':
                expected = vmd.vmd_decompose(signal, init_omega=vmd.hr_adavmd_initial_omega(7))
                assert np.array_equal(expected, modes), 'Instrumentation changed output'
            weights, scores, meta = vmd.hr_adavmd_mode_scores(modes)
            weighted = modes*weights[:, None]
            metrics, rows, corr, power, freq = mode_metrics(weighted)
            summaries.append(dict(case=name, variant=variant, **metrics,
                total_clipped_coefficients=sum(r['clip_count'] for r in trace),
                first_clip_iteration=next((r['iteration'] for r in trace if r['clip_count']), None),
                iterations=len(trace), final_relative_update=trace[-1]['relative_update'],
                raw_reconstruction_relative_error=float(np.linalg.norm(raw.sum(axis=0)-zscore_1d(signal))/np.linalg.norm(zscore_1d(signal)))))
            mode_rows.extend(dict(case=name, variant=variant, **r, weight=float(weights[i]),
                normalized_score=float(scores[i]), **meta['mode_diagnostics'][i]) for i, r in enumerate(rows))
            pair_rows.extend(dict(case=name, variant=variant, mode_a=i, mode_b=j, correlation=float(corr[i,j]))
                for i in range(7) for j in range(i+1,7))
            traces.extend(dict(case=name, variant=variant, **r) for r in trace)
            arrays[name+'__'+variant] = weighted
    for name, rows in [('mode_summary', summaries), ('modes', mode_rows), ('mode_pairs', pair_rows),
                       ('iterations', traces), ('timeline', timelines), ('label_sensitivity', label_rows)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'), index=False)
    np.savez_compressed(out/'mode_arrays.npz', **arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for axrow, name in zip(axes, list(series)[:3]):
        m = arrays[name+'__historical']
        _, _, corr, power, freq = mode_metrics(m)
        axrow[0].plot(np.arange(200)/20, m[:, :200].T)
        axrow[0].set(title=name, xlabel='Time (s)', ylabel='Weighted modes (overlap)')
        axrow[1].imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
        axrow[1].set(title='Pairwise correlation', xlabel='Mode', ylabel='Mode')
        axrow[2].plot(freq, (power/np.maximum(power.sum(axis=1, keepdims=True), 1e-30)).T)
        axrow[2].set(xlim=(0, 3), title='Normalized power (overlap)', xlabel='Hz')
    fig.tight_layout(); fig.savefig(out/'mode_independence.png', dpi=150); plt.close(fig)
    write_json(out/'local_summary.json', dict(elapsed_sec=time.perf_counter()-start, inputs=inputs,
        archived_feature_source_matches=True, instrumentation_exact_reproduction=True,
        mode_gate='FAIL', timing_gate='UNVERIFIED', retraining_allowed=False,
        scope='Three full records and synthetic signal; exported training windows audited separately.',
        diagnostic_change='Update sum_modes after each mode. Not a production fix or validated reference VMD.',
        source_sha256=digest(__file__)))


def kaggle_audit(out):
    start = time.perf_counter(); root = Path('/kaggle/input')
    manifests = list(root.rglob('training_exports/manifest.csv'))
    assert len(manifests) == 1, manifests
    manifest = pd.read_csv(manifests[0]); export = manifests[0].parent
    rows = []; examples = {}; meta_examples = []
    for idx, row in manifest.iterrows():
        path = export/'windows'/row.dataset/row.split/Path(row.window_path).name
        with np.load(path, allow_pickle=False) as z:
            m = z['x_time']; metrics, modes, corr, _, _ = mode_metrics(m)
            meta = json.loads(str(z['meta_json'].item()))
            weights = meta['feature_meta']['mode_weights']
            rows.append(dict(sample_tag=row.sample_tag, split=row.split, window_start=int(row.window_start),
                sha256=digest(path), **metrics, min_weight=min(weights), max_weight=max(weights),
                frequency_arrays_identical=bool(np.array_equal(z['x_freq'], np.broadcast_to(z['x_freq'][0], z['x_freq'].shape)))))
            if row.sample_tag in ('p01_Angle_30_deg__80_cm_r2', 'p08_Distance_40_cm_r3', 'p08_Orientation_80_cm__left_side_r3'):
                key=row.sample_tag+'__'+str(row.window_start)
                examples[key+'__x_time']=m.copy(); examples[key+'__x_freq']=z['x_freq'].copy()
                meta_examples.append(dict(key=key, source_sha256=digest(path), feature_meta=meta['feature_meta'], modes=modes))
        if idx % 500 == 0: print('windows', idx, flush=True)
    pd.DataFrame(rows).to_csv(out/'exported_window_modes.csv', index=False)
    np.savez_compressed(out/'exported_examples.npz', **examples)
    write_json(out/'exported_example_metadata.json', meta_examples)
    logs = []; references = []
    evidence = out/'raw_time_evidence'; evidence.mkdir(exist_ok=True)
    # Only tiny sidecars; never read the hundreds of GB of ADC data.
    for path in sorted(root.rglob('*')):
        if not path.is_file(): continue
        if path.name not in ('data_LogFile.txt', 'data_Raw_LogFile.csv') and not (path.suffix.lower()=='.csv' and '_HR_refs' in str(path)):
            continue
        text = path.read_text(encoding='utf-8-sig', errors='replace')
        key = hashlib.sha256(str(path).encode()).hexdigest()[:12]+'_'+path.name
        (evidence/key).write_bytes(path.read_bytes())
        info = dict(source=str(path), archived_file=key, sha256=digest(path))
        if path.name == 'data_LogFile.txt':
            lines=text.splitlines()
            configs=[s for s in lines if 'API:FrameConfig,' in s]
            starts=[s for s in lines if 'API:SensorStart,' in s]
            logs.append(dict(**info, kind='radar_config_log', frame_configs=configs, sensor_starts=starts))
        elif path.name == 'data_Raw_LogFile.csv':
            logs.append(dict(**info, kind='capture_log', text=text))
        else:
            try:
                df=pd.read_csv(path, skiprows=3, header=None)
                t=pd.to_timedelta(df.iloc[:,1]).dt.total_seconds().to_numpy()
                y=pd.to_numeric(df.iloc[:,2]).to_numpy()
                references.append(dict(**info, n=len(t), start_s=float(t[0]), end_s=float(t[-1]),
                    span_s=float(t[-1]-t[0]), min_step_s=float(np.diff(t).min()), max_step_s=float(np.diff(t).max()),
                    nonpositive_steps=int(np.count_nonzero(np.diff(t)<=0)), nonfinite_labels=int(np.count_nonzero(~np.isfinite(y))),
                    first_three_lines=text.splitlines()[:3]))
            except Exception as exc:
                references.append(dict(**info, error=repr(exc)))
    write_json(out/'radar_logs.json', logs); write_json(out/'reference_times.json', references)
    write_json(out/'remote_summary.json', dict(elapsed_sec=time.perf_counter()-start,
        manifest_sha256=digest(manifests[0]), windows=len(rows), raw_sidecars=len(logs)+len(references),
        exactly_duplicate_windows=sum(r['all_arrays_exactly_equal'] for r in rows),
        rank1_windows=sum(r['numerical_rank']==1 for r in rows),
        minimum_pair_correlation=min(r['min_pair_correlation'] for r in rows),
        weights_all_one_windows=sum(r['min_weight']==r['max_weight']==1 for r in rows),
        retraining_allowed=False, source_sha256=digest(__file__)))
    print('COMPLETE', flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--kaggle', action='store_true')
    p.add_argument('--archive', type=Path, default=Path('experiment_runs/ftu_source_audit_2026-10-13'))
    p.add_argument('--output', type=Path, default=Path('experiment_runs/ftu_p0_audit_2026-10-14'))
    args=p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.kaggle: kaggle_audit(args.output)
    else: local_audit(args.archive, args.output)


if __name__ == '__main__': main()
