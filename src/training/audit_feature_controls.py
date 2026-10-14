"""Frozen-checkpoint signal-use controls; no training or label-based donor selection."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = ('original', 'reencoded', 'time_shuffle_time_only', 'freq_shuffle_freq_only',
              'time_shuffle_reencoded', 'circular_shift_time_only', 'circular_shift_reencoded',
              'replace_time_only', 'replace_freq_only', 'replace_both')


def frequency_from_time(x):
    """Exact Hann/log/row-zscore/global-zscore recipe on supplied time features.

    Exported time features have already been globally normalized. Original
    pre-normalization amplitudes cannot be recovered, so `reencoded` is the
    matched reference for every coherent recomputation control.
    """
    window = np.hanning(x.shape[-1]).astype(np.float32)
    f = np.log1p(abs(np.fft.rfft(x*window, axis=-1))).astype(np.float32)
    f -= f.mean(axis=-1, keepdims=True)
    s = f.std(axis=-1, keepdims=True)
    f /= np.where(s > 1e-6, s, 1.)
    f -= f.mean(axis=(-2,-1), keepdims=True)
    s = f.std(axis=(-2,-1), keepdims=True)
    return f / np.where(s > 1e-6, s, 1.)


def control_plan(groups, n_time, n_freq, seed):
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    if len(set(groups)) < 2:
        raise ValueError('Cross-subject replacement requires at least two subjects')
    donors = np.asarray([rng.choice(np.flatnonzero(groups != g)) for g in groups])
    return dict(donors=donors,
                time_order=np.stack([rng.permutation(n_time) for _ in groups]),
                freq_order=np.stack([rng.permutation(n_freq) for _ in groups]),
                shift=rng.integers(1, n_time, len(groups)))


def perturb(x_time, x_freq, plan, condition):
    t, f = x_time.copy(), x_freq.copy()
    if condition == 'original': return t, f
    if condition.startswith('time_shuffle'):
        t = np.take_along_axis(t, plan['time_order'][:,None,:], axis=-1)
    elif condition.startswith('freq_shuffle'):
        f = np.take_along_axis(f, plan['freq_order'][:,None,:], axis=-1)
    elif condition.startswith('circular_shift'):
        t = np.stack([np.roll(v, int(s), axis=-1) for v,s in zip(t,plan['shift'])])
    elif condition.startswith('replace'):
        if condition in ('replace_time_only','replace_both'): t = t[plan['donors']]
        if condition in ('replace_freq_only','replace_both'): f = f[plan['donors']]
    elif condition != 'reencoded':
        raise ValueError(condition)
    if condition.endswith('reencoded'): f = frequency_from_time(t)
    return t, f


def run(model_dir, export_dir, output, expected_predictions, seeds=(11,29,47), batch_size=64):
    import torch
    from src.models.factory import create_model
    from src.training.audit_predictions import load_verified, metrics, sha256
    from src.training.datasets import RadarWindowDataset, LabelStats

    started = time.perf_counter()
    torch.set_num_threads(2)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model_dir, export_dir = Path(model_dir), Path(export_dir)
    config = json.loads((model_dir/'run_config.json').read_text())
    if config['datasets'] != ['FTU']: raise ValueError('Expected FTU-only source model')
    manifest_hash = sha256(export_dir/'manifest.csv')
    expected_hash = config.get('provenance',{}).get('dataset_manifest_sha256')
    if not expected_hash or not manifest_hash.startswith(expected_hash):
        raise ValueError('Missing or mismatched training manifest provenance')
    ckpt = torch.load(model_dir/'best.pt', map_location='cpu', weights_only=False)
    model = create_model(config['model'], ckpt['config'])
    model.load_state_dict(ckpt['model_state'], strict=True); model.eval()
    stats = LabelStats(**ckpt['label_stats'])
    ds = RadarWindowDataset(export_dir, 'test', stats, {'FTU'})
    samples = [ds[i] for i in range(len(ds))]
    xt = np.stack([s['x_time'].numpy() for s in samples])
    xf = np.stack([s['x_freq'].numpy() for s in samples])
    if not np.isfinite(xt).all() or not np.isfinite(xf).all(): raise ValueError('Nonfinite features')
    records = pd.DataFrame(ds.rows)
    labels = records.label_heart_rate.astype(float).to_numpy()
    groups = records.group_key.to_numpy()
    manifest = pd.read_csv(export_dir/'manifest.csv')
    prior = load_verified(expected_predictions, manifest, 'FTU', 'test')
    from src.training.audit_predictions import KEYS
    records.window_start = records.window_start.astype(int)
    prior = records[KEYS].merge(prior, on=KEYS, validate='one_to_one')
    prediction_rows, summaries, timings = [], [], {}
    original = None
    for seed in seeds:
        plan = control_plan(groups, xt.shape[-1], xf.shape[-1], seed)
        references = {}
        for condition in CONDITIONS:
            tick = time.perf_counter()
            t, f = perturb(xt, xf, plan, condition)
            with torch.inference_mode():
                pred = np.concatenate([model(torch.from_numpy(t[i:i+batch_size]),
                    torch.from_numpy(f[i:i+batch_size])).cpu().numpy().reshape(-1)
                    for i in range(0,len(t),batch_size)])*stats.std+stats.mean
            if not np.isfinite(pred).all(): raise ValueError('Nonfinite model prediction')
            if condition == 'original':
                if not np.allclose(pred, prior.pred_bpm, atol=1e-3, rtol=0):
                    raise ValueError(f'Historical predictions mismatch: max={max(abs(pred-prior.pred_bpm))}')
                if original is not None and not np.array_equal(pred, original):
                    raise ValueError('Frozen inference is not deterministic')
                original = pred.copy()
            references[condition] = pred
            reference = 'reencoded' if condition.endswith('_reencoded') else 'original'
            baseline = references[reference]
            frame = pd.DataFrame({'label_bpm':labels, 'pred_bpm':pred})
            score = metrics(frame, {'train_mean':stats.mean})
            delta = abs(pred-labels)-abs(baseline-labels)
            for group in ['ALL',*sorted(set(groups))]:
                mask = np.ones(len(groups),dtype=bool) if group == 'ALL' else groups == group
                summaries.append(dict(seed=seed,condition=condition,reference=reference,group=group,
                    **metrics(frame.loc[mask], {'train_mean':stats.mean}),
                    delta_mae=float(delta[mask].mean()),
                    mean_abs_prediction_change=float(abs(pred[mask]-baseline[mask]).mean())))
            for i,row in records.iterrows():
                prediction_rows.append(dict(seed=seed, condition=condition, reference=reference,
                    **{k:row[k] for k in KEYS}, label_bpm=labels[i], pred_bpm=float(pred[i]),
                    reference_pred_bpm=float(baseline[i]), delta_abs_error=float(delta[i]),
                    donor_index=int(plan['donors'][i]) if condition.startswith('replace') else -1,
                    donor_group=groups[plan['donors'][i]] if condition.startswith('replace') else ''))
            timings[f'{seed}/{condition}'] = time.perf_counter()-tick
            print(seed,condition,'MAE',score['mae'],'delta',float(delta.mean()),flush=True)
    pd.DataFrame(prediction_rows).to_csv(output/'predictions.csv',index=False)
    pd.DataFrame(summaries).to_csv(output/'controls.csv',index=False)
    metadata = dict(checkpoint_sha256=sha256(model_dir/'best.pt'),manifest_sha256=manifest_hash,
        run_config=config, expected_predictions_sha256=sha256(expected_predictions),
        test_n=len(ds), seeds=list(seeds), device='cpu', torch_version=torch.__version__,
        elapsed_sec=time.perf_counter()-started, timings=timings,
        validation={'historical_predictions_reproduced_atol_bpm':1e-3,'strict_state_dict':True,
                    'frozen_deterministic':True,'cross_subject_donors_only':True},
        caveats=['Single-branch changes break time/frequency consistency.',
                 'Reencoded controls recompute Hann/log spectrum from normalized x_time; compare to reencoded.',
                 'Circular shift is an invariance check, not a guaranteed HR-label disruption.',
                 'Random donors use other test subjects with replacement, never labels.',
                 'Perturbations may be out of distribution; sensitivity is not causal proof.'])
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','export-dir','output','expected-predictions'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    run(**vars(args))
