"""CPU-only collapse audit of frozen predictions, with manifest-verified pairing."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ['dataset', 'group_key', 'sample_tag', 'window_start']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correlation(y, p):
    if len(y) < 3 or np.std(y) < 1e-8 or np.std(p) < 1e-8:
        return None
    return float(np.corrcoef(y, p)[0, 1])


def metrics(frame, constants):
    y, p = frame.label_bpm.to_numpy(), frame.pred_bpm.to_numpy()
    sy, sp = float(np.std(y)), float(np.std(p))
    mae = float(np.mean(abs(p-y)))
    result = dict(n=len(y), label_mean=float(np.mean(y)), pred_mean=float(np.mean(p)),
                  label_std=sy, pred_std=sp, r_var=sp/sy if sy > 1e-8 else None,
                  pearson_r=correlation(y, p), mae=mae,
                  rmse=float(np.sqrt(np.mean((p-y)**2))))
    for name, value in constants.items():
        base = float(np.mean(abs(y-value)))
        result[name+'_mae'] = base
        result['gain_vs_'+name] = base-mae
    return result


def load_verified(predictions, manifest, dataset, split):
    frame = pd.read_csv(predictions, dtype={'group_key':str, 'sample_tag':str})
    frame = frame.loc[frame.dataset == dataset].copy()
    if frame.empty or not np.isfinite(frame[['label_bpm', 'pred_bpm']]).all().all():
        raise ValueError('Empty or nonfinite predictions')
    selected = manifest.loc[(manifest.dataset == dataset) & (manifest.split == split)].copy()
    for table in (frame, selected):
        table.window_start = pd.to_numeric(table.window_start, errors='raise')
        if table.duplicated(KEYS).any():
            raise ValueError('Duplicate window identities')
    joined = frame.merge(selected[KEYS+['label_heart_rate']], on=KEYS, how='outer',
                         indicator=True, validate='one_to_one')
    if not (joined['_merge'] == 'both').all():
        raise ValueError('Prediction windows differ from the declared test split')
    if not np.allclose(joined.label_bpm, joined.label_heart_rate, atol=1e-4, rtol=0):
        raise ValueError('Prediction labels differ from manifest labels')
    return frame


def audit(predictions, export_dir, run_config, output, dataset='FTU', split='test',
          sampling_rate=20., expected_mae=None):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(export_dir)/'manifest.csv'
    manifest = pd.read_csv(manifest_path)
    config = json.loads(Path(run_config).read_text(encoding='utf-8'))
    expected_hash = config.get('provenance', {}).get('dataset_manifest_sha256')
    actual_hash = sha256(manifest_path)
    if expected_hash and not actual_hash.startswith(expected_hash):
        raise ValueError('Manifest hash differs from checkpoint run provenance')
    frame = load_verified(predictions, manifest, dataset, split)
    build_config = json.loads((Path(export_dir)/'build_config.json').read_text(encoding='utf-8'))
    window_size = int(build_config['window_size'])
    if not np.isclose(sampling_rate, build_config['sampling_rate_hz']):
        raise ValueError('Plot sampling rate differs from export configuration')
    train = manifest.loc[(manifest.dataset == dataset) & (manifest.split == 'train')]
    if train.empty:
        raise ValueError('Training labels required for a non-oracle constant baseline')
    train_groups = set(train.group_key)
    if train_groups & set(frame.group_key):
        raise ValueError('Train/test group leakage')
    mean = float(np.mean(train.label_heart_rate.to_numpy(dtype=np.float32)))
    if config.get('datasets') != [dataset]:
        raise ValueError('This audit requires a single-source checkpoint')
    if not np.isclose(mean, config['label_stats']['mean'], atol=1e-4, rtol=0):
        raise ValueError('Training mean differs from checkpoint label statistics')
    constants = {'train_mean':mean, 'train_median':float(train.label_heart_rate.median())}
    overall = metrics(frame, constants)
    if expected_mae is not None and abs(overall['mae']-expected_mae) > 1e-4:
        raise ValueError('Original predictions do not reproduce declared MAE')
    tables = {}
    for name, keys in [('subject',['group_key']), ('record',['group_key','sample_tag'])]:
        rows = []
        for key, group in frame.groupby(keys, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            rows.append(dict(zip(keys,key)) | metrics(group, constants))
        tables[name] = pd.DataFrame(rows)
        tables[name].to_csv(output/f'by_{name}.csv', index=False)
    centered = frame.copy()
    for col in ('label_bpm','pred_bpm'):
        centered[col] -= centered.groupby('group_key')[col].transform('mean')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    axes[0].scatter(frame.label_bpm, frame.pred_bpm, s=9, alpha=.4)
    lo, hi = float(frame.label_bpm.min()), float(frame.label_bpm.max())
    axes[0].plot([lo,hi],[lo,hi], '--', color='gray')
    axes[0].set(xlabel='Reference HR (bpm)', ylabel='Predicted HR (bpm)')
    axes[1].hist(frame.pred_bpm-frame.label_bpm, bins=25, alpha=.65, label='model')
    axes[1].hist(mean-frame.label_bpm, bins=25, alpha=.45, label='train mean')
    axes[1].set(xlabel='Signed error (bpm)', ylabel='Windows')
    axes[1].legend()
    fig.tight_layout(); fig.savefig(output/'errors.png', dpi=160); plt.close(fig)
    # Separate axes for each continuous record; never concatenate sessions as time.
    for subject, data in frame.groupby('group_key'):
        records = list(data.groupby('sample_tag', sort=True))
        fig, axes = plt.subplots(int(np.ceil(len(records)/4)), 4,
                                 figsize=(16, 2.2*int(np.ceil(len(records)/4))), squeeze=False)
        for ax, (tag, group) in zip(axes.flat, records):
            group = group.sort_values('window_start')
            t = (group.window_start.to_numpy()+window_size/2)/sampling_rate
            ax.plot(t, group.label_bpm, label='reference')
            ax.plot(t, group.pred_bpm, label='prediction')
            ax.axhline(mean, color='gray', ls='--', label='train mean')
            ax.set_title(tag, fontsize=7)
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[len(records):]: ax.set_visible(False)
        axes.flat[0].legend(fontsize=6)
        fig.supxlabel('Window center (seconds from record start)'); fig.supylabel('HR (bpm)')
        fig.tight_layout(); fig.savefig(output/('trajectories_'+subject.replace('/','_')+'.png'), dpi=140)
        plt.close(fig)
    selected = tables['record']
    examples = { 'best_mae': selected.loc[selected.mae.idxmin(),'sample_tag'],
                 'worst_mae': selected.loc[selected.mae.idxmax(),'sample_tag'],
                 'largest_hr_variation': selected.loc[selected.label_std.idxmax(),'sample_tag'] }
    result = dict(overall=overall, centered_within_subject=metrics(centered, {}),
                  constants=constants, examples_for_continuous_signal_audit=examples,
                  provenance={str(p):sha256(p) for p in (predictions,manifest_path,run_config)},
                  validation={'unique_windows':True,'labels_match_manifest':True,
                              'train_test_groups_disjoint':True, 'train_mean_matches_checkpoint':True,
                              'manifest_hash_matches_run':bool(expected_hash)},
                  elapsed_sec=time.perf_counter()-started,
                  limitations=['Two-subject test results do not establish population generalization.',
                               'Undefined correlation or variance ratio is reported as null.',
                               'Record selection uses test errors for explanation, not model selection.'])
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('predictions','export-dir','run-config','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--dataset', default='FTU')
    parser.add_argument('--split', default='test', choices=['test','val'])
    parser.add_argument('--sampling-rate', type=float, default=20.)
    parser.add_argument('--expected-mae', type=float)
    args = parser.parse_args()
    print(json.dumps(audit(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
