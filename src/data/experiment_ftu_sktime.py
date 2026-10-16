"""Unmodified sktime 1.2.0 VMD baseline: local signals or window-local FTU RDA.

Run with a temporary dependency: uv run --locked --no-sync --with sktime==1.2.0
python -m src.data.experiment_ftu_sktime. No neural-network training occurs.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import sktime
from sktime.libs.vmdpy import VMD

from src.features.edacm import zscore_1d, target_edacm_signal
from src.features.hr_adavmd import hr_adavmd_decompose, hr_adavmd_mode_scores
from src.data.audit_ftu_p0 import digest, write_json

PARAMS=dict(alpha=2000.,tau=0.,K=7,DC=False,init=1,tol=1e-7)


def metrics(modes):
    x=np.asarray(modes,dtype=np.float64)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite mode output')
    z=x-x.mean(axis=1,keepdims=True)
    norm=np.linalg.norm(z,axis=1); sv=np.linalg.svd(z,compute_uv=False)
    corr=z@z.T/np.maximum(norm[:,None]*norm[None,:],1e-30)
    pair=np.triu_indices(len(x),1)
    power=abs(np.fft.rfft(z*np.hanning(x.shape[1]),axis=1))**2
    f=np.fft.rfftfreq(x.shape[1],1/20.)
    return dict(numerical_rank=int(np.sum(sv>max(sv[0]*1e-6,1e-12))),
        rank1_fraction=float(sv[0]**2/max(float((sv**2).sum()),1e-30)),
        nonzero_modes=int(np.count_nonzero(norm>1e-6)),
        duplicate_pairs=sum(np.array_equal(x[i],x[j]) for i,j in zip(*pair)),
        max_abs_correlation=float(np.max(abs(corr[pair]))) if len(pair[0]) else 0.,
        mode_std=x.std(axis=1).tolist(),
        mode_peak_hz=f[power.argmax(axis=1)].tolist(),
        mode_centroid_hz=((power*f).sum(axis=1)/np.maximum(power.sum(axis=1),1e-30)).tolist())


def official(signal,k=7):
    x=zscore_1d(signal).astype(np.float64)
    start=time.perf_counter()
    raw,_,omega=VMD(x,**dict(PARAMS,K=k))
    assert raw.shape==(k,len(x)), (raw.shape,len(x))
    normalized=np.stack([zscore_1d(m) for m in raw])
    weights,_,meta=hr_adavmd_mode_scores(normalized)
    weighted=normalized*weights[:,None]
    info=dict(elapsed_sec=time.perf_counter()-start,**metrics(raw),
        reconstruction_relative_error=float(np.linalg.norm(raw.sum(axis=0)-x)/max(np.linalg.norm(x),1e-30)),
        omega_history_rows=len(omega),at_iteration_cap=bool(len(omega)>=499),
        final_centers_hz=(omega[-1]*20).tolist(),mode_weights=weights.tolist(),
        weighted_rank=metrics(weighted)['numerical_rank'])
    return raw,weighted,info


def provenance():
    assert sktime.__version__=='1.2.0', sktime.__version__
    return dict(sktime_version=sktime.__version__,numpy_version=np.__version__,
        official_source_sha256=digest(inspect.getfile(VMD)),parameters=PARAMS,
        source_sha256=digest(__file__),implementation='unmodified sktime.libs.vmdpy.VMD',
        note='Uniform initialization. HR-band scoring is applied AFTER official VMD; not the legacy physiology-initialized solver.',
        cap_note='499 omega rows flags the iteration cap; it does not prove the final update failed the tolerance check.',
        training_launched=False)


def local(out):
    start=time.perf_counter(); inputs={}; cases={}; rows=[]; arrays={}
    root=Path('experiment_runs/ftu_source_audit_2026-10-13/e2a_continuous')
    for p in sorted(root.glob('*signals.npz')):
        inputs[str(p)]=digest(p)
        with np.load(p) as z:
            name=p.stem.removesuffix('_signals'); signal=z['edacm_fused'].copy()
            cases[name]=signal
            for s in z['window_starts']:
                cases[name+'__excerpt_'+str(s)]=signal[int(s):int(s)+256]
    t=np.arange(1200)/20
    cases.update(synthetic_single=np.sin(2*np.pi*1.2*t),
        synthetic_three=2*np.sin(2*np.pi*.25*t)+np.sin(2*np.pi*1.2*t)+.4*np.sin(2*np.pi*2.1*t),
        synthetic_noise=np.random.default_rng(47).normal(size=1200),
        synthetic_zero=np.zeros(256),synthetic_constant=np.full(256,3.),
        synthetic_odd=np.sin(2*np.pi*1.2*np.arange(257)/20))
    for name,x in cases.items():
        for variant in ('legacy_hr_adavmd','sktime_vmd'):
            if variant=='sktime_vmd': raw,weighted,info=official(x)
            else:
                weighted,meta=hr_adavmd_decompose(x)
                raw=None; info=dict(**metrics(weighted),mode_weights=meta['mode_weights'])
            rows.append(dict(case=name,variant=variant,**info))
            arrays[name+'__'+variant]=weighted
            if raw is not None: arrays[name+'__raw']=raw
        print(name,flush=True)
    # Correctly specified component count tests identifiable tone recovery.
    for name,k,expected in [('synthetic_single',1,[1.2]),('synthetic_three',3,[.25,1.2,2.1])]:
        raw,weighted,info=official(cases[name],k)
        peaks=np.sort(info['mode_peak_hz'])
        assert np.allclose(peaks,expected,atol=20/len(cases[name]))
        rows.append(dict(case=name,variant=f'sktime_K{k}',**info))
        arrays[name+f'__raw_K{k}']=raw
    pd.DataFrame(rows).to_csv(out/'local_comparison.csv',index=False)
    write_json(out/'local_diagnostics.json',rows)
    np.savez_compressed(out/'local_modes.npz',**arrays)
    write_json(out/'local_summary.json',dict(**provenance(),elapsed_sec=time.perf_counter()-start,
        inputs=inputs,signals=len(cases),scope='3 full records + 24 excerpts of full-record EDACM + 6 synthetic controls; excerpts do not reproduce window-local target selection.'))


def kaggle(out):
    start=time.perf_counter(); root=Path('/kaggle/input')
    manifests=list(root.rglob('training_exports/manifest.csv'));assert len(manifests)==1
    manifest=pd.read_csv(manifests[0]); rows=[]; phases=[]; raw_modes=[]; tags=[]
    saved_legacy=[]; rebuilt_legacy=[]
    # One predeclared first window per record, independent of HR or prediction quality.
    selected=manifest.sort_values('window_start').groupby('sample_tag',sort=True).first().reset_index()
    rda={p.stem:p for p in root.rglob('*.npz') if 'ftu-rda-exports' in str(p)}
    for i,row in selected.iterrows():
        path=rda[row.sample_tag]
        with np.load(path,allow_pickle=False) as z:
            radar=z['radar'][int(row.window_start):int(row.window_end)]
        phase,selection=target_edacm_signal(radar)
        raw,weighted,info=official(phase)
        legacy,lmeta=hr_adavmd_decompose(phase)
        export=manifests[0].parent/'windows'/row.dataset/row.split/Path(row.window_path).name
        with np.load(export,allow_pickle=False) as z:
            expected=zscore_1d(legacy)
            err=float(np.max(abs(expected-z['x_time'])))
            original_meta=json.loads(str(z['meta_json'].item()))['feature_meta']
            saved_legacy.append(z['x_time'][0].copy()); rebuilt_legacy.append(expected[0])
        # A diverging legacy solver is sensitive to numerical/library differences.
        # Keep the discrepancy visible; compare BOTH algorithms on the same rebuilt
        # EDACM signal, never claim exact reproduction of historical model features.
        rows.append(dict(sample_tag=row.sample_tag,split=row.split,window_start=int(row.window_start),
            legacy_reproduction_max_error=err,legacy_duplicate_pairs=metrics(legacy)['duplicate_pairs'],
            legacy_matches_at_1e4=bool(err<=1e-4),rda_sha256=digest(path),
            target_bins_match=selection['target_bins_dar']==original_meta['target_bins_dar'],
            selection=selection,**info))
        phases.append(phase);raw_modes.append(raw);tags.append(row.sample_tag)
        if i%50==0:print('records',i,flush=True)
    pd.DataFrame(rows).to_csv(out/'window_comparison.csv',index=False)
    write_json(out/'window_diagnostics.json',rows)
    np.savez_compressed(out/'window_signals.npz',sample_tags=np.asarray(tags),
                        phase=np.stack(phases),raw_modes=np.stack(raw_modes),
                        saved_legacy_channel0=np.stack(saved_legacy),rebuilt_legacy_channel0=np.stack(rebuilt_legacy))
    write_json(out/'remote_summary.json',dict(**provenance(),elapsed_sec=time.perf_counter()-start,
        manifest_sha256=digest(manifests[0]),records=len(rows),selection='first manifest window of every record'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--kaggle',action='store_true')
    p.add_argument('--output',type=Path,default=Path('experiment_runs/ftu_sktime_baseline_2026-10-16'))
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    (kaggle if a.kaggle else local)(a.output)


if __name__=='__main__': main()
