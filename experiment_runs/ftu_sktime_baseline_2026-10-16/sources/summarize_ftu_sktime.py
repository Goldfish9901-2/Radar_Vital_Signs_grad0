"""Validate and summarize the unmodified package baseline and timing controls."""
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from scipy.signal import detrend
from src.data.audit_ftu_p0 import digest, write_json
from src.features.edacm import zscore_1d
from src.features.hr_adavmd import hr_adavmd_mode_scores


def main():
    started=time.perf_counter()
    out=Path('experiment_runs/ftu_sktime_baseline_2026-10-16')
    remote=Path('tmp/ftu_sktime_result/ftu_sktime')
    manifest_path=Path('experiment_runs/ftu_source_audit_2026-10-13/e2a_evidence/training_exports/manifest.csv')
    manifest=pd.read_csv(manifest_path)
    meta=json.loads((remote/'remote_summary.json').read_text())
    assert meta['manifest_sha256']==digest(manifest_path)
    selected=manifest.sort_values('window_start').groupby('sample_tag',sort=True).first().reset_index()
    table=pd.read_csv(remote/'window_comparison.csv')
    assert len(table)==len(selected)==440 and not table.sample_tag.duplicated().any()
    assert table.sample_tag.tolist()==selected.sample_tag.tolist()
    assert np.array_equal(table.window_start,selected.window_start)
    details=json.loads((remote/'window_diagnostics.json').read_text())
    checks=[];spectra=[]
    with np.load(remote/'window_signals.npz',allow_pickle=False) as z:
        assert z['raw_modes'].shape==(440,7,256) and z['phase'].shape==(440,256)
        assert z['sample_tags'].tolist()==table.sample_tag.tolist()
        for i,row in table.iterrows():
            raw=z['raw_modes'][i]; phase=z['phase'][i]
            assert np.isfinite(raw).all()
            reference_signal=zscore_1d(phase).astype(np.float64)
            err=float(np.linalg.norm(raw.sum(axis=0)-reference_signal)/max(np.linalg.norm(reference_signal),1e-30))
            assert abs(err-row.reconstruction_relative_error)<1e-10
            norm=np.stack([zscore_1d(m) for m in raw]);weights,_,_=hr_adavmd_mode_scores(norm)
            assert np.allclose(weights,details[i]['mode_weights'],atol=1e-6)
            pairs=sum(np.array_equal(raw[a],raw[b]) for a in range(7) for b in range(a+1,7))
            assert pairs==row.duplicate_pairs
            signal_dict={'EDACM':phase[None,:],
                'legacy_rebuilt':z['rebuilt_legacy_channel0'][i][None,:],
                'sktime_raw_power':raw,'sktime_scored_power':norm*weights[:,None]}
            for method,signal in signal_dict.items():
                power=abs(np.fft.rfft(detrend(signal,axis=-1)*np.hanning(256),axis=-1))**2
                power=power.sum(axis=0); f=np.fft.rfftfreq(256,1/20.)
                band=(f>=.75)&(f<=2.5);ridge=float(f[band][power[band].argmax()]*60)
                reference=float(selected.iloc[i].label_heart_rate)
                spectra.append(dict(sample_tag=row.sample_tag,split=row.split,method=method,
                    ridge_bpm=ridge,historical_reference_bpm=reference,abs_error_bpm=abs(ridge-reference)))
            checks.append(err)
        # Copy arrays only after independent numerical validation.
        for filename in ('remote_summary.json','window_comparison.csv','window_diagnostics.json','window_signals.npz'):
            shutil.copyfile(remote/filename,out/filename)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,2,figsize=(11,10))
        for axrow,i in zip(axes,[0,220,439]):
            raw=z['raw_modes'][i];normalized=np.stack([zscore_1d(m) for m in raw])
            axrow[0].plot(np.arange(256)/20,normalized.T,linewidth=.8)
            axrow[0].set(title=table.sample_tag.iloc[i],xlabel='Time (s)',ylabel='Per-mode normalized amplitude')
            p=abs(np.fft.rfft(raw*np.hanning(256),axis=1))**2
            axrow[1].plot(np.fft.rfftfreq(256,.05),(p/np.maximum(p.sum(axis=1,keepdims=True),1e-30)).T)
            axrow[1].set(xlim=(0,10),xlabel='Hz',ylabel='Per-mode normalized power')
        fig.tight_layout();fig.savefig(out/'official_modes.png',dpi=150);plt.close(fig)
    train=manifest.loc[manifest.split=='train','label_heart_rate']
    for method,constant in [('train_mean_constant',train.mean()),('train_median_constant',train.median())]:
        for row in selected.itertuples():
            spectra.append(dict(sample_tag=row.sample_tag,split=row.split,method=method,
                ridge_bpm=float(constant),historical_reference_bpm=float(row.label_heart_rate),
                abs_error_bpm=abs(float(constant)-float(row.label_heart_rate))))
    spectral=pd.DataFrame(spectra);spectral.to_csv(out/'spectral_ridges.csv',index=False)
    spectral.groupby('method').agg(windows=('sample_tag','size'),ridge_mae_bpm=('abs_error_bpm','mean')).to_csv(out/'spectral_summary.csv')
    spectral.groupby(['split','method']).agg(windows=('sample_tag','size'),mae_bpm=('abs_error_bpm','mean')).to_csv(out/'spectral_summary_by_split.csv')
    summary=dict(elapsed_sec=time.perf_counter()-started,records=440,
        official_duplicate_records=int((table.duplicate_pairs>0).sum()),
        official_rank_counts=table.numerical_rank.value_counts().to_dict(),
        at_iteration_cap=int(table.at_iteration_cap.sum()),
        reconstruction_error_median=float(np.median(checks)),reconstruction_error_max=float(np.max(checks)),
        legacy_feature_mismatches=int((~table.legacy_matches_at_1e4).sum()),
        legacy_feature_max_error=float(table.legacy_reproduction_max_error.max()),
        target_bin_mismatches=int((~table.target_bins_match).sum()),
        legacy_rebuilt_duplicate_records=int((table.legacy_duplicate_pairs==21).sum()),
        validated_phase_mode_arrays=440,source_sha256=digest(__file__),
        caveat='Spectral ridge comparison is descriptive, uses historical unsynchronized labels and first window per record, not subject-level tracking or NN validation.',
        time_alignment_gate='UNVERIFIED',retraining_allowed=False)
    write_json(out/'validation.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
