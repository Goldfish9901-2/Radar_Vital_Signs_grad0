"""Continuous-record FTU stage spectra for descriptive frozen-model audit cases."""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend

from src.features.edacm import edacm_phase, target_edacm_signal
from src.features.hr_adavmd import hr_adavmd_decompose
from src.training.audit_predictions import sha256


def power_windows(signals, fs=20., n=256, hop=128):
    signals = np.atleast_2d(signals)
    starts = np.arange(0, signals.shape[-1]-n+1, hop)
    windows = np.stack([signals[:,s:s+n] for s in starts],axis=1)
    windows = detrend(windows,axis=-1,type='linear')
    power = abs(np.fft.rfft(windows*np.hanning(n),axis=-1))**2
    return power,np.fft.rfftfreq(n,1/fs),starts


def main():
    started = time.perf_counter()
    root = Path('/kaggle/input')
    out = Path('/kaggle/working/ftu_continuous'); out.mkdir(parents=True,exist_ok=True)
    spec = json.loads(Path(__file__).with_name('ftu_cases.json').read_text())
    pred_paths = list(root.rglob('preds_cycleformer_FTU_test.csv'))
    if len(pred_paths) != 1: raise ValueError('Ambiguous prediction source')
    preds = pd.read_csv(pred_paths[0])
    build = json.loads((pred_paths[0].parent/'training_exports/build_config.json').read_text())
    fs = float(build['sampling_rate_hz']); n = int(build['window_size']); hop = int(build['stride'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics, records = [],[]
    for category,tag in spec.items():
        tick = time.perf_counter()
        candidates = [p for p in root.rglob(tag+'.npz') if 'ftu-rda-exports' in str(p)]
        if len(candidates) != 1: raise ValueError(f'Ambiguous RDA source for {tag}: {candidates}')
        path = candidates[0]
        with np.load(path,allow_pickle=False) as data:
            radar = data['radar']; ref_time = data['time']; hr = data['heart_rate']
        if radar.ndim != 4 or not np.iscomplexobj(radar) or not np.isfinite(radar).all():
            raise ValueError('Require finite complex continuous RDA')
        pred = preds.loc[preds.sample_tag == tag].sort_values('window_start').copy()
        if pred.empty: raise ValueError('No matching frozen predictions')
        # Reproduce the historical export label mapping explicitly; do not silently
        # align a 1 Hz reference by array index or optimize an offset against predictions.
        from src.data.build_training_dataset import build_frame_times, interpolate_label_to_frames
        frame_time = build_frame_times(ref_time,len(radar))
        frame_hr = interpolate_label_to_frames(hr,ref_time,frame_time)
        expected = np.array([np.nanmean(frame_hr[int(s):int(s)+n]) for s in pred.window_start])
        label_error = float(np.max(abs(expected-pred.label_bpm)))
        if label_error > 1e-4: raise ValueError(f'Continuous reference label mismatch: {label_error}')
        fused,selection = target_edacm_signal(radar)
        d,a,r = selection['target_bins_dar'][0]
        trace = radar[:,d,a,r]
        raw = np.unwrap(np.angle(trace)).astype(np.float64)
        single = edacm_phase(trace)
        modes,vmd_meta = hr_adavmd_decompose(fused,sampling_rate_hz=fs)
        stages = {'RDA unwrap (top selected bin)':raw[None,:],
                  'EDACM (same single bin)':single[None,:],
                  'EDACM (selected-bin fusion)':fused[None,:],
                  'HR-AdaVMD (sum of channel powers)':modes}
        f = None; starts = None; stage_powers = []
        for name,signals in stages.items():
            p,f,starts = power_windows(signals,fs,n,hop)
            stage_powers.append(p.sum(axis=0))
        assert np.array_equal(starts,pred.window_start.to_numpy())
        band = (f >= .75) & (f <= 2.5)
        bpm = f[band]*60; t = (starts+n/2)/fs
        fig,axes = plt.subplots(4,1,figsize=(11,12),sharex=True,sharey=True)
        for ax,(name,_),power in zip(axes,stages.items(),stage_powers):
            p = power[:,band]
            relative = p/np.maximum(power[:,1:].sum(axis=1,keepdims=True),1e-30)
            db = 10*np.log10(np.maximum(relative,1e-8))
            ax.pcolormesh(t,bpm,db.T,shading='nearest',vmin=-50,vmax=0,cmap='magma')
            ax.plot(t,pred.label_bpm,'c.-',label='reference window mean')
            ax.plot(t,pred.pred_bpm,'w.--',label='frozen model')
            ax.set_title(name);ax.set_ylabel('Frequency (bpm)')
            ridge = bpm[np.argmax(p,axis=1)]
            for i,s in enumerate(starts):
                near = int(np.argmin(abs(bpm-pred.label_bpm.iloc[i])))
                metrics.append(dict(category=category,sample_tag=tag,stage=name,window_start=int(s),
                    reference_bpm=float(pred.label_bpm.iloc[i]),model_bpm=float(pred.pred_bpm.iloc[i]),
                    dominant_ridge_bpm=float(ridge[i]),ridge_abs_error=float(abs(ridge[i]-pred.label_bpm.iloc[i])),
                    hr_band_fraction=float(relative[i].sum()),
                    reference_nearest_bin_fraction=float(relative[i,near]),
                    band_total_power=float(p[i].sum()),non_dc_total_power=float(power[i,1:].sum())))
        axes[0].legend(loc='upper right');axes[-1].set_xlabel('Window center (s; nominal radar frame rate)')
        fig.suptitle(tag+' | continuous 60 s processing; descriptive cases')
        fig.tight_layout();fig.savefig(out/(category+'_stages.png'),dpi=160);plt.close(fig)
        # Use the same selected Doppler/angle for every range bin to isolate range.
        range_phase = np.unwrap(np.angle(radar[:,d,a,:]),axis=0).T
        rp,_,_ = power_windows(range_phase,fs,n,hop)
        rp = rp.mean(axis=1)
        rr = rp[:,band]/np.maximum(rp[:,1:].sum(axis=1,keepdims=True),1e-30)
        fig,ax = plt.subplots(figsize=(9,4))
        im = ax.pcolormesh(bpm,np.arange(radar.shape[-1]),10*np.log10(np.maximum(rr,1e-8)),
                           shading='nearest',vmin=-50,vmax=0,cmap='magma')
        ax.axvline(float(pred.label_bpm.mean()),color='cyan',label='mean reference')
        ax.axvline(float(pred.pred_bpm.mean()),color='white',ls='--',label='mean prediction')
        for rd in sorted(set(x[2] for x in selection['target_bins_dar'])):
            ax.axhline(rd,color='lime',ls=':',alpha=.6)
        ax.set(xlabel='Frequency (bpm)',ylabel='Cropped RDA range index',
               title=f'{tag} | fixed Doppler={d}, angle={a}')
        ax.legend();fig.colorbar(im,ax=ax,label='dB fraction of non-DC power')
        fig.tight_layout();fig.savefig(out/(category+'_range_frequency.png'),dpi=160);plt.close(fig)
        np.savez_compressed(out/(category+'_signals.npz'),raw_unwrap=raw,edacm_single=single,
            edacm_fused=fused,vmd_modes=modes,range_phase=range_phase,stage_power=np.stack(stage_powers),
            frequencies_hz=f,window_starts=starts,reference_window_hr=pred.label_bpm.to_numpy(),
            predicted_window_hr=pred.pred_bpm.to_numpy(),reference_time=ref_time,reference_hr=hr)
        records.append(dict(category=category,sample_tag=tag,source=str(path),source_sha256=sha256(path),
            radar_shape=list(radar.shape),sampling_rate_hz=fs,frame_rate_independently_verified=False,
            source_reference_time_min=float(np.nanmin(ref_time)),source_reference_time_max=float(np.nanmax(ref_time)),
            historical_frame_time_span=float(frame_time[-1]-frame_time[0]),nominal_radar_span=(len(radar)-1)/fs,
            label_mapping='Historical build_frame_times/interpolate_label_to_frames reproduced; no fitted lag',
            max_window_label_error_bpm=label_error,selection=selection,vmd=vmd_meta,
            elapsed_sec=time.perf_counter()-tick))
        print(tag,'done',records[-1]['elapsed_sec'],flush=True)
    pd.DataFrame(metrics).to_csv(out/'stage_metrics.csv',index=False)
    report = dict(records=records,elapsed_sec=time.perf_counter()-started,
        prediction_sha256=sha256(pred_paths[0]),selection_spec=spec,
        caveats=['Samples selected by test errors/HR variance for explanation, not independent validation.',
                 'Full-record target selection and VMD differ from original window-local model preprocessing.',
                 'RDA phase is already after the existing ADC-to-RDA conversion, not raw ADC phase.',
                 'HR-AdaVMD power sums seven channels; color scale measures spectral allocation, not amplitude gain.',
                 'Export reference-to-frame mapping is reproduced, not independent synchronization evidence.',
                 'RDA range indices are cropped, not native physical range; fixed Doppler/angle range slice.',
                 'Native 256-point frequency resolution at 20 Hz is 4.6875 bpm.'])
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__ == '__main__': main()
