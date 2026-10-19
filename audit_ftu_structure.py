"""Archived complex-DC diagnostics under the committed six-record protocol.

No ADC conversion, learning, HR-guided selection, or time-offset optimization.
All output powers are fractions of the fixed 45--150 bpm band.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path('experiment_runs/ftu_frequency_structure')
SRC = Path('experiment_runs/ftu_adc_controlled')
ARMS = ['legacy_mean', 'full_mean', 'legacy_none', 'full_none']
SCORE = ['legacy_mean', 'legacy_none', 'full_none']
SHIFTS = [0, -20, -15, 15, 20]
FREQ = np.fft.rfftfreq(4096, .05) * 60
BAND = (FREQ >= 45) & (FREQ <= 150)
FREQ = FREQ[BAND]


def detrend(x):
    t = np.arange(x.shape[-1], dtype=float)
    t -= t.mean()
    return x - x.mean(axis=-1, keepdims=True) - (x @ t / (t @ t))[..., None] * t


def spectrum(x):
    p = abs(np.fft.rfft(detrend(x) * np.hanning(x.shape[-1]), n=4096, axis=-1)[..., BAND]) ** 2
    den = p.sum(axis=-1, keepdims=True)
    return np.divide(p, den, out=np.full_like(p, np.nan), where=den > 0)


def make_segments(z, ref, starts, length, scale):
    """cell x segment x sample; reject undefined samples, never gap-fill."""
    seg = np.stack([z[:, s:s+length] for s in starts], axis=1).astype(np.complex128)
    base = np.stack([ref[:, s:s+length] for s in starts], axis=1)
    amp = abs(seg)
    floor = np.sqrt(np.mean(abs(base)**2, axis=-1)) * scale * 1e-6
    threshold = np.median(abs(ref[:, :256]), axis=-1)[:, None, None] * scale * .01
    valid = (np.isfinite(seg).all(axis=-1) & (amp > 0).all(axis=-1)
             & (np.sqrt(np.mean(amp**2, axis=-1)) > floor)
             & ((amp < threshold).mean(axis=-1) <= .01))
    delta = np.angle(seg[..., 1:] * seg[..., :-1].conj())
    phase = np.concatenate([np.zeros_like(delta[..., :1]), np.cumsum(delta, axis=-1)], axis=-1)
    phase[~valid] = np.nan
    logamp = np.full_like(amp, np.nan)
    np.log(amp, out=logamp, where=amp > 0)
    logamp[~valid] = np.nan
    return phase, spectrum(phase), spectrum(logamp), valid


def labels(d, starts, length):
    rt, rh = d['reference_time'], d['reference_hr']
    values = np.empty((3, 5, len(starts)))
    for off in range(3):
        for j, shift in enumerate(SHIFTS):
            for k, s in enumerate(starts):
                q = np.arange(s, s+length)/20 + off
                if shift:
                    q = (q + shift) % 60
                v = np.interp(q, rt, rh, left=np.nan, right=np.nan)
                values[off, j, k] = v.mean()
    common = np.isfinite(values).all(axis=(0, 1)) & (starts >= 256)
    return values, common


def scalar_metrics(p, valid, y, native):
    if len(y)==0:
        return dict(coverage=np.nan, mae=np.nan, within5=np.nan, near_power=np.nan,
                    centered_rmse=np.nan, r=np.nan, std_ratio=np.nan, ridge_step=np.nan)
    peak = FREQ[np.nan_to_num(p, nan=-1).argmax(axis=-1)]
    use = valid & np.isfinite(p).all(axis=-1)
    near = (p * (abs(FREQ[None, None, :] - y[None, :, None]) <= native)).sum(axis=-1)
    err = peak - y[None, :]
    if not use.any():
        return dict(coverage=0., mae=np.nan, within5=np.nan, near_power=np.nan,
                    centered_rmse=np.nan, r=np.nan, std_ratio=np.nan, ridge_step=np.nan)
    a, b = [], []
    for cell in range(len(p)):
        v = use[cell]
        if v.any():
            a.extend(peak[cell, v] - peak[cell, v].mean())
            b.extend(y[v] - y[v].mean())
    a, b = np.array(a), np.array(b)
    r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
    steps = abs(np.diff(peak, axis=1))[use[:, 1:] & use[:, :-1]]
    # Equal-cell means; reliability differences do not give higher-coverage cells more weight.
    def equal(v):
        means = [v[c, use[c]].mean() for c in range(len(p)) if use[c].any()]
        return float(np.mean(means))
    return dict(coverage=float(use.mean()), mae=equal(abs(err)), within5=equal((abs(err) <= 5).astype(float)),
                near_power=equal(near), centered_rmse=float(np.sqrt(np.mean((a-b)**2))), r=float(r),
                std_ratio=float(a.std()/b.std()) if b.std() else np.nan,
                ridge_step=float(np.median(steps)) if len(steps) else np.nan)


def main():
    begin = time.perf_counter()
    for name, h in json.loads((OUT/'input_hashes.json').read_text()).items():
        assert hashlib.sha256((SRC/name).read_bytes()).hexdigest() == h, name
    (OUT/'spectra').mkdir(exist_ok=True)
    (OUT/'figures').mkdir(exist_ok=True)
    old_masks = np.load(SRC/'reliability_masks.npz')
    rows, qualities, cells, separations, selections, descriptive = [], [], [], [], [], []
    validations = []
    for path in sorted((SRC/'signals').glob('*.npz')):
        tag = path.stem
        d = np.load(path)
        roi = d['roi'].astype(int)
        beams = d['calibration_energy'][:, roi].argmax(axis=0)
        panel = beams * 8 + np.arange(8)
        targets = np.array([b*8 + int(np.where(roi == r)[0][0]) for b, r in d['targets']])
        ref = d['full_none_DC_roi'].transpose(1, 2, 0).reshape(128, -1)
        assert ref.shape == (128, 1200)
        np.testing.assert_allclose(abs(ref[:, :256]).mean(axis=-1).reshape(16, 8),
                                   d['calibration_energy'][:, roi], rtol=2e-6)
        for k, (b, r) in enumerate(zip(beams, roi)):
            selections.append(dict(record=tag, panel_cell=k, beam=int(b), range=int(r)))
        saved = dict(roi=roi, panel_beams=beams, panel_cells=panel, target_cells=targets,
                     target_weights=d['weights'], frequency_bpm=FREQ)
        for arm in ARMS:
            z = d[arm+'_DC_roi'].transpose(1, 2, 0).reshape(128, -1)
            np.testing.assert_array_equal(z[targets], d[arm+'_z'])
            scale = .125 if arm.startswith('legacy') else 1.
            recompute = []
            for s in d['starts']:
                v, base = z[targets, s:s+256], ref[targets, s:s+256]
                recompute.append(np.isfinite(v).all(axis=1) & (abs(v)>0).all(axis=1)
                                 & (np.sqrt(np.mean(abs(v)**2, axis=1)) > 1e-6*scale*np.sqrt(np.mean(abs(base)**2, axis=1))))
            np.testing.assert_array_equal(recompute, old_masks[tag+'_'+arm])
            validations.append(dict(record=tag, arm=arm, original_masks_equal=True, targets_equal=True))
            amp = abs(z.astype(np.complex128))
            delta = np.angle(z[:, 1:].astype(np.complex128) * z[:, :-1].astype(np.complex128).conj())
            for c in range(128):
                qualities.append(dict(record=tag, arm=arm, beam=c//8, range=int(roi[c%8]),
                    rms=float(np.sqrt(np.mean(amp[c]**2))),
                    amplitude_ratio=float(np.sqrt(np.mean(amp[c]**2))/np.sqrt(np.mean(abs(ref[c].astype(np.complex128))**2))/scale),
                    amplitude_cv=float(amp[c].std()/amp[c].mean()) if amp[c].mean() else np.nan,
                    dropout=float((amp[c] < .01*scale*np.median(abs(ref[c,:256]))).mean()),
                    nonfinite=int((~np.isfinite(z[c])).sum()), zero=int((amp[c]==0).sum()),
                    near_pi_fraction=float((abs(delta[c]) > .9*np.pi).mean()),
                    abs_increment_p99=float(np.quantile(abs(delta[c]), .99))))
            if arm not in SCORE:
                continue
            for length in [512, 768]:
                starts = np.arange(0, 1200-length+1, 20)
                phase, p, ap, valid = make_segments(z, ref, starts, length, scale)
                ys, common = labels(d, starts, length)
                native = 1200/length
                fusion = np.sum(detrend(phase[targets]) * d['weights'][:, None, None], axis=0)[None]
                fp = spectrum(fusion)
                fv = valid[targets].all(axis=0)[None]
                saved.update({f'{arm}_{length}_phase_panel':p[panel].astype('float32'),
                              f'{arm}_{length}_amplitude_panel':ap[panel].astype('float32'),
                              f'{arm}_{length}_valid_panel':valid[panel],
                              f'{arm}_{length}_valid_all':valid,
                              f'{arm}_{length}_mean_phase_all':np.nanmean(p, axis=1).astype('float32'),
                              f'{arm}_{length}_mean_amplitude_all':np.nanmean(ap, axis=1).astype('float32'),
                              f'{arm}_{length}_fusion':fp.astype('float32'),
                              f'{arm}_{length}_valid_fusion':fv,
                              f'starts_{length}':starts, f'labels_{length}':ys, f'common_{length}':common})
                for off in range(3):
                    true_common=np.isfinite(ys[:,0]).all(axis=0)&(starts>=256)
                    for kind, spectra, mask in [('panel', p[panel], valid[panel]), ('fusion', fp, fv)]:
                        descriptive.append(dict(record=tag,arm=arm,length=length,offset=off,kind=kind,
                            common_windows=int(true_common.sum()),
                            **scalar_metrics(spectra[:,true_common],mask[:,true_common],ys[off,0,true_common],native)))
                    for j, shift in enumerate(SHIFTS):
                        for kind, spectra, mask in [('panel', p[panel], valid[panel]), ('fusion', fp, fv)]:
                            metrics = scalar_metrics(spectra[:,common], mask[:,common], ys[off,j,common], native)
                            rows.append(dict(record=tag, arm=arm, length=length, offset=off, shift=shift,
                                             kind=kind, common_windows=int(common.sum()), **metrics))
                        for c in range(8):
                            metrics = scalar_metrics(p[panel[c]:panel[c]+1,common], valid[panel[c]:panel[c]+1,common], ys[off,j,common], native)
                            cells.append(dict(record=tag,arm=arm,length=length,offset=off,shift=shift,beam=int(beams[c]),range=int(roi[c]),**metrics))
                        if arm == 'full_none' and shift:
                            diff = ys[off,j,common]-ys[off,0,common]
                            separations.append(dict(record=tag,length=length,offset=off,shift=shift,
                                common_windows=int(common.sum()),
                                mean_absolute_separation=float(abs(diff).mean()) if len(diff) else np.nan,
                                max_absolute_separation=float(abs(diff).max()) if len(diff) else np.nan,
                                native_bin_bpm=native, fraction_within_native=float((abs(diff)<native).mean()) if len(diff) else np.nan))
        np.savez_compressed(OUT/'spectra'/f'{tag}.npz', **saved)
        plot_record(tag, saved, d)
        print(tag, 'complete', flush=True)
    pd.DataFrame(rows).to_csv(OUT/'record_metrics.csv', index=False)
    pd.DataFrame(cells).to_csv(OUT/'cell_metrics.csv', index=False)
    pd.DataFrame(qualities).to_csv(OUT/'cell_quality.csv', index=False)
    pd.DataFrame(separations).to_csv(OUT/'label_separation.csv', index=False)
    pd.DataFrame(selections).to_csv(OUT/'panel_selection.csv', index=False)
    pd.DataFrame(descriptive).to_csv(OUT/'descriptive_true_only.csv',index=False)
    table = pd.DataFrame(rows)
    gates=[]
    for (record,arm), group in table[(table.length==512)&(table.kind=='panel')].groupby(['record','arm']):
        if arm=='legacy_mean': continue
        passes=[]
        for off,g in group.groupby('offset'):
            true=g[g['shift']==0].iloc[0]; controls=g[g['shift']!=0]
            cover=true.coverage>=.9; hit=true.within5>=.6
            power=bool((true.near_power-controls.near_power>=.01).all())
            error=bool((true.mae<controls.mae).all())
            passed=bool(cover and hit and power and error);passes.append(passed)
            gates.append(dict(record=record,arm=arm,offset=off,coverage_pass=cover,within5_pass=hit,
                              power_pass=power,mae_pass=error,offset_pass=passed,evaluable=bool(true.common_windows>0)))
        for g in gates[-3:]:g['record_pass']=all(passes)
    pd.DataFrame(gates).to_csv(OUT/'gate.csv',index=False)
    (OUT/'validation.json').write_text(json.dumps(dict(input_hashes_verified=10,
        original_mask_comparisons=validations, calibration_panel_verified=True),indent=2))
    (OUT/'execution.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-begin,
        protocol_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        python_versions=dict(numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
        local=True,training=False,new_exports=False,command='uv run --locked --no-sync python audit_ftu_structure.py'),indent=2))


def plot_record(tag, saved, d):
    plt.rcParams.update({'font.size':7})
    fig, axes=plt.subplots(3,8,figsize=(20,8),sharex=True,sharey=True)
    starts=saved['starts_512']; centers=(starts+256)/20
    for row,arm in enumerate(SCORE):
        p=saved[f'{arm}_512_phase_panel']
        for c in range(8):
            ax=axes[row,c]
            display=np.where(saved[f'{arm}_512_valid_panel'][c,:,None],p[c],np.nan)
            ax.pcolormesh(centers,FREQ,10*np.log10(np.maximum(display.T,1e-6)),vmin=-45,vmax=-10,shading='nearest')
            ax.plot(d['reference_time'],d['reference_hr'],'w-',lw=1)
            ax.axvline(12.8,color='red',ls=':',lw=.7)
            ax.set_ylim(45,150);ax.set_xlim(centers[0],centers[-1])
            if row==0:ax.set_title(f"R{saved['roi'][c]} / B{saved['panel_beams'][c]}")
            if c==0:ax.set_ylabel(arm+'\nbpm')
            if row==2:ax.set_xlabel('radar seconds')
    fig.suptitle(tag+' | P2 phase spectra, 512 frames; white=HR(+0s), red=calibration end\nBand-normalized dB; zero padding is display only; blank=unreliable')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'figures'/f'{tag}_phase.png',dpi=120);plt.close(fig)
    fig,axes=plt.subplots(2,8,figsize=(20,5),sharex=True,sharey=True)
    for row,kind in enumerate(['phase','amplitude']):
        p=saved[f'full_none_512_{kind}_panel']
        for c in range(8):
            axes[row,c].pcolormesh(centers,FREQ,10*np.log10(np.maximum(p[c].T,1e-6)),vmin=-45,vmax=-10,shading='nearest')
            axes[row,c].plot(d['reference_time'],d['reference_hr'],'w-',lw=1)
            axes[row,c].set_ylim(45,150);axes[row,c].set_xlim(centers[0],centers[-1])
            if c==0:axes[row,c].set_ylabel(kind+' / bpm')
            axes[row,c].set_title(f"R{saved['roi'][c]} B{saved['panel_beams'][c]}")
    fig.suptitle(tag+' | full_none P2 vs log-amplitude, 512 frames; white=HR(+0s)')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'figures'/f'{tag}_amplitude.png',dpi=120);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,9))
    for row,arm in enumerate(SCORE):
        # Equal-beam descriptive mean per saved range, not HR-selected.
        p=saved[f'{arm}_512_mean_phase_all'].reshape(16,8,-1)
        axes[row,0].pcolormesh(FREQ,saved['roi'],10*np.log10(np.maximum(np.nanmean(p,axis=0),1e-6)),vmin=-45,vmax=-10,shading='nearest')
        axes[row,0].set_ylabel(arm+' range')
        axes[row,1].imshow(saved[f'{arm}_512_valid_all'].mean(axis=1).reshape(16,8),vmin=0,vmax=1,aspect='auto',origin='lower')
        axes[row,1].set_ylabel('beam');axes[row,1].set_xlabel('saved range index')
    axes[0,0].set_title('All-beam mean normalized phase power');axes[0,1].set_title('Reliable STFT fraction, each of128 cells')
    axes[-1,0].set_xlabel('bpm');fig.suptitle(tag+' | descriptive spatial maps, no HR selection')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'figures'/f'{tag}_maps.png',dpi=120);plt.close(fig)


if __name__=='__main__':main()
