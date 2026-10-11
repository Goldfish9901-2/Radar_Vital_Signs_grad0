"""Fixed p03 bins/RX: preserve complex traces, spectra and quality diagnostics."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from scipy.signal import detrend
from src.features.edacm import edacm_phase
from src.data.diagnose_bgt60_adc import save_json


def coefficients(adc, bins):
    out = np.empty((len(adc), adc.shape[1], len(bins), 2), dtype=np.complex128)
    for start in range(0, len(adc), 64):
        x = np.array(adc[start:start+64], dtype=np.float64, copy=True)
        x -= x.mean(axis=-1, keepdims=True)
        r = np.fft.fft(x, axis=-1)[..., bins]
        out[start:start+len(x), :, :, 0] = r[:, :, 0, :]
        out[start:start+len(x), :, :, 1] = r.mean(axis=2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'subject':3, 'fixed_centers':{'0.3m':14,'0.6m':22}, 'fixed_rx_zero_based':2,
                'neighbor_radius_bins':2, 'fs_hz_assumed':30, 'window_frames':384,'stride':192,
                'fast_time_dc':'mean','chirp_clutter':'none','offsets_s':[0,-60,60,-120,120],
                'roi_selection':'frozen prior audit; no label optimization',
                'quality_rules':'descriptive only; no window exclusions or tuned thresholds',
                'timing':'shared start and 30Hz unverified','training':False,'vmd':False,
                'axes':'complex[frame,rx,range,coefficient]; spectra[window,rx,range,coefficient,method,frequency]',
                'coefficients':['first_chirp','zero_doppler'],'methods':['unwrap','edacm']}
    save_json(out/'protocol.json', protocol)
    summary = {}
    for distance,center in [('0.3m',14),('0.6m',22)]:
        files=[p for p in args.input_root.rglob('radar_raw_data.npy') if p.parent.name==distance
               and re.search(r'participant_3(?:[/\\])',str(p),re.I)]
        if len(files)!=1: raise ValueError(f'Expected exactly one p03 {distance} ADC file, got {len(files)}')
        raw=files[0]; adc=np.load(raw,mmap_mode='r')
        if adc.shape!=(18000,3,16,512): raise ValueError(f'Unexpected shape {adc.shape}')
        parent=next(p for p in raw.parents if p.name.lower()=='participant_3')
        ref=np.genfromtxt(parent/'HR_Ref_Data/Participant3'/distance/'HR_ref.csv',delimiter=',',skip_header=1)
        if ref.ndim!=2 or not np.isfinite(ref).all() or not np.all(np.diff(ref[:,0])>0): raise ValueError('Invalid reference')
        bins=np.arange(center-2,center+3)
        z=coefficients(adc,bins)
        starts=np.arange(0,len(z)-383,192)
        freqs=np.fft.rfftfreq(384,1/30)
        mask=(freqs>=.75)&(freqs<=2.5)
        spectra=np.empty((len(starts),3,5,2,2,len(freqs)),dtype=np.float32)
        phases=np.empty((len(starts),3,5,2,2,384),dtype=np.float32)
        rows=[]
        for wi,start in enumerate(starts):
            times=np.arange(start,start+384)/30
            for rx in range(3):
              for bi,b in enumerate(bins):
                for ci,c in enumerate(protocol['coefficients']):
                  trace=z[start:start+384,rx,bi,ci]
                  amp=np.abs(trace)
                  circular=np.angle(trace[1:]*trace[:-1].conj())
                  for mi,method in enumerate(protocol['methods']):
                    phase=detrend(np.unwrap(np.angle(trace))) if mi==0 else edacm_phase(trace)
                    power=np.abs(np.fft.rfft(phase*np.hanning(384)))**2
                    spectra[wi,rx,bi,ci,mi]=power
                    phases[wi,rx,bi,ci,mi]=phase
                    peak=np.flatnonzero(mask)[np.argmax(power[mask])]
                    valid=np.std(phase)>1e-6 and power[mask].sum()>1e-12
                    row={'window_start':int(start),'time_s':float(times.mean()),'rx':rx,'bin':int(b),
                         'coefficient':c,'method':method,'fixed_baseline':bool(rx==2 and b==center),
                         'fft_hr_bpm':float(freqs[peak]*60) if valid else None,
                         'amplitude_median':float(np.median(amp)),'amplitude_p01':float(np.quantile(amp,.01)),
                         'amplitude_relative_p01':float(np.quantile(amp,.01)/(np.median(amp)+1e-30)),
                         'circular_jump_fraction_gt_pi_over_2':float(np.mean(abs(circular)>np.pi/2)),
                         'phase_variance':float(np.var(phase)),
                         'hr_band_ratio':float(power[mask].sum()/(power[1:].sum()+1e-30)),
                         'reference':{}}
                    for offset in protocol['offsets_s']:
                        labels=np.interp(times+offset,ref[:,0],ref[:,1],left=np.nan,right=np.nan)
                        row['reference'][str(offset)]=float(labels.mean()) if np.isfinite(labels).all() else None
                    rows.append(row)
        # Full-resolution complex sequences allow IQ/phase reinspection without ADC.
        np.savez_compressed(out/f'p03_{distance}_traces.npz',complex=z,range_bins=bins,
                            time_s=np.arange(len(z))/30,reference=ref,
                            full_unwrap=np.unwrap(np.angle(z),axis=0),window_starts=starts,
                            window_phases=phases,window_power=spectra,frequencies_hz=freqs)
        save_json(out/f'p03_{distance}_windows.json',rows)
        # Label-free time stability: every window retains every RX/bin.
        energy=np.array([(abs(z[s:s+384,:,:,1])**2).mean(axis=0) for s in starts])
        save_json(out/f'p03_{distance}_stability.json',{
            'window_starts':starts.tolist(),'zero_doppler_energy':energy.tolist(),
            'strongest_local_bin_per_rx':bins[np.argmax(energy,axis=2)].tolist(),
            'strongest_rx_per_bin':np.argmax(energy,axis=1).tolist(),
            'note':'Diagnostic only; baseline bin/RX remains fixed. Full-record ROI power cannot establish temporal stability.'})
        summary[distance]={}
        for c in protocol['coefficients']:
          for method in protocol['methods']:
            baseline=[r for r in rows if r['fixed_baseline'] and r['coefficient']==c and r['method']==method]
            groups={'full':baseline,'matched':[r for r in baseline if all(v is not None for v in r['reference'].values())]}
            groups.update({f'block_{t}':[r for r in baseline if t<=r['window_start']/30<t+150] for t in [0,150,300,450]})
            for group,rs in groups.items():
              for offset in protocol['offsets_s']:
                valid=[r for r in rs if r['fft_hr_bpm'] is not None and r['reference'][str(offset)] is not None]
                y=np.array([r['reference'][str(offset)] for r in valid]); pred=np.array([r['fft_hr_bpm'] for r in valid])
                corr=float(np.corrcoef(y,pred)[0,1]) if len(y)>1 and min(y.std(),pred.std())>.01 else None
                summary[distance][f'{c}/{method}/{group}/{offset}']={'windows':len(rs),'valid':len(y),
                    'mae_bpm':float(abs(y-pred).mean()) if len(y) else None,'pearson_r':corr}
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(12,10))
        baseline=z[:,2,2,1]
        axes[0].plot(baseline.real[::10],baseline.imag[::10],'.',ms=1)
        axes[0].set(xlabel='I',ylabel='Q',title=f'p03 {distance}: fixed bin {center}, RX 2, zero Doppler IQ')
        axes[1].plot(np.arange(len(z))/30,abs(baseline))
        axes[1].set(xlabel='Time (s)',ylabel='Amplitude')
        db=10*np.log10(np.maximum(spectra[:,2,2,1,0,:],1e-30))
        axes[2].pcolormesh((starts+191.5)/30,freqs*60,db.T,shading='auto',vmin=np.quantile(db,.05),vmax=np.quantile(db,.99))
        axes[2].plot(ref[:,0],ref[:,1],color='white',lw=1,label='Reference HR; shared start assumed')
        axes[2].set(xlabel='Time (s)',ylabel='Frequency (bpm)',ylim=(0,180))
        axes[2].legend(); fig.tight_layout(); fig.savefig(out/f'p03_{distance}_diagnostic.png',dpi=160); plt.close(fig)
        print(distance,'complete',flush=True)
    save_json(out/'summary.json',summary)
    (out/'REPORT.md').write_text('# p03 paired complex diagnostics\n\nFixed prior bins/RX, no VMD/training. Complex traces, all RX/neighbor bins, window phases and full spectra retained.\n\nTiming and metric distance remain unverified. No label-based bin or offset selection.\n',encoding='utf-8')


if __name__=='__main__': main()
