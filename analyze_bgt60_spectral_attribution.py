"""Frozen radar-only peak extraction, followed by reference-based attribution."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.signal import detrend, find_peaks

FS=30.
N=384
DF=FS/N
PROTOCOL={'window_frames':N,'stride_frames':192,'fs_assumed_hz':FS,
          'respiration_band_hz':[.1,.6],'hr_band_hz':[.75,2.5],'top_peaks':3,
          'harmonic_orders':[2,3,4,5,6],'matching_tolerance_hz':DF,
          'error_threshold_bpm':5.,'negative_comb':'orders + 0.5, same tolerance',
          'peak_rule':'local maxima including HR-band endpoints; top three by power; endpoint flag retained',
          'selection':'fixed prior p03 center/RX2, zero Doppler unwrap; no ECG-dependent selection',
          'timing':'shared start/30Hz unverified','training':False,
          'interpretation':'harmonic proximity is compatible with interference, not proof of respiratory origin'}


def top_peaks(power, indices):
    band=power[indices]
    peaks=find_peaks(band)[0].tolist()
    if band[0]>=band[1]:peaks.append(0)
    if band[-1]>=band[-2]:peaks.append(len(band)-1)
    chosen=sorted(set(peaks),key=lambda p:(-band[p],p))[:3]
    return indices[chosen].tolist()


def harmonic_mask(frequencies, respiration_hz, shifted=False):
    orders=np.arange(2,7)+(0.5 if shifted else 0)
    return np.any(abs(frequencies[:,None]-respiration_hz*orders[None,:])<=DF+1e-12,axis=1)


def analyze_phase(phase):
    frequencies=np.fft.rfftfreq(N,1/FS)
    power=abs(np.fft.rfft(phase*np.hanning(N)))**2
    rr_mask=(frequencies>=.1)&(frequencies<=.6)
    hr_mask=(frequencies>=.75)&(frequencies<=2.5)
    rr_indices=np.flatnonzero(rr_mask);hr_indices=np.flatnonzero(hr_mask)
    valid=np.std(phase)>1e-6
    respiration=float(frequencies[rr_indices[np.argmax(power[rr_mask])]]) if valid and power[rr_mask].sum()>1e-12 else None
    peak=int(hr_indices[np.argmax(power[hr_mask])])
    prediction=float(frequencies[peak]*60) if valid and power[hr_mask].sum()>1e-12 else None
    peaks=top_peaks(power,hr_indices)
    h=harmonic_mask(frequencies,respiration) if respiration else np.zeros(len(frequencies),bool)
    null=harmonic_mask(frequencies,respiration,True) if respiration else np.zeros(len(frequencies),bool)
    row={'radar_rr_hz':respiration,'prediction_bpm':prediction,'lowest_hr_bin':bool(peak==hr_indices[0]),
         'rr_search_edge':bool(respiration in frequencies[[rr_indices[0],rr_indices[-1]]]) if respiration else None,
         'top_hr_peaks':[{'hz':float(frequencies[i]),'bpm':float(frequencies[i]*60),'power':float(power[i]),
                          'band_endpoint':i in [hr_indices[0],hr_indices[-1]],'near_harmonic':bool(h[i])} for i in peaks],
         'prediction_near_harmonic':bool(h[peak]) if respiration and prediction else None,
         'prediction_near_shifted_comb':bool(null[peak]) if respiration and prediction else None,
         'harmonic_hr_grid_coverage':float(h[hr_mask].mean()) if respiration else None,
         'shifted_hr_grid_coverage':float(null[hr_mask].mean()) if respiration else None,
         'harmonic_hr_energy_fraction':float(power[h&hr_mask].sum()/(power[hr_mask].sum()+1e-30)) if respiration else None,
         'harmonics':[{'order':k,'hz':float(k*respiration),'energy':float(power[abs(frequencies-k*respiration)<=DF+1e-12].sum())}
                      for k in range(2,7) if .75-DF<=k*respiration<=2.5+DF] if respiration else []}
    return row,power


def summarize(rows):
    valid=[r for r in rows if r['reference_hr_bpm'] is not None and r['prediction_bpm'] is not None]
    result={'windows':len(rows),'valid':len(valid)}
    for name,subset in [('all',valid),('wrong',[r for r in valid if r['absolute_error_bpm']>5]),
                        ('within_5',[r for r in valid if r['absolute_error_bpm']<=5])]:
        result[name]={'windows':len(subset)}
        for flag in ['prediction_near_harmonic','prediction_near_shifted_comb','reference_hr_near_harmonic','prediction_near_reference_rr_harmonic','reference_hr_in_top3','lowest_hr_bin','rr_search_edge']:
            eligible=[r[flag] for r in subset if r[flag] is not None]
            result[name][flag+'_fraction']=float(np.mean(eligible)) if eligible else None
        for field in ['harmonic_hr_grid_coverage','shifted_hr_grid_coverage','harmonic_hr_energy_fraction','reference_rr_error_bpm']:
            values=[r[field] for r in subset if r[field] is not None]
            result[name][field+'_mean']=float(np.mean(values)) if values else None
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=Path,default=Path('tmp/bgt60_p03_pair_v2_results/pair_audit'))
    parser.add_argument('--output-dir',type=Path,default=Path('tmp/bgt60_p03_harmonic_audit'))
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True);clock=time.perf_counter()
    (out/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    summaries={};inputs={}
    for distance in ['0.3m','0.6m']:
        path=args.input_dir/f'p03_{distance}_traces.npz'
        inputs[distance]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        with np.load(path) as data:
            trace=data['complex'][:,2,2,1];starts=data['window_starts'];ref=data['reference']
        rows=[];spectra=[]
        for start in starts:
            phase=detrend(np.unwrap(np.angle(trace[start:start+N])))
            row,power=analyze_phase(phase);spectra.append(power)
            # Reference is read only after extracting radar peaks/harmonic energies.
            times=np.arange(start,start+N)/FS
            hr=np.interp(times,ref[:,0],ref[:,1],left=np.nan,right=np.nan)
            rr=np.interp(times,ref[:,0],ref[:,2],left=np.nan,right=np.nan)
            label=float(hr.mean()) if np.isfinite(hr).all() else None
            rr_label=float(rr.mean()) if np.isfinite(rr).all() else None
            row.update({'window_start':int(start),'reference_hr_bpm':label,'reference_rr_bpm':rr_label,
                        'absolute_error_bpm':abs(row['prediction_bpm']-label) if row['prediction_bpm'] is not None and label is not None else None,
                        'reference_rr_error_bpm':abs(row['radar_rr_hz']*60-rr_label) if row['radar_rr_hz'] and rr_label is not None else None,
                        'prediction_near_reference_rr_harmonic':bool(harmonic_mask(np.array([row['prediction_bpm']/60]),rr_label/60)[0]) if row['prediction_bpm'] is not None and rr_label is not None and rr_label>0 else None,
                        'reference_hr_in_top3':bool(any(abs(p['bpm']-label)<=DF*60+1e-10 for p in row['top_hr_peaks'])) if label is not None else None,
                        'reference_hr_near_harmonic':bool(harmonic_mask(np.array([label/60]),row['radar_rr_hz'])[0]) if label is not None and row['radar_rr_hz'] else None})
            rows.append(row)
        summaries[distance]={'full':summarize(rows),
            'blocks':{str(t):summarize([r for r in rows if t<=r['window_start']/30<t+150]) for t in [0,150,300,450]}}
        (out/f'p03_{distance}_windows.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
        np.savez_compressed(out/f'p03_{distance}_spectra.npz',power=np.array(spectra),frequencies_hz=np.fft.rfftfreq(N,1/FS),window_starts=starts)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))
    (out/'input_provenance.json').write_text(json.dumps(inputs,indent=2))
    (out/'timing.json').write_text(json.dumps({'seconds':time.perf_counter()-clock,'execution':'local uv','training':False},indent=2))
    report=['# p03 spectral peak attribution (frozen diagnostic)', '',
        'Radar peaks are extracted before reference use. Reference HR/RR are used only to audit coincidences; no estimator, bin or threshold is changed.',
        'Respiration search 0.1–0.6 Hz; HR 0.75–2.5 Hz; top three local peaks, including flagged band endpoints. Orders 2–6; tolerance one FFT bin (0.078125 Hz = 4.6875 bpm). Wrong means absolute HR error >5 bpm.',
        '', '| Distance | Error group | Windows | Radar-harmonic match | Half-order control | Reference HR near radar harmonic | Prediction near reference-RR harmonic | HR in top3 (diagnostic) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for distance,entry in summaries.items():
        for group in ['wrong','within_5']:
            s=entry['full'][group]
            fields=['prediction_near_harmonic_fraction','prediction_near_shifted_comb_fraction','reference_hr_near_harmonic_fraction','prediction_near_reference_rr_harmonic_fraction','reference_hr_in_top3_fraction']
            report.append(f"| {distance} | {group} | {s['windows']} | "+' | '.join(f"{s[k]:.3f}" if s[k] is not None else 'NA' for k in fields)+' |')
    report+=['', 'Harmonic masks and shifted masks have different boundary coverage; their grid coverage is preserved per window and in summary.json. Neither control is a statistical null test. Matching is not proof of respiratory origin.',
        'Reference HR often coincides with an estimated respiratory multiple, including in accurate windows. Removing such peaks could remove real heartbeat information. Top3 recall is label-assisted diagnosis, not deployable oracle accuracy.',
        'Respiration frequency resolution is coarse and the lowest/upper search bins can contain drift or noise. Reference RR is an existing 1Hz column under unverified shared-start synchronization; no raw ECG is available.',
        'Both p03 cases are exploratory; overlapping windows are not independent experiments. Do not use these labels to choose harmonic subtraction parameters.']
    (out/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    print('\n'.join(report))


if __name__=='__main__':main()
