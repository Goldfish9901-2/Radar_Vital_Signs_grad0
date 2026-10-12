"""Literature-inspired positive phase-correlation fusion, not full paper reproduction."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.signal import detrend
from src.data.evaluate_bgt60_selection import phase_fft_hr, metric

THRESHOLD=.75


def fuse_phase(phase, anchor, correlation_phase=None):
    """Same-RX five-bin phase; never accepts reference HR."""
    x=np.asarray(phase,dtype=np.float64)
    features=x if correlation_phase is None else np.asarray(correlation_phase,dtype=np.float64)
    norm=np.linalg.norm(features,axis=0)
    rho=np.zeros(x.shape[1])
    if norm[anchor]>1e-12:
        np.divide(features.T@features[:,anchor],norm*norm[anchor],out=rho,where=norm*norm[anchor]>1e-12)
    rho=np.clip(rho,-1,1)
    accepted=rho>THRESHOLD
    accepted[anchor]=True
    weights=np.where(accepted,np.maximum(rho,0),0)
    weights[anchor]=1.
    weights/=weights.sum()
    return x@weights,{'correlations':rho.tolist(),'weights':weights.tolist(),
                         'contributors':np.flatnonzero(weights>0).tolist()}


def hr_correlation_phase(phase):
    """Fixed FFT projection used only for correlations, not reconstructed output."""
    f=np.fft.rfftfreq(len(phase),1/30.)
    spectrum=np.fft.rfft(phase,axis=0)
    spectrum[(f<.75)|(f>2.5)]=0
    return np.fft.irfft(spectrum,n=len(phase),axis=0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=Path,default=Path('tmp/bgt60_p03_pair_v2_results/pair_audit'))
    parser.add_argument('--output-dir',type=Path,default=Path('tmp/bgt60_p03_fusion_results'))
    parser.add_argument('--hr-correlation-ablation',action='store_true')
    parser.add_argument('--projected-reconstruction-ablation',action='store_true')
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True);clock=time.perf_counter()
    if args.projected_reconstruction_ablation:args.hr_correlation_ablation=True
    protocol={'source':'https://doi.org/10.3390/s26175587','source_sections':['3.2','4.1'],
        'status':'literature-inspired component ablation; not full paper reproduction',
        'threshold':THRESHOLD,'rule':'positive full-window detrended unwrapped phase Pearson >0.75; same RX, five local bins; correlation weights normalized; anchor always retained',
        'anchors':['fixed RX2 center','window mean zero-Doppler power maximum over 15 cells'],
        'sign_alignment':'positive correlation only, no negative sign flip; constant phase offsets removed by detrend',
        'fft':'unchanged 384-frame, 30Hz assumed, Hann, 0.75–2.5Hz max power; no interpolation',
        'omitted':['dynamic IQ center tracking','CEEMD','harmonic subtraction','autocorrelation estimator','time smoothing'],
        'reference':'evaluation only, offsets 0/-60/+60/-120/+120; shared start unverified',
        'scope':'two exploratory p03 records only','training':False,
        'caution':'full-phase correlation can be dominated by respiration; it is not proof of HR information'}
    protocol['hr_correlation_ablation']=args.hr_correlation_ablation
    protocol['projected_reconstruction_ablation']=args.projected_reconstruction_ablation
    if args.hr_correlation_ablation:
        protocol['additional_rule']='same threshold/weights, but correlation uses 0.75–2.5 Hz FFT-projected phase; weighted output uses original detrended phase; no label optimization'
    if args.projected_reconstruction_ablation:
        protocol['reconstruction_control']='hrprojected uses exactly hrcorr weights but reconstructs HR-projected phase; an explicit preprocessing ablation, not selection-only improvement; finite-window projection may introduce boundary artifacts'
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    summaries={};provenance={}
    for distance in ['0.3m','0.6m']:
        path=args.input_dir/f'p03_{distance}_traces.npz'
        provenance[distance]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
        with np.load(path) as archive:
            z=archive['complex'][:,:,:,1];bins=archive['range_bins'];starts=archive['window_starts'];ref=archive['reference']
        rows=[];diagnostics=[];saved=[]
        methods=['fixed_single','fixed_fusion','energy_single','energy_fusion']
        if args.hr_correlation_ablation:methods=['fixed_single','fixed_fusion','fixed_hrcorr','energy_single','energy_fusion','energy_hrcorr']
        if args.projected_reconstruction_ablation:methods=['fixed_single','fixed_fusion','fixed_hrcorr','fixed_hrprojected','fixed_single_projected','energy_single','energy_fusion','energy_hrcorr','energy_hrprojected','energy_single_projected']
        for wi,start in enumerate(starts):
            window=z[start:start+384]
            phase=detrend(np.unwrap(np.angle(window),axis=0),axis=0)
            energy=(abs(window)**2).mean(axis=0)
            flat=int(np.argmax(energy));energy_rx,energy_bin=divmod(flat,5)
            predictions={};details={};signals=[]
            for anchor_name,rx,bi in [('fixed',2,2),('energy',energy_rx,energy_bin)]:
                single=phase[:,rx,bi]
                fused,detail=fuse_phase(phase[:,rx,:],bi)
                predictions[anchor_name+'_single']=phase_fft_hr(single)
                predictions[anchor_name+'_fusion']=phase_fft_hr(fused)
                details[anchor_name]={'rx':rx,'anchor_bin':int(bins[bi]),'contributor_bins':bins[detail['contributors']].tolist(),**detail}
                signals.extend([single,fused])
                if args.hr_correlation_ablation:
                    hr_fused,hr_detail=fuse_phase(phase[:,rx,:],bi,hr_correlation_phase(phase[:,rx,:]))
                    predictions[anchor_name+'_hrcorr']=phase_fft_hr(hr_fused)
                    details[anchor_name+'_hrcorr']={'rx':rx,'anchor_bin':int(bins[bi]),'contributor_bins':bins[hr_detail['contributors']].tolist(),**hr_detail}
                    signals.append(hr_fused)
                    if args.projected_reconstruction_ablation:
                        projected=hr_correlation_phase(phase[:,rx,:])@np.array(hr_detail['weights'])
                        predictions[anchor_name+'_hrprojected']=phase_fft_hr(projected)
                        signals.append(projected)
                        single_projected=hr_correlation_phase(phase[:,rx,:])[:,bi]
                        predictions[anchor_name+'_single_projected']=phase_fft_hr(single_projected)
                        signals.append(single_projected)
            saved.append(np.stack(signals))
            diagnostics.append({'window_start':int(start),'anchors':details})
            times=np.arange(start,start+384)/30
            for offset in [0,-60,60,-120,120]:
                labels=np.interp(times+offset,ref[:,0],ref[:,1],left=np.nan,right=np.nan)
                label=float(labels.mean()) if np.isfinite(labels).all() else None
                for method in methods:
                    rows.append({'method':method,'window_start':int(start),'offset_s':offset,'label_bpm':label,'prediction_bpm':predictions[method]})
        summaries[distance]={}
        for method in methods:
            r=[a for a in rows if a['method']==method]
            eligible=set.intersection(*[{a['window_start'] for a in r if a['offset_s']==o and a['label_bpm'] is not None} for o in [0,-60,60,-120,120]])
            for offset in [0,-60,60,-120,120]:
                rr=[a for a in r if a['offset_s']==offset]
                groups={'full':rr,'matched':[a for a in rr if a['window_start'] in eligible]}
                groups.update({f'block_{t}':[a for a in rr if t<=a['window_start']/30<t+150] for t in [0,150,300,450]})
                for group,rs in groups.items():summaries[distance][f'{method}/{group}/{offset}']=metric(rs)
        phases=np.stack(saved)
        spectra=abs(np.fft.rfft(phases*np.hanning(384),axis=-1))**2
        np.savez_compressed(out/f'p03_{distance}_fusion_windows.npz',phases=phases,power=spectra,
                            window_starts=starts,frequencies_hz=np.fft.rfftfreq(384,1/30),methods=np.array(methods))
        (out/f'p03_{distance}_windows.json').write_text(json.dumps({'predictions':rows,'fusion_details':diagnostics},indent=2,allow_nan=False))
        print(distance,'fusion complete',flush=True)
    for name,data in [('summary',summaries),('input_provenance',provenance)]:
        (out/f'{name}.json').write_text(json.dumps(data,indent=2,allow_nan=False))
    report=['# Phase-correlation fusion ablation (local uv)','',
        'Literature-inspired component test, not full Xu 2026 reproduction. No label-based fusion, VMD, harmonic filtering or smoothing.',
        '[Source, sections 3.2 and 4.1](https://doi.org/10.3390/s26175587): positive phase correlation and weighted neighboring-bin aggregation.',
        'All methods use the same FFT estimator. Base fusion uses full detrended phase correlation; any HR-band correlation ablation is specified in protocol.json.',
        '', '| Distance | Method | Valid/total | Full MAE | Full r | Matched r |','|---|---|---:|---:|---:|---:|']
    for d,stats in summaries.items():
        for method in methods:
            m=stats[f'{method}/full/0'];r=stats[f'{method}/matched/0']['pearson_r']
            report.append(f"| {d} | {method} | {m['valid']}/{m['windows']} | {m['mae_bpm']:.3f} | {m['pearson_r']:.3f} | {r:.3f} |")
    report+=['','All segment and offset controls are in summary.json; all contributing-bin correlations and weights are preserved.',
             'p03 was selected after earlier screening. No cross-subject efficacy claim; frame rate and absolute synchronization remain unverified.']
    if args.hr_correlation_ablation:
        report+=['','Additional hrcorr methods compute correlation on a fixed HR-band projection, but reconstruct original phase. This is a new exploratory component, not a published-method reproduction. All methods share the same FFT estimator; no thresholds tuned.']
    if args.projected_reconstruction_ablation:
        report+=['','hrprojected uses the same hrcorr weights but reconstructs HR-band-projected phase. This is explicitly a preprocessing change, not evidence that fusion/selection alone improves. Finite-window FFT projection can have boundary artifacts.']
    (out/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    timing={'seconds':time.perf_counter()-clock,'execution':'local uv','training':False,'threshold_tuned':False}
    (out/'timing.json').write_text(json.dumps(timing,indent=2));print(json.dumps(timing))


if __name__=='__main__':main()
