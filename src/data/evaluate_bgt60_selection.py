"""Frozen causal channel selection; reference HR is used only for evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import detrend

RULES={'energy':'mean |z|^2', 'quality':'log(E/max(E)) + 2*log(max(p01(|z|)/median(|z|),1e-6)) - 4*jump_fraction - 0.25*switch',
       'jump_threshold_rad':float(np.pi/2), 'initial_previous':None,
       'candidate_order':'RX major, ascending range bin; ties use first',
       'phase':'independent per-window unwrap + linear detrend; no channel phase stitching',
       'fft':'384 samples, 30 Hz assumed, Hann, max power in 0.75–2.5 Hz, no interpolation',
       'quality_filter':'none; all windows evaluated; no ECG-dependent rules or parameters'}


def select_channels(z, starts):
    energy_choices,quality_choices,features=[],[],[]
    previous=None
    for start in starts:
        x=z[start:start+384].reshape(384,-1)
        amp=np.abs(x)
        energy=np.mean(amp**2,axis=0)
        ratio=np.quantile(amp,.01,axis=0)/(np.median(amp,axis=0)+1e-30)
        jumps=np.mean(np.abs(np.angle(x[1:]*x[:-1].conj()))>np.pi/2,axis=0)
        score=np.log(np.maximum(energy/(energy.max()+1e-30),1e-30))+2*np.log(np.maximum(ratio,1e-6))-4*jumps
        if previous is not None:score-=.25*(np.arange(len(score))!=previous)
        energy_choices.append(int(np.argmax(energy)))
        previous=int(np.argmax(score));quality_choices.append(previous)
        features.append({'energy':energy.tolist(),'relative_amp_p01':ratio.tolist(),'jump_fraction':jumps.tolist(),'quality_score':score.tolist()})
    return {'energy':energy_choices,'quality':quality_choices},features


def fft_hr(trace):
    phase=detrend(np.unwrap(np.angle(trace)))
    power=np.abs(np.fft.rfft(phase*np.hanning(len(phase))))**2
    f=np.fft.rfftfreq(len(phase),1/30.)
    mask=(f>=.75)&(f<=2.5)
    if np.std(phase)<=1e-6 or power[mask].sum()<=1e-12:return None
    return float(f[np.flatnonzero(mask)[np.argmax(power[mask])]]*60)


def metric(rows):
    valid=[r for r in rows if r['prediction_bpm'] is not None and r['label_bpm'] is not None]
    if not valid:return {'windows':len(rows),'valid':0,'mae_bpm':None,'pearson_r':None}
    p=np.array([r['prediction_bpm'] for r in valid]);y=np.array([r['label_bpm'] for r in valid])
    return {'windows':len(rows),'valid':len(valid),'mae_bpm':float(np.mean(abs(p-y))),
            'pearson_r':float(np.corrcoef(p,y)[0,1]) if min(p.std(),y.std())>.01 else None,
            'prediction_std_bpm':float(p.std()),'reference_std_bpm':float(y.std()),
            'low_edge_fraction':float(np.mean(p==46.875))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    protocol={'rules':RULES,'offsets_s':[0,-60,60,-120,120],
        'records':{'0.3m':{'center':14,'rx':2},'0.6m':{'center':22,'rx':2}},
        'quality_rule_status':'new exploratory heuristic frozen before this run; not optimized on HR',
        'scope':'p03 only, no cross-subject claim; rules require validation on other subjects',
        'synchronization':'shared start/30Hz unverified','training':False,'vmd':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    summary={};provenance={};controls={}
    for distance,center in [('0.3m',14),('0.6m',22)]:
        paths=list(args.input_root.rglob(f'p03_{distance}_traces.npz'))
        if len(paths)!=1:raise ValueError(f'Expected one mounted pair archive: {distance}, found {len(paths)}')
        path=paths[0]
        provenance[distance]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
        with np.load(path) as data:
            z=data['complex'][:,:,:,1];bins=data['range_bins'];starts=data['window_starts'];ref=data['reference']
        assert z.shape==(18000,3,5) and bins[2]==center
        # This function has no label argument; choices are frozen before reference interpolation.
        choices,features=select_channels(z,starts)
        choices['fixed']=[12]*len(starts)
        traces=z.reshape(len(z),15)
        all_predictions=[[fft_hr(traces[s:s+384,ch]) for ch in range(15)] for s in starts]
        rows=[];selection={}
        for strategy,channels in choices.items():
            selection[strategy]={'switches':sum(a!=b for a,b in zip(channels,channels[1:])),
                'choices':[{'start':int(s),'rx':int(ch//5),'bin':int(bins[ch%5])} for s,ch in zip(starts,channels)]}
            for wi,(start,ch) in enumerate(zip(starts,channels)):
                times=np.arange(start,start+384)/30
                for offset in protocol['offsets_s']:
                    y=np.interp(times+offset,ref[:,0],ref[:,1],left=np.nan,right=np.nan)
                    rows.append({'strategy':strategy,'window_start':int(start),'rx':int(ch//5),'bin':int(bins[ch%5]),
                                 'offset_s':offset,'prediction_bpm':all_predictions[wi][ch],
                                 'label_bpm':float(y.mean()) if np.isfinite(y).all() else None})
        summary[distance]={}
        for strategy in choices:
            r=[r for r in rows if r['strategy']==strategy]
            eligible=set.intersection(*[{a['window_start'] for a in r if a['offset_s']==o and a['label_bpm'] is not None} for o in protocol['offsets_s']])
            for offset in protocol['offsets_s']:
                rr=[a for a in r if a['offset_s']==offset]
                for name,subset in {'full':rr,'matched':[a for a in rr if a['window_start'] in eligible],
                      **{f'block_{t}':[a for a in rr if t<=a['window_start']/30<t+150] for t in [0,150,300,450]}}.items():
                    summary[distance][f'{strategy}/{name}/{offset}']=metric(subset)
        # Pre-specified attribution controls, reported separately from three strategies.
        controls[distance]={}
        candidates=[(2,center),(2,23)] if distance=='0.6m' else [(rx,14) for rx in range(3)]
        for rx,b in candidates:
            ch=rx*5+int(np.flatnonzero(bins==b)[0]);rs=[]
            for wi,start in enumerate(starts):
                if not ((start/30<150) if distance=='0.6m' else (start/30>=450)):continue
                y=np.interp(np.arange(start,start+384)/30,ref[:,0],ref[:,1],left=np.nan,right=np.nan)
                rs.append({'prediction_bpm':all_predictions[wi][ch],'label_bpm':float(y.mean()) if np.isfinite(y).all() else None})
            controls[distance][f'rx{rx}_bin{b}']=metric(rs)
        (out/f'p03_{distance}_selections.json').write_text(json.dumps({'selection':selection,'features':features,'window_predictions':rows},indent=2,allow_nan=False))
        print(distance,'complete',flush=True)
    for name,value in [('summary',summary),('prespecified_controls',controls),('input_provenance',provenance)]:
        (out/f'{name}.json').write_text(json.dumps(value,indent=2,allow_nan=False))
    report=['# p03 frozen ROI/RX selection comparison','',
            'All strategies share the same independent-window unwrap FFT. Quality parameters were frozen before execution; no label-based selection, no model training.',
            'Exploratory p03 evaluation only. Absolute synchronization remains unverified. See protocol.json for the explicit heuristic.',
            '', '| Record | Strategy | Full valid/total | Full MAE | Full r | Switches |','|---|---|---:|---:|---:|---:|']
    for distance,stats in summary.items():
        sels=json.loads((out/f'p03_{distance}_selections.json').read_text())['selection']
        for strategy in ['fixed','energy','quality']:
            m=stats[f'{strategy}/full/0']
            report.append(f"| {distance} | {strategy} | {m['valid']}/{m['windows']} | {m['mae_bpm']} | {m['pearson_r']} | {sels[strategy]['switches']} |")
    report+=['','Per-block results and fixed offsets are in summary.json. Bin23 early and RX late controls are in prespecified_controls.json; no winner is chosen from their HR error.']
    (out/'REPORT.md').write_text('\n'.join(report))


if __name__=='__main__':main()
