"""Local frozen-rule follow-up: paired outcomes, selection behavior and invariance."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from src.data.evaluate_bgt60_selection import select_channels, fft_hr


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=Path,default=Path('tmp/bgt60_p03_selection_results/selection_audit'))
    parser.add_argument('--trace-dir',type=Path,default=Path('tmp/bgt60_p03_pair_v2_results/pair_audit'))
    parser.add_argument('--output-dir',type=Path,default=Path('tmp/bgt60_p03_selection_followup'))
    args=parser.parse_args();start_clock=time.perf_counter();args.output_dir.mkdir(parents=True,exist_ok=True)
    summary=json.loads((args.input_dir/'summary.json').read_text())
    findings={};report=['# Frozen selection follow-up (local uv)', '',
        'No rule changes, threshold tuning or label-based selection. p03 exploratory cases only; no new subjects.',
        'Tests: independent per-channel constant phase rotations; selection causality against future-signal modification.',
        'Paired errors use identical zero-offset valid windows. No overlapping-window significance tests.', '',
        '| Record | Strategy | Full MAE | Full r | Improved / worsened / tied windows | Selection counts |',
        '|---|---|---:|---:|---|---|']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for distance in ['0.3m','0.6m']:
        data=json.loads((args.input_dir/f'p03_{distance}_selections.json').read_text())
        rows=data['window_predictions']
        baseline={r['window_start']:r for r in rows if r['strategy']=='fixed' and r['offset_s']==0}
        findings[distance]={}
        fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
        for strategy in ['fixed','energy','quality']:
            rs=[r for r in rows if r['strategy']==strategy and r['offset_s']==0]
            valid=[r for r in rs if r['label_bpm'] is not None and r['prediction_bpm'] is not None
                   and baseline[r['window_start']]['prediction_bpm'] is not None]
            delta=np.array([abs(r['prediction_bpm']-r['label_bpm'])-abs(baseline[r['window_start']]['prediction_bpm']-r['label_bpm']) for r in valid])
            selections=data['selection'][strategy]['choices']
            counts=Counter(f"RX{r['rx']}/bin{r['bin']}" for r in selections)
            switches=[r['start'] for prev,r in zip(selections,selections[1:]) if (prev['rx'],prev['bin'])!=(r['rx'],r['bin'])]
            predicted={r['window_start']:r['prediction_bpm'] for r in rs}
            jumps=[abs(predicted[s]-predicted[s-192]) for s in switches if predicted[s] is not None and predicted[s-192] is not None]
            m=summary[distance][f'{strategy}/full/0']
            zero=summary[distance][f'{strategy}/matched/0']
            controls={str(o):{'mae_minus_zero':summary[distance][f'{strategy}/matched/{o}']['mae_bpm']-zero['mae_bpm'],
                             'r':summary[distance][f'{strategy}/matched/{o}']['pearson_r']} for o in [-60,60,-120,120]}
            findings[distance][strategy]={'paired_valid_windows':len(valid),'mean_error_change_bpm':float(delta.mean()),
                'improved':int(np.sum(delta<-.000001)),'worsened':int(np.sum(delta>.000001)),
                'tied':int(np.sum(abs(delta)<=.000001)),'selection_counts':dict(counts),
                'switches':len(switches),'prediction_jump_at_switch_median_bpm':float(np.median(jumps)) if jumps else None,
                'matched_zero_r':zero['pearson_r'],'matched_shift_controls':controls,
                'note':'Prediction jumps at switches are descriptive; overlapping-window changes and channel changes are confounded.'}
            e=findings[distance][strategy]
            report.append(f"| {distance} | {strategy} | {m['mae_bpm']:.3f} | {m['pearson_r']:.3f} | {e['improved']}/{e['worsened']}/{e['tied']} | {dict(counts)} |")
            axes[0].plot([(r['window_start']+191.5)/30 for r in rs],[r['prediction_bpm'] for r in rs],label=strategy)
            axes[1].step([(r['start']+191.5)/30 for r in selections],[r['bin'] for r in selections],where='mid',label=strategy)
            axes[2].step([(r['start']+191.5)/30 for r in selections],[r['rx'] for r in selections],where='mid',label=strategy)
        rs=[r for r in rows if r['strategy']=='fixed' and r['offset_s']==0]
        axes[0].plot([(r['window_start']+191.5)/30 for r in rs],[r['label_bpm'] for r in rs],'k--',label='Reference (shared start assumed)')
        axes[0].set(ylabel='HR (bpm)',title=f'p03 {distance}: same FFT, frozen selectors')
        axes[1].set(ylabel='Native range bin');axes[2].set(ylabel='RX (zero based)',xlabel='Window midpoint (s)',yticks=[0,1,2])
        for ax in axes:ax.legend(loc='upper right');ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(args.output_dir/f'p03_{distance}_selection.png',dpi=160);plt.close(fig)
        with np.load(args.trace_dir/f'p03_{distance}_traces.npz') as archive:
            z=archive['complex'][:,:,:,1];starts=archive['window_starts']
        original,_=select_channels(z,starts)
        rotations=np.exp(1j*np.arange(15).reshape(3,5)*.37)
        rotated=z*rotations
        modified,_=select_channels(rotated,starts)
        assert original==modified,'Constant channel phase rotation changed selectors'
        max_diff=0.
        for strategy in original:
            for s,ch in zip(starts,original[strategy]):
                rx,bi=divmod(ch,5)
                a=fft_hr(z[s:s+384,rx,bi]);b=fft_hr(rotated[s:s+384,rx,bi])
                assert (a is None)==(b is None)
                if a is not None:max_diff=max(max_diff,abs(a-b))
        assert max_diff<1e-8,'Rotation changed FFT estimates'
        cutoff=9000;prefix=starts[starts+384<=cutoff]
        altered=z.copy();altered[cutoff:]*=np.exp(1j*np.arange(len(z)-cutoff)[:,None,None])*3
        prefix_original,_=select_channels(z,prefix);prefix_altered,_=select_channels(altered,prefix)
        assert prefix_original==prefix_altered,'Future signal changed earlier choices'
        findings[distance]['invariance']={'constant_channel_rotation_choices_identical':True,
            'max_prediction_difference_bpm':max_diff,'future_modification_prefix_choices_identical':True,
            'prefix_windows':len(prefix),'cutoff_frame':cutoff}
        print(distance,'follow-up complete',flush=True)
    report+=['','## Interpretation', '',
        'A lower MAE without preserved within-record correlation is not reliable HR tracking. The quality heuristic may switch more than energy selection despite its switch penalty; this does not justify tuning the penalty on these reference labels.',
        'Do not interpret power movement as chest displacement. These two records were selected after screening; no cross-subject validation or confidence claim is made.',
        'The independent-window estimator is invariant to constant channel phase rotations, so a constant inter-channel phase offset alone does not explain changes in this frequency estimator. Time-varying channel artifacts remain possible.',
        'Raw arrays are not included in the Git summary; input hashes are in the original experiment artifacts. No model training or Kaggle submission was performed for this follow-up.']
    (args.output_dir/'FOLLOWUP.md').write_text('\n'.join(report),encoding='utf-8')
    (args.output_dir/'followup.json').write_text(json.dumps(findings,indent=2,allow_nan=False))
    timing={'seconds':time.perf_counter()-start_clock,'rule_changes':False,'training':False,
            'execution':'local uv','invariance_checks_passed':True}
    (args.output_dir/'timing.json').write_text(json.dumps(timing,indent=2))
    print(json.dumps(timing))


if __name__=='__main__':main()
