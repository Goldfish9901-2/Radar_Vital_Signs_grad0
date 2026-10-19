"""Create the six-record report from saved diagnostic artifacts, no selection."""
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd

ROOT=Path('experiment_runs/ftu_frequency_structure')


def table(frame, cols, digits=3):
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,row in frame.iterrows():
        vals=[]
        for c in cols:
            v=row[c]
            vals.append('NA' if pd.isna(v) else
                        (f'{v:.{digits}g}' if v != 0 and abs(v) < 10**(-digits) else f'{v:.{digits}f}')
                        if isinstance(v,float) else str(v))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)


def main():
    m=pd.read_csv(ROOT/'record_metrics.csv');d=pd.read_csv(ROOT/'descriptive_true_only.csv')
    g=pd.read_csv(ROOT/'gate.csv');s=pd.read_csv(ROOT/'label_separation.csv');q=pd.read_csv(ROOT/'cell_quality.csv')
    ex=json.loads((ROOT/'execution.json').read_text());v=json.loads((ROOT/'independent_validation.json').read_text())
    base=d[(d.length==512)&(d.kind=='panel')&(d.offset==0)].copy()
    base['subject']=base.record.str[:3]
    pivot=base.pivot(index='subject',columns='arm',values='mae').reset_index()
    counts=g.groupby('arm').agg(evaluable=('evaluable',lambda x:int(x.sum()/3)),passes=('record_pass',lambda x:int(x.sum()/3))).reset_index()
    gate_detail=g[g.offset==0].copy();gate_detail['subject']=gate_detail.record.str[:3]
    ctrl=m[(m.length==512)&(m.kind=='panel')&(m.offset==0)]
    comp=[]
    for (record,arm),group in ctrl.groupby(['record','arm']):
        true=group[group['shift']==0].iloc[0];wrong=group[group['shift']!=0]
        if not true.common_windows:continue
        comp.append(dict(subject=record[:3],arm=arm,windows=int(true.common_windows),coverage=true.coverage,
            mae=true.mae,within5=true.within5,near_power=true.near_power,
            min_power_advantage=float((true.near_power-wrong.near_power).min()),
            min_mae_advantage=float((wrong.mae-true.mae).min())))
    comp=pd.DataFrame(comp)
    summary_input=d[(d.length==512)&(d.offset==0)]
    df_summary=summary_input.groupby(['arm','kind'])[['mae','centered_rmse','r','std_ratio','within5','coverage','near_power']].mean().reset_index()
    df_summary=df_summary.merge(summary_input.groupby(['arm','kind']).mae.count().rename('records_with_mae').reset_index(),on=['arm','kind'])
    qsum=q.groupby('arm')[['amplitude_ratio','dropout','near_pi_fraction','abs_increment_p99']].median().reset_index()
    sep=s.dropna(subset=['mean_absolute_separation']).groupby(['length','offset'])[['mean_absolute_separation','max_absolute_separation','fraction_within_native']].mean().reset_index()
    report=f'''# Six-record archived Range/Beam frequency structure diagnostic

The archived DC-only region does not establish the prespecified 5/6-record evidence needed to expand to54 records. Neither candidate passes any record. Three records are **unevaluable under the frozen common-control support**, rather than demonstrated correspondence failures. Training, VMD changes and adaptation remain paused. This is a six-record diagnostic, not a population estimate or proof that heartbeat information is absent.

## Scope and protocol

Protocol committed before new spectra: `bb786e7`; support addendum committed before the complete rerun: `c4a4091`. The first attempt stopped on an empty-support reduction; it did not complete a results report. The disclosed addition is correct-label-only descriptive scoring, not a replacement gate.

Only existing archived complex DC sequences are used: six frozen training-subject records,16 beams ×8 saved ranges,1200 frames at assumed20Hz. No ADC reload/export, additional subjects, model fitting, HR-guided cell/peak selection or offset search. This does not evaluate nonzero Doppler or ranges outside the saved ROI. Calibration is first256 frames; each saved range uses the highest calibration `full_none` mean-amplitude beam, same eight identities across arms. Original three targets and weights are retained for separate fusion analysis. No oracle is computed.

## Time support limitation

p03/p04/p05 references end at59s. Circular wrong-label queries modulo60 enter59--60s, so intersecting all controls and all +0/+1/+2s offsets leaves zero common windows at both lengths. No extrapolation or imputation is used. These records retain spectra, quality checks and correct-label-only descriptive scores. They cannot satisfy the gate. Other records have common counts shown below. The controls' artificial circular wrap is itself a diagnostic limitation; this is not a permutation test or causal proof.

## Frozen expansion gate

Primary512 frames: coverage≥.9; ≥.6 fixed-panel peaks within5bpm; reference-near power ≥.01 above **each** wrong-label control; MAE below **each** control. All criteria at all three offsets, and the same candidate in≥5/6 records. No best offset/cell is chosen.

{table(counts,['arm','evaluable','passes'])}

## Six-record descriptive results, correct labels only

512frames, default +0s; MAE in bpm, eight fixed cells equally weighted. These windows intersect the three correct-label offsets only, so are **not** the control-comparison set. See `descriptive_true_only.csv` for all offsets, both lengths, panel and original fixed fusion. The narrowed band and longer windows differ from the prior256-frame audit; these are not a one-factor comparison with its MAE.

{table(pivot,['subject','legacy_mean','legacy_none','full_none'])}

Record-equal means over available scores. Candidates have six scored records; legacy_mean has only four because p03/p04 are unreliable. Coverage averages include all six. These unequal available-record means are **not a paired estimate of improvement**; correlations are descriptive, not population inference.

{table(df_summary,['arm','kind','records_with_mae','mae','centered_rmse','r','std_ratio','within5','coverage','near_power'])}

Both none candidates cover the entire fixed panel in all six records. Their smaller observed phase increments and improved peak MAE in some records (notably p05/p07) do not establish recovery of HR tracking: within5 fractions remain below the frozen .6 threshold in the evaluable records, and power gains over all wrong-label controls remain below .01. legacy_mean has no reliable panel score for p03/p04; NA is not a large error value or a zero.

## Correct vs shifted labels on identical support

Primary512frames, +0s. `min_power_advantage` is true power minus the largest control power; `min_mae_advantage` is the smallest control MAE minus true MAE. Positive values favor the correct labels; the power gate additionally requires≥.01. All +1/+2s comparisons are saved in `record_metrics.csv`; gate decisions use all three, without choosing the best.

{table(comp,['subject','arm','windows','coverage','mae','within5','near_power','min_power_advantage','min_mae_advantage'])}

Mean absolute true/control label separation, averaged over available record/control combinations; compare with native Fourier bin spacings2.34375/1.5625bpm:

{table(sep,['length','offset','mean_absolute_separation','max_absolute_separation','fraction_within_native'])}

Long-window averages and low reference dynamics can make shifted labels weak controls. A visually nearby ridge alone is insufficient. A failed or unevaluable gate cannot establish absence of weaker HR information.

## Amplitude and phase reliability

Across all6×128 cells, median quality diagnostics below. `amplitude_ratio` is relative to the same full_none cell, correcting the8/64 versus64/64 coherent DC scaling. Dropout is below1% of the scaled full_none calibration median. `near_pi_fraction` concerns observed P2 increments, not measured chest displacement. Raw quality numbers are retained even when phase scoring is rejected.

{table(qsum,['arm','amplitude_ratio','dropout','near_pi_fraction','abs_increment_p99'],6)}

full_mean is retained as a numerical-floor mechanism control, not given physiological phase scores. Segment reliability requires finite/nonzero observations, RMS above1e-6 of the scaled reference and≤1% dropout. Invalid segments remainNaN; expected empty-spectrum mean warnings denote unscorable cells, not filled data. Original8×3 target masks were independently reproduced for all24 record/arm combinations. New masks are separately archived.

## Spectral and metric definitions

P2 uses complex128 conjugate principal increments; cumulative phase is detrended independently per segment. Phase origin is removed by detrending, never treated as a discontinuity. Original fusion sums the three detrended phases with unchanged weights; final scalar normalization does not change normalized power. Log-amplitude spectra are separate diagnostics. No HR-guided filtering or peak selection.

Hann windows512/768frames (25.6/38.4s), hop1s, FFT4096 for display. Native bin spacings2.34375/1.5625bpm; display grid0.29296875bpm **does not increase resolving power**, and Hann broadens spectral response. Primary scoring excludes all segments overlapping initial12.8s calibration. Reference labels are mean interpolated HR over the exact segment support, no extrapolation. Positive offset queries reference at radar time+offset; these are assumptions, not measured synchronization.

Power is normalized within45--150bpm. Reference-near fraction integrates ±one native bin. Global band maximum is a descriptive readout, not an assumed heartbeat peak. Primary MAE/near power/within5 use equal-cell means over valid windows; coverage and missingness are explicit. Dynamic RMSE/r/std ratio pool time deviations after removing each cell's temporal mean; they do not imply absolute HR calibration. `ridge_step` reports median adjacent valid peak change; smoothness alone is not correspondence evidence. Overlapping windows are not independent samples; no window-count significance tests are performed.

## Figures and archived evidence

Each record has four PNGs in `figures/`: phase spectra for three arms across the fixed eight cells; full_none phase vs log-amplitude spectra; descriptive all-beam range-frequency maps and128-cell reliability; raw full_none amplitude/P2 increments/unwrapped phase. White curves are referenceHR at+0s, red raw-amplitude lines mark dropout thresholds. Spectral colors are within-band normalized dB (−45 to−10); blank cells are unreliable. Mean maps include the continuous recording, whereas numerical scores use post-calibration windows.

`spectra/*.npz` stores both lengths, frequencies, segment starts, all reference/control labels, common-support masks, fixed cells, original fusion,128-cell validity and mean spectra. `cell_metrics.csv`, `cell_quality.csv`, `panel_selection.csv`, `label_separation.csv` and `gate.csv` provide underlying values. Input archives are referenced by10 verified hashes, not duplicated.

## Execution and independent checks

Local uv diagnostic elapsed **{ex['elapsed_seconds']:.2f}s**; independent validation and raw-trace plots **{v['elapsed_seconds']:.2f}s**. No training/Kaggle run. Independent unwrap+SciPy linear detrend agrees with conjugate-increment spectra: maximum absolute normalized-power difference **{v['spectrum_max_abs_error']:.3g}**. Checked {v['reliability_mask_checks']} segment reliability masks, all saved labels/common supports, {v['primary_metric_checks']} primary metric values, and all gate decisions. Source scripts, hashes and execution versions are archived. The failed initial run is documented above; successful-run timing is in `execution.json`.

## Decision

Do not expand to54 records or restart model training on this evidence. The archived DC region has not shown the required stable reference correspondence, and the frozen shifted-control comparison is unavailable in three records. A future protocol must explicitly solve the finite-reference-support control design before claiming a six-record negative-control conclusion. Continue upstream coherence/complex-signal interpretation and radar/reference time-axis auditing; no conclusion here treats the ADC candidates as a repaired pipeline.
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    (ROOT/'sources').mkdir(exist_ok=True)
    for name in ['prepare_ftu_structure.py','audit_ftu_structure.py','verify_ftu_structure.py','summarize_ftu_structure.py']:
        shutil.copyfile(name,ROOT/'sources'/name)
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}
    (ROOT/'artifact_manifest.json').write_text(json.dumps(hashes,indent=2))
    print(table(counts,['arm','evaluable','passes']))
    print(table(pivot,['subject','legacy_mean','legacy_none','full_none']))


if __name__=='__main__':main()
