"""Freeze one finite experiment separating target definition and residual readout."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path('experiment_runs/ftu_residual_calibration');ROOT.mkdir(parents=True,exist_ok=True)
paths=['experiment_runs/ftu_reference_support_v2/polar_reference_values.csv',
       'experiment_runs/ftu_reference_support_v2/selected_support.csv',
       'experiment_runs/ftu_reference_support_v2/decision.json',
       'experiment_runs/ftu_evaluation_controls/protocol.json','audit_ftu_structure.py']
protocol=dict(date='2026-10-26',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    purpose='Window-mean HR and beyond-linear-trend residual identifiability calibration; development only',
    source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
    no_new_radar=True,no_training=True,previous_0_of_6_preserved=True,
    fs_hz=20,frames=1200,lengths_frames=[512,768],primary_length=512,hop_frames=20,
    common_windows='Exactly prior selected_support all-offset/all-control intersection; no windows shortened or reselected',
    target='Unchanged sampled full-window arithmetic mean: Y_alpha(s,L)=sum(h_alpha[k], k=s..s+L-1)/L',
    decomposition='OLS fit m(t)=a+bt on all reference samples of each development record; r(t)=piecewise-linear HR(t)-m(t); h_alpha=m+alpha*r',
    alphas=[0,.5,1,2],alpha_1='Only observed residual amplitude; other multipliers are diagnostic, not real performance',
    phase_convention='Known phase at frame boundaries: phi[k+1]-phi[k]=2*pi*h_alpha[k]/(60*20). Piecewise-constant frequency per0.05s interval. This exactly matches the existing discrete mean target. Unlike the previous experiment trapezoidal phase integration, both readouts here share this newly frozen generator.',
    ideal_readout='60*(phi[s+L]-phi[s])/(2*pi*(L/20)); underlying known phase is used even in noisy trace cases. Oracle statistic/unit check, not estimated phase or deployable algorithm.',
    trace='cos(phi[k]+initial_phase)+sigma*standard_normal[k]; no phase-extractor/ADC/RDA claim; no extrapolation of HR',
    seeds=list(range(20261026,20261058)),noise_sigma=[0,.25],
    pairing='Per record/seed same initial phase and standard-normal sequence for all alpha/noise cases; SeedSequence([seed, sorted_record_index]) preserves first3previous pairs',
    fft='Exact existing detrend/spectrum/scalar_metrics; Hann, nfft4096,45-150bpm global peak, +/-one native bin reference power',
    metrics=['per-record MAE/RMSE versus exact sampled target','paired MAE improvement over alpha0 same-readout trend','MAE improvement over known mean-trend benchmark',
             'paired residual estimate pred_alpha-pred_0 vs alpha*(Y_1-M)','raw/centered residual RMSE; centered residualR2/r/gain; target residual mean and RMS'],
    uncertainty='Per-record5th/50th/95th percentiles over32paired phase/noise realizations; conditional nuisance spread, NOT subject confidence intervals. No window bootstrap or treating seeds/overlapping windows as independent biological samples. Unknown reference/sync error remains unquantified.',
    evaluation_notes='No universal record-pass gate. Keep all6records and alpha levels including p04/p09. Compute raw metrics even when precision-limited; suppress centeredR2 only for <=1e-6bpm centered residual RMS or <5windows.',
    fft_precision_annotation='For each record/length/noise compute alpha0 centered null jitter RMS vs across-seed per-window median,95th percentile. Operational floor=max(display_grid/sqrt(12),null_jitter95). Target centered residual RMS below this floor is labelled dynamic-unevaluable-at-this-FFT-precision, not algorithm failure. Grid/sqrt12 is a reporting convention for this quantized readout, NOT native Fourier resolution or an identifiability theorem. Oracle evaluated separately.',
    stable_increment_annotation='For evaluable FFT residuals only: q05(MAE_gain_vs_paired_trend)>0 AND q05(centered_residualR2)>0 gives synthetic-stable-increment annotation; otherwise no stable evidence at this readout. No5/6 requirement or significance claim.',
    unchanged_controls='Retain all original -20/-15/-10/-5 shifts and0/1/2offsets as secondary FFT metrics for everyalpha. No old0.01threshold, control or record is removed/relaxed; this experiment does not replace old gates.',
    phase_check_tolerance_bpm=1e-9,
    exit='If exact phase-mean fails tolerance, fix implementation first. Otherwise report per-record FFT/readout and precision limits. If alpha1 lacks stable residual evidence, restrict that record/condition to window-mean/structure claims. Regardless development result, no validated real detailed-tracking claim or training/expansion without frozen independent-record validation.',
    independent_validation='All6records/subjects repeatedly examined. Repartitioning them is not independent validation. New records from genuinely untouched subjects preferred, or another dataset with traceable reference-generation and synchronization. No independent data acquired in this experiment.')
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
print('Frozen full-window target,phase convention,alpha grid,32paired seeds and finite exit; no output scores generated.')
