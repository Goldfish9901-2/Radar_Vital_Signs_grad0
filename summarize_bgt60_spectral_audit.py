"""Archive frozen spectral attribution and existing subject-baseline reaudit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,default=Path('experiment_runs/bgt60_spectral_subject_audit_2026-10-12'))
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    stages={'spectral_p03':'tmp/bgt60_p03_harmonic_audit',
            'subject_baseline':'tmp/bgt60_eight_subject_baseline',
            'peak_controls':'tmp/bgt60_peak_controls'}
    for name,path in stages.items():
        dest=out/name;dest.mkdir(exist_ok=True)
        for file in Path(path).iterdir():
            if file.is_file():shutil.copy2(file,dest/file.name)
        for file in dest.glob('*.json'):json.loads(file.read_text())
        for file in dest.glob('*.npz'):
            with zipfile.ZipFile(file) as archive:assert archive.testzip() is None
            with np.load(file) as data:assert data['power'].shape==(92,193) and np.isfinite(data['power']).all()
    baseline=json.loads((out/'subject_baseline/summary.json').read_text())
    assert len(baseline['subjects'])==8 and len(baseline['records'])==32
    assert baseline['pooled']['valid']==2912
    controls=json.loads((out/'peak_controls/summary.json').read_text())
    report=['# Spectral attribution and eight-subject baseline audit', '',
        'Local uv only; no estimator changes, training or Kaggle submission. All thresholds and candidate peak rules were fixed before computing attribution.',
        '', '## Spectral attribution', '',
        'Fixed p03 bin/RX and zero-Doppler unwrap are reused. Respiration is estimated by peak search in 0.1–0.6 Hz; HR in 0.75–2.5 Hz. Top three peaks include explicitly flagged band endpoints. Harmonic orders 2–6 and one-bin tolerance (4.6875 bpm) are fixed. Reference HR/RR only audit the radar-derived peaks; no label chooses a channel or peak.',
        '', 'At 0.6 m, 71.9% of wrong estimates match radar-derived respiratory multiples, but 68.8% also match a half-order-shifted comb. Only 31.2% match multiples derived from reference RR. These coincidences do not establish a respiratory origin; mask coverage and coarse respiration estimates are documented.',
        'At 0.3 m, 83.6% of accurate windows match estimated harmonics, and 82.2% have reference HR near those harmonics. Blind harmonic removal could suppress useful cardiac components. In 0.6 m wrong windows, reference HR is near a top-three candidate in 53.1%, but that is label-assisted diagnosis, not oracle estimator accuracy.',
        '', '## Candidate-peak negative controls', '',
        '| Record | Matched windows | Top3 HR proximity at 0 s | -60 s | +60 s | -120 s | +120 s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for distance in ['0.3m','0.6m']:
        entry=controls[distance]['candidate_controls']['all_matched']
        report.append(f"| {distance} | {entry['windows']} | "+' | '.join(f"{entry['hit_rates'][str(o)]:.3f}" for o in [0,-60,60,-120,120])+' |')
    report+=['', '0.6 m top3 hit rate is unchanged at +60 s, so candidate availability alone is weak evidence for synchronous HR. The nonoverlapping matched subset also lacks a zero-offset advantage at 0.6 m. Slow HR and a coarse, tolerant three-peak comparison are plausible confounds, not proven mechanisms.',
        'The nonoverlapping and reference-RR-agreement strata are descriptive checks. They do not authorize quality-based window removal; RR agreement uses reference labels. No significance tests treat overlapping windows as independent.',
        '', '## Eight-subject frozen baseline', '',
        'The existing 32-record native Range FFT/label-free local peak/whole-record energy RX/zero-Doppler unwrap output already covers all eight subjects. This stage reaudits that actual output, rather than rerunning ADC or presenting it as new independent data. No rule was extended from p03 to an untouched cohort.',
        '', '| Subject | Valid windows | FFT MAE | Other-seven-subject mean MAE | Pearson r |','|---|---:|---:|---:|---:|']
    for subject,s in baseline['subjects'].items():report.append(f"| {subject} | {s['valid']} | {s['mae_bpm']:.3f} | {s['constant_mae_bpm']:.3f} | {s['pearson_r']:.3f} |")
    report+=[ '',f"Subject macro FFT MAE: {baseline['subject_macro_mae_bpm']:.3f} bpm; constant: {baseline['subject_macro_constant_mae_bpm']:.3f} bpm. Mean within-record r: {baseline['record_r_arithmetic_mean']:.3f} (arithmetic descriptive mean).",
        'The constant uses only reference HR from other seven subjects. Each test subject has 364 valid out of 368 overlapping windows. Full-record totals are 2,912 valid / 2,944 windows, distinct from earlier matched-interior scores.',
        '', '## Evidence and reproduction', '',
        'All source hashes, per-window candidates and harmonic energies, candidate control hits, per-record metrics and measured local timings are retained. The two small spectral arrays contain window PSDs, not raw ADC. The large source complex archives remain local and in the prior Kaggle paired output.',
        'Three synthetic spectral checks pass. Fixed HR predictions and reference labels exactly match the prior paired audit. JSON, spectral CRC/shape/finite values and 32-record/eight-subject coverage are checked.',
        '', '```cmd', 'setup_local_uv.cmd',
        'uv run --locked --no-sync python validate_bgt60_spectral_attribution.py',
        'uv run --locked --no-sync python analyze_bgt60_spectral_attribution.py',
        'uv run --locked --no-sync python analyze_bgt60_subject_baseline.py',
        'uv run --locked --no-sync python analyze_bgt60_peak_controls.py',
        'uv run --locked --no-sync python summarize_bgt60_spectral_audit.py --output-dir tmp/spectral_archive_reproduction', '```', '',
        'Spectral commands require paired NPZ files at the default input paths. The subject reaudit reads the already committed compressed direct-phase output. Script options permit alternate paths.',
        '', '## Decision and limits', '',
        'Do not introduce large decomposition models or tune harmonic suppression using p03 based on this diagnostic alone. Respiratory-harmonic attribution remains unproven; broad constant-baseline failure remains. No CycleFormer or domain adaptation training resumed.',
        'p03 is exploratory and all eight subjects were examined previously. Neither stage is independent validation. The assumed 30 Hz frame rate and shared start remain unverified; the reference is 1 Hz HR/RR, not raw ECG.']
    (out/'README.md').write_text('\n'.join(report),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(8)
    ax.bar(x-.2,[v['mae_bpm'] for v in baseline['subjects'].values()],.4,label='Fixed phase FFT')
    ax.bar(x+.2,[v['constant_mae_bpm'] for v in baseline['subjects'].values()],.4,label='Other-seven-subject mean')
    ax.set(xlabel='Subject',ylabel='Full-record MAE (bpm)',xticks=x,xticklabels=list(baseline['subjects']))
    ax.legend();fig.tight_layout();fig.savefig(out/'subject_baseline.png',dpi=160);plt.close(fig)
    (out/'validation.json').write_text(json.dumps({'synthetic_tests':3,'baseline_predictions_unchanged':True,
        'spectra_crc_shape_finite':True,'subjects':8,'records':32,'valid_windows':2912,'training':False},indent=2))
    files=[f for f in out.rglob('*') if f.is_file() and f.name!='artifact_manifest.json']
    manifest=[{'file':f.relative_to(out).as_posix(),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(files)]
    (out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Archived',len(files),'validated artifacts')


if __name__=='__main__':main()
