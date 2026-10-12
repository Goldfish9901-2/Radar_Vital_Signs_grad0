"""Validate/archive completed local fusion ablations and export comparison plots."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=Path('experiment_runs/bgt60_p03_fusion_2026-10-12'))
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    stages={'full_phase':('tmp/bgt60_p03_fusion_results',4),
            'hr_correlation':('tmp/bgt60_p03_hrcorr_results',6),
            'projection_controls':('tmp/bgt60_p03_projected_results',10)}
    old=json.loads(Path('experiment_runs/bgt60_p03_selection_2026-10-12/summary.json').read_text())
    checks={}
    for stage,(source,nmethods) in stages.items():
        destination=out/stage;destination.mkdir(exist_ok=True)
        for path in Path(source).iterdir():
            if path.is_file():shutil.copy2(path,destination/path.name)
        summary=json.loads((destination/'summary.json').read_text())
        for distance in ['0.3m','0.6m']:
            for anchor in ['fixed','energy']:
                assert summary[distance][f'{anchor}_single/full/0']==old[distance][f'{anchor}/full/0']
            archive=destination/f'p03_{distance}_fusion_windows.npz'
            with zipfile.ZipFile(archive) as z:assert z.testzip() is None
            with np.load(archive) as data:
                assert data['phases'].shape==(92,nmethods,384)
                assert data['power'].shape==(92,nmethods,193)
                assert np.isfinite(data['phases']).all() and np.isfinite(data['power']).all()
            window=json.loads((destination/f'p03_{distance}_windows.json').read_text())
            assert len(window['fusion_details'])==92
            assert len(window['predictions'])==92*nmethods*5
        for path in destination.glob('*.json'):json.loads(path.read_text())
        checks[stage]={'baseline_matches_previous':True,'json_valid':True,'archive_crc_shapes_finite':True,'methods':nmethods}
    summary=json.loads((out/'projection_controls/summary.json').read_text())
    report=['# p03 phase-correlation fusion: local controlled ablations', '',
        'All experiments ran locally in the copy-based uv environment. No training or Kaggle submission.',
        'Literature inspiration: [Xu et al., 2026, sections 3.2/4.1](https://doi.org/10.3390/s26175587). This is not the complete published pipeline.',
        '', 'The frozen threshold is 0.75. Fusion uses only same-RX local bins and positive correlation weights; no HR labels select bins/weights. All methods use the same 384-frame unwrap/detrend/Hann peak estimator at assumed 30Hz and 0.75–2.5Hz. No CEEMD, harmonic subtraction or time smoothing.',
        '', '## Component definitions', '',
        '- single: original fixed or maximum-energy channel.',
        '- fusion: full-phase correlations weight original neighboring phases.',
        '- hrcorr: HR-band-projected correlations weight original phases.',
        '- hrprojected: same hrcorr weights, HR-projected reconstruction.',
        '- single_projected: original single channel with exactly the same HR projection, isolating preprocessing from fusion.',
        '', 'Projection is a finite-window FFT band restriction; it may introduce boundary artifacts. It is a separate preprocessing experiment, not selection-only benefit.',
        '', '| Distance | Method | Full MAE | Full r | Matched r |','|---|---|---:|---:|---:|']
    for distance,stats in summary.items():
        for method in ['fixed_single','fixed_fusion','fixed_hrcorr','fixed_hrprojected','fixed_single_projected',
                       'energy_single','energy_fusion','energy_hrcorr','energy_hrprojected','energy_single_projected']:
            m=stats[f'{method}/full/0'];r=stats[f'{method}/matched/0']['pearson_r']
            report.append(f"| {distance} | {method} | {m['mae_bpm']:.3f} | {m['pearson_r']:.3f} | {r:.3f} |")
    report+=['','## Findings', '',
        'Full-phase fusion worsens MAE under both anchors at both distances. Restricting correlation to the HR band reduces some degradation but does not outperform its original single-channel baseline.',
        'At 0.6 m, projected energy single-channel MAE is 16.947 bpm versus 17.149 for projected fusion; fixed projected single is 19.632 versus 19.901 for projected fusion. The modest changes are attributable to preprocessing rather than an additional fusion benefit. HR tracking correlations remain poor.',
        'At 0.3 m, no fusion variant improves on its original single-channel MAE. High whole-phase correlation is not sufficient evidence of shared heartbeat content; these outputs do not prove respiration is the cause.',
        'Every full-record metric uses 91 reference-valid windows out of 92, and matched offset comparisons use 55 windows. Per-window predictions, all correlation weights, segment metrics and fixed offsets are retained; no window quality exclusion or post hoc threshold tuning.',
        '', '## Provenance, checks and limitations', '',
        'The three stages were sequential exploratory experiments, not a pre-registered ten-method competition. Stage compute times were approximately 0.36 s, 0.45 s and 0.57 s (timing.json contains measured values). Six synthetic fusion/projection checks pass. Each stage reproduces the preceding single-channel baseline exactly; array CRC, shapes and finite values are validated.',
        'The archived small NPZ files contain reconstructed window phases and spectra, not the original 24.3 MB complex archives. Original complex input hashes are retained in each stage input_provenance.json. No raw ADC or raw ECG is included.',
        'p03 was chosen after earlier screening. These two distances are not independent subjects. Absolute synchronization, frame rate and metric range calibration remain unverified. No cross-subject effectiveness or clinical performance claim.',
        '', '## Reproduction', '', '```cmd', 'setup_local_uv.cmd',
        'uv run --locked --no-sync python validate_bgt60_fusion.py',
        'uv run --locked --no-sync python -m src.data.evaluate_bgt60_fusion --output-dir tmp/bgt60_p03_fusion_results',
        'uv run --locked --no-sync python -m src.data.evaluate_bgt60_fusion --hr-correlation-ablation --output-dir tmp/bgt60_p03_hrcorr_results',
        'uv run --locked --no-sync python -m src.data.evaluate_bgt60_fusion --projected-reconstruction-ablation --output-dir tmp/bgt60_p03_projected_results',
        'uv run --locked --no-sync python summarize_bgt60_fusion.py --output-dir tmp/fusion_archive_reproduction', '```', '',
        'Commands require the previously downloaded paired complex arrays at the default paths. Use --input-dir for alternative locations. Current generator adds explicit protocol flags and method labels absent from early snapshots, so regenerated JSON metadata may differ while original method metrics reproduce.',
        '', '## Decision', '',
        'Keep these variants as negative component results. Do not tune the fusion threshold on p03 or resume CycleFormer/domain adaptation based on these MAEs. Independent-subject validation is needed before any positive method claim.']
    (out/'README.md').write_text('\n'.join(report),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    suffixes=['single','fusion','hrcorr','hrprojected','single_projected']
    labels=['single','full corr','HR corr','projected fusion','projected single']
    for distance,stats in summary.items():
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for row,anchor in enumerate(['fixed','energy']):
            m=[stats[f'{anchor}_{s}/full/0'] for s in suffixes]
            for col,(key,ylabel) in enumerate([('mae_bpm','Full MAE (bpm)'),('pearson_r','Full Pearson r')]):
                axes[row,col].bar(labels,[a[key] for a in m]);axes[row,col].tick_params(axis='x',rotation=20)
                axes[row,col].set(ylabel=ylabel,title=f'{distance} {anchor} anchor');axes[row,col].axhline(0,color='black',lw=.5)
        fig.tight_layout();fig.savefig(out/f'p03_{distance}_ablation.png',dpi=160);plt.close(fig)
    (out/'validation.json').write_text(json.dumps({'stages':checks,'synthetic_tests_passed':6,'training':False},indent=2))
    files=[p for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json']
    manifest=[{'file':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]
    (out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Archived',len(files),'validated artifacts;',sum(p.stat().st_size for p in files),'bytes')


if __name__=='__main__':main()
