"""Freeze protocol/six hash-selected records and bundle one raw provenance job."""
import base64,hashlib,io,json,shutil,subprocess,zipfile
from pathlib import Path
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent;out=root/'experiment_runs/ftu_adc_phase';out.mkdir(exist_ok=True)
(out/'sources').mkdir(exist_ok=True);(out/'config_evidence').mkdir(exist_ok=True)
for src,dest in [('src/data/loaders/export_all_datasets.py','current_export.py'),('src/data/loaders/ftu_loader.py','current_ftu_loader.py'),
    ('audit/vs_code/Radar_Vital_Signs_grad0/src/data/loaders/export_all_datasets.py','archived_export.py'),
    ('audit/vs_code/Radar_Vital_Signs_grad0/src/data/loaders/ftu_loader.py','archived_ftu_loader.py'),
    ('audit_ftu_adc_phase.py','audit_ftu_adc_phase.py'),('prepare_ftu_adc_phase.py','prepare_ftu_adc_phase.py')]:
    shutil.copyfile(root/src,out/'sources'/dest)
selected=pd.read_csv(root/'experiment_runs/ftu_target_audit/selected_records.csv').sort_values('selection_hash').groupby('group_key').first().reset_index()
assert len(selected)==6
selected.to_csv(out/'selected_records.csv',index=False)
shutil.copyfile(root/'experiment_runs/ftu_target_audit/source_hashes.json',out/'expected_rda_hashes.json')
logs=json.loads((root/'tmp/ftu_p0_result/ftu_p0/radar_logs.json').read_text())
audit=pd.read_csv(root/'experiment_runs/ftu_p0_audit_2026-10-14/raw_timing_audit.csv')
evidence=[]
for row in selected.itertuples():
    a=audit[audit.sample_tag==row.sample_tag].iloc[0];h=a.radar_log_sha256
    if pd.isna(h):evidence.append(dict(sample_tag=row.sample_tag,log_available=False));continue
    log=next(x for x in logs if x['sha256']==h)
    p=root/'tmp/ftu_p0_result/ftu_p0/raw_time_evidence'/log['archived_file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==h
    text=p.read_text(errors='replace')
    names=['ProfileConfig','ChannelConfig','AdcOutConfig','DataFmtConfig','ChirpConfig','FrameConfig','LaneConfig','LvdsLaneConfig']
    last={name:[line for line in text.splitlines() if 'API:'+name+',' in line][-1:] for name in names}
    evidence.append(dict(sample_tag=row.sample_tag,log_available=True,source=log['source'],sha256=h,last_config_lines=last))
    shutil.copyfile(p,out/'config_evidence'/f'{row.sample_tag}_data_LogFile.txt')
(out/'configuration_evidence.json').write_text(json.dumps(evidence,indent=2))
protocol=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),training=False,
    purpose='phase-fidelity gate and empirical historic-source reproduction; do not claim historical command from current defaults',
    selected_records='minimum existing selection_hash per original training subject; labels/errors not consulted',
    provenance_record=selected.iloc[0].sample_tag,
    provenance='all1200 frames; old archived and current legacy conversion, both unpackers; full raw hash; match global ROI and existing RDA hash',
    variants=dict(legacy_mean=['legacy','chirp_mean'],legacy_none=['legacy','none'],full_mean=['full_fft_crop','chirp_mean'],full_none=['full_fft_crop','none']),
    chirp_indices=np.linspace(0,127,64,dtype=np.int32).tolist(),adc_samples=250,rx=4,frame_hz=20,chirp_count=128,
    pri_s=57e-6,pri_basis='available ProfileConfig idle700/ramp5000 in 10ns units; conditional on API-unit interpretation; p03 log absent',
    tx_rx='archive logs ChannelConfig 3,15; ChirpConfig tx-mask3; fourRX, bothTX enabled; do not assume TDM virtual8-element array',
    iq_layout='declared loader chirp-major RX blocks, I1 I2 Q1 Q2; roundtrip only checks declared layout, vendor confirmation remains open',
    synthetic=dict(frames=400,hr_hz=1.2,phase_amplitude_rad=.2,target_range_bin=12,angle_bin=8,doppler_hz=0,amplitude=1000,
        cases=['frame_constant','continuous_motion withinchirp using57usPRI'],quantization='int16 nearest, declared IQ pack then actual loader unpack',
        validity='amplitude >1e-6 of analytic no-clutter coherent gain at every frame; otherwise phase undefined',
        thresholds=dict(phase_rmse_rad=.01,peak_error_bpm=3),delay='none; compare phase against chirp0 phase; report integration-time bias'),
    coordinates='nonuniform legacy indices forbid exact ordinary FFT Hz axis; compare only common physical DC bin4 and same beam/range; save exact-time frequency response kernels for other bins',
    downstream='P2 principal increments, cumulative; synthetic phase fidelity before detrend/normalization; fixed .5-2.75Hz Hann peak',
    stop='any synthetic arm loses target amplitude or phase/HR gate => no real HR ablation; provenance-only raw comparison still allowed; no54-record expansion or training',
    future_real='only after gate passes: six selected records, common initial256-frame no-label calibration ROI/beam, same scored starts>=256, P2 identical detrend/zscore/fixed peak; no HR-driven target/offset')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
shutil.copyfile(root/'tmp/ftu_rda_provenance/dataset-metadata.json',out/'dataset-metadata.json')
files={p.relative_to(out).as_posix():p.read_bytes() for p in out.rglob('*') if p.is_file() and p.parts[-2] in ['sources','config_evidence']}
for name in ['selected_records.csv','expected_rda_hashes.json','configuration_evidence.json','protocol.json']:files[name]=(out/name).read_bytes()
bundle=io.BytesIO()
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in files.items():z.writestr(name,data)
stage=root/'tmp/ftu_adc_phase_upload';stage.mkdir(exist_ok=True)
runner='''import base64,io,os,sys,zipfile,shutil,json,time
from pathlib import Path
code=Path('/kaggle/working/code');code.mkdir()
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:z.extractall(code)
sys.path.insert(0,str(code/'sources'))
from audit_ftu_adc_phase import provenance
out=Path('/kaggle/working/ftu_adc_provenance')
provenance(Path('/kaggle/input'),code,out)
shutil.make_archive('/kaggle/working/000_ftu_adc_provenance','zip',out)
'''.replace('BUNDLE',repr(base64.b64encode(bundle.getvalue()).decode()))
(stage/'run.py').write_text(runner)
(stage/'kernel-metadata.json').write_text(json.dumps(dict(id='goldfish9901/ftu-adc-phase-provenance',title='FTU ADC Phase Provenance',code_file='run.py',
    language='python',kernel_type='script',is_private=True,enable_gpu=False,enable_internet=False,
    dataset_sources=['goldfish9901/ftu-rda-exports','goldfish9901/ftu-part1'],kernel_sources=[],competition_sources=[],model_sources=[]),indent=2))
(out/'source_manifest.json').write_text(json.dumps({p:hashlib.sha256(b).hexdigest() for p,b in files.items()},indent=2))
print(stage)
