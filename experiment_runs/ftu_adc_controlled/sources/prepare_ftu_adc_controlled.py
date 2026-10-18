"""Freeze corrected readout timing, numerical reporting, and six-record 2x2."""
import base64,hashlib,io,json,shutil,subprocess,zipfile
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent;out=root/'experiment_runs/ftu_adc_controlled';out.mkdir(exist_ok=True);(out/'sources').mkdir(exist_ok=True)
for src,dest in [('src/data/loaders/export_all_datasets.py','current_export.py'),('src/data/loaders/ftu_loader.py','current_ftu_loader.py'),
    ('src/features/edacm.py','edacm.py'),('audit_ftu_adc_phase.py','audit_ftu_adc_phase.py'),('audit_ftu_adc_controlled.py','audit_ftu_adc_controlled.py'),
    ('prepare_ftu_adc_controlled.py','prepare_ftu_adc_controlled.py')]:shutil.copyfile(root/src,out/'sources'/dest)
for name in ['selected_records.csv','expected_rda_hashes.json','configuration_evidence.json']:
    shutil.copyfile(root/'experiment_runs/ftu_adc_phase'/name,out/name)
protocol=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),training=False,offset_search=False,
    purpose='six frozen records; candidates no-chirp-mean must pass; known failing old arms mechanism controls only',
    selection='unchanged minimum prior hash per original six training subjects',
    pri_s=57e-6,pri_basis='conditional interpretation of archived profile; p03 absent log explicitly flagged',
    chirp_indices=np.linspace(0,127,64,dtype=int).tolist(),
    synthetic_cases=[dict(id='baseline72',hz=1.2,phase_amplitude=.2,initial_phase=0),
        dict(id='offgrid_weak1',hz=1.13,phase_amplitude=.05,initial_phase=.4),
        dict(id='offgrid_weak2',hz=1.67,phase_amplitude=.02,initial_phase=1.1),
        dict(id='offgrid_weak3',hz=2.21,phase_amplitude=.01,initial_phase=2.)],
    synthetic='400frames,ADC amplitude1000,4RX/128chirps/250ADC,range12,boresight8,continuous phase int16 quantized; independent TI-port reader matches production',
    candidates=['legacy_none','full_none'],mechanism_controls=['legacy_mean','full_mean'],
    gate=dict(candidates_only=True,all_cases_required=True,min_amplitude_fraction=1e-6,max_rmse_mean_time_rad=.001,max_peak_error_bpm=3,
        mean_time='configuration chirp-time centroid; no fitted lag; additionally report chirp0 and exact coherent-observation errors'),
    precision='per condition complex64 and complex128 directDC sums; absolute error,reference norm,true relative error only above amplitude floor,and RMS error/theoretical coherent gain; no denominator floor disguised as relative error',
    layout='independent port from TI SWRA581B fig11/section9.2; verify raw word count, vendor-vs-production all frames, capture logs; p03 missing hardware log remains conditional',
    real=dict(records=6,frames=1200,calibration_frames=[0,256],
        common_physical_doppler='DC only,shifted index4 for both FFTs; nonzero indices not interchangeable',
        range_rule='full_none DC mean absolute amplitude first256frames,mean over16beams,exclude raw range0/1,argmax; fixed contiguous8bins',
        target_rule='common full_none calibration top3 energy beam/range in fixed ROI,stable index tie break; weights normalized amplitudes; same targets/weights allarms',
        windows='N256,starts0:128:896; score starts>=256,>=80% finite reference; 36 windows per arm',
        phase_floor='window target RMS>1e-6 of full_none RMS scaled by chirpcount/64; all samples nonzero finite; below floor scores undefined, not zero; retain raw complex signals and reliability flags',
        postprocessing='P2 increments,cumsum float32,exact archived detrend_linear and zscore_1d,Hann power,fixed .5-2.75Hz argmax; singles and fixed fusion',
        labels='original exported reference,frame/20+0s,no extrapolation,no lag search; HR only scoring',
        metrics='MAE,centeredRMSE,centeredr,centeredstd ratio,pooledR2,reference-near power; paired record changes; no mean per-recordR2 claims',
        invalid_control='report abstentions; compare methods on same common valid windows for each pair,also report candidate coverage; do not require known-nullcontrol valid for candidate scoring'),
    expansion='no automatic54record expansion: require both candidates valid and at least one candidate shows lower MAE AND centeredRMSE AND higher reference-near power vs legacy_mean in >=5/6 records with >=4 common fused windows/record,pooled centeredr>0 and pooled centeredRMSE below baseline; no statistical independence claim; no training')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
docs=dict(url='https://www.ti.com/lit/an/swra581b/swra581b.pdf',title='TI SWRA581B, revision October2018',access_date='2026-10-18',
    sections=['6/figure11,p10','9.2,pp16-17','9,p15 packet reorder/header removal precondition'],
    independent_basis='deviceAR1642 +DCA1000 two-lane logs select the documented2I2Q format; reader port built from vendor pseudocode,not second copy of loader',
    limitation='no hardware known-tone injection; p03 radar log unavailable; payload ordering supported by capture logs and exact file size but not proved by identical readers')
(out/'vendor_sources.json').write_text(json.dumps(docs,indent=2))
(out/'source_manifest.json').write_text(json.dumps({p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='source_manifest.json'},indent=2))
print(out)
