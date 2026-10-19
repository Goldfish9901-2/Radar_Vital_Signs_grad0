"""Freeze six-record archived Range/Beam diagnostics before viewing new spectra."""
import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path('experiment_runs/ftu_frequency_structure');root.mkdir(parents=True,exist_ok=True)
source=Path('experiment_runs/ftu_adc_controlled')
for name in ['selected_records.csv','selection.json']:shutil.copyfile(source/name,root/name)
manifest=json.loads((source/'artifact_manifest.json').read_text())
files=['signals/'+p.name for p in sorted((source/'signals').glob('*.npz'))]+['reliability_masks.npz','selection.json','selected_records.csv']
hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files}
assert len([x for x in files if x.endswith('.npz') and x.startswith('signals/')])==6
for name,h in hashes.items():assert h==manifest[name],name
(root/'input_hashes.json').write_text(json.dumps(hashes,indent=2))
protocol=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source=str(source),records=6,
    training=False,new_export=False,new_samples=False,oracle=False,
    scope='saved DopplerDC only,16beams x8 fixed range ROI; no claims about unexported spatial region or nonzero Doppler',
    band_bpm=[45,150],fs_hz=20,calibration_frames=[0,256],
    variants=['legacy_mean','full_mean','legacy_none','full_none'],
    scoring_variants=['legacy_mean','legacy_none','full_none'],candidate_variants=['legacy_none','full_none'],
    cell_selection='for each saved range,highest mean absolute amplitude beam in saved full_none calibration_energy; tie lowest beam; same8cells all variants',
    full_map='all128 cells quality and mean spectrum maps,descriptive only; no HR-based selection or best-cell result',
    phase='continuous complex128 conjugate principal increments,P2; cumulative separately in each valid STFT segment,linear detrend; no VMD,no HR-guided filters',
    amplitude='log amplitude fluctuation,linear detrend,identical STFT; separate diagnostic,not replacement phase performance',
    reliability=dict(numeric_floor_ratio=1e-6,baseline='samecell full_none RMS scaled by chirpcount/64',
        dropout_threshold='.01 times full_none first256-frame median amplitude scaled by chirpcount/64; shared physical threshold for allarms',
        segment_valid='all finite nonzero complex samples,RMS above numericfloor,<=1percent amplitude dropout; no gaps imputed',
        original_masks='preserve and recompute original8window x3target masks independently; new128cell masks separately labelled'),
    stft=dict(lengths_frames=[512,768],hop_frames=20,nfft=4096,hann=True,
        native_resolution_bpm=[2.34375,1.5625],display_grid_bpm=.29296875,
        scoring='segments entirely after calibration,centers from length//2 to1200-length//2 step20,valid reference at every queried sample',
        amplitude_phase='normalized power within45-150bpm; continuous observation windows,not concatenated standardized windows'),
    time=dict(default_offset_s=0,offsets_s=[0,1,2],definition='reference query = frame/20 +offset; not measured synchronization',
        negative_label_shifts_s=[-20,-15,15,20],negative='circular shift reference queries within physical60sec record; preserves label marginal but introduces artificial wrap',
        labels='average interpolated reference across same STFT sample support,no extrapolation',
        common_windows='perlength/perrecord intersect finite HR support for all3offsets and4negative shifts; same time centers for allcells andarms; arm reliability coverage reported separately',
        no_best_offset=True),
    scoring='panel equal-cell mean over fixed8cells; report percell and original fixed-fusion separately; no selection by score',
    reference_power='normalized45-150bpm power within +/-one native Fourier bin around window-mean HR; HR only offline scoring',
    peak='global maximum in45-150bpm on display grid,descriptive; zero padding does not increase resolving power',
    diagnostics=['raw amplitude/dropouts','phase increment percentiles and near-pi fraction','relative amplitude and phase spectra','local spectral stability','reference near power','ridge/HR distance','negative label separations vs native resolution'],
    expansion=dict(primary_length=512,independent_unit='record,not overlapping window',
        record_pass='for a candidate: validpanelcoverage>=.9,>=.6 panelcell-time peaks within5bpm of true HR,mean referencepower exceeds EACH4negative controls by>=.01,MAE smaller than EACH control; criteria must hold at all3offsets; no bestoffset',
        dataset_pass='same fixed candidate passes>=5of6records; only warrants preregistration of54record validation,no automatic extension',
        limitation='time-shift controls may be weak when HR variation smaller than resolution; record separations explicitly reported; failure is not proof of absent heartbeat',
        continuation='until criterion met keep training,VMD incremental experiments,and domain adaptation paused'))
(root/'protocol.json').write_text(json.dumps(protocol,indent=2))
print('Frozen6records,10verifiedinputs; no spectra inspected')
