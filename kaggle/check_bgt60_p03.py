import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import argparse
import gzip

parser = argparse.ArgumentParser(description='Check p03 records without model training')
parser.add_argument('--input-dir', type=Path, default=Path('experiment_runs/bgt60_direct_phase_2026-10-10'))
parser.add_argument('--range-profile-dir', type=Path, default=Path('tmp/bgt60_adc_diagnostics_results/diagnostics'))
args = parser.parse_args()
p=args.input_dir
source=p/'windows.jsonl'
stream=source.open() if source.exists() else gzip.open(p/'windows.jsonl.gz', 'rt', encoding='utf-8')
with stream:
    rows=[json.loads(line) for line in stream if '"sample": "p03_' in line]
records=json.loads((p/'records.json').read_text())
result={}
def stats(rs):
    rs=[r for r in rs if r['label_bpm'] is not None and r['fft_hr_bpm'] is not None]
    if not rs:return {'windows':0}
    y=np.array([r['label_bpm'] for r in rs]); x=np.array([r['fft_hr_bpm'] for r in rs])
    return {'windows':len(rs),'mae':float(abs(x-y).mean()),'r':float(np.corrcoef(x,y)[0,1]) if min(x.std(),y.std())>.01 else None,
            'reference_mean':float(y.mean()),'reference_std':float(y.std()),'prediction_mean':float(x.mean()),'prediction_std':float(x.std()),
            'within_5_bpm_fraction':float(np.mean(abs(x-y)<=5)), 'low_edge_fraction':float(np.mean(x==46.875)),
            'phase_variance_median':float(np.median([r['phase_variance'] for r in rs])),
            'hr_band_ratio_median':float(np.median([r['hr_band_ratio'] for r in rs]))}
for sample in ['p03_0.3m','p03_0.6m']:
    selected=[r for r in rows if r['sample']==sample]
    result[sample]={'record':next(r for r in records if r['sample']==sample),'methods':{}}
    for coefficient in ['first_chirp','zero_doppler','adjacent_doppler']:
      for method in ['complex_real','complex_imag','unwrap','edacm']:
        rs=[r for r in selected if r['coefficient']==coefficient and r['method']==method and r['label_offset_s']==0]
        matched=[r for r in rs if r['window_start']/30>=121 and (r['window_start']+384)/30<=480]
        # Four disjoint time blocks, no averaging across distances.
        blocks={str(t):stats([r for r in rs if t<=r['window_start']/30<t+150]) for t in [0,150,300,450]}
        controls={str(o):stats([r for r in selected if r['coefficient']==coefficient and r['method']==method
                              and r['label_offset_s']==o and r['window_start']/30>=121 and (r['window_start']+384)/30<=480]) for o in [0,-60,60,-120,120]}
        # Nonoverlapping windows to check overlap-driven correlation, fixed stride 384.
        nonoverlap=stats([r for r in rs if r['window_start']%384==0])
        result[sample]['methods'][coefficient+'/'+method]={'all':stats(rs),'matched':stats(matched),'blocks':blocks,'controls':controls,'nonoverlap':nonoverlap}
    # Native whole-record range power from previous ADC audit, no ECG selection.
    path=args.range_profile_dir/(sample+'_short_native_range_profiles.npz')
    power=np.load(path)['mean']; center=result[sample]['record']['selected_native_bin']
    result[sample]['native_roi']={'center':center,'power':float(power[center]),
        'power_fraction':float(power[center]/power.sum()),'neighbor_power':power[center-2:center+3].tolist()}
(p/'P03_CHECK.json').write_text(json.dumps(result,indent=2,allow_nan=False))
for sample,e in result.items():
 print(sample,'roi',e['native_roi'],'rx',e['record']['rx'])
 for method in ['first_chirp/unwrap','zero_doppler/unwrap','zero_doppler/edacm','adjacent_doppler/unwrap']:
  print(method,e['methods'][method])
