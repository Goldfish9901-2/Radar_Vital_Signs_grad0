"""Create primary-lead-only review strips; no radar files or scores are read."""
from pathlib import Path
import hashlib
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('experiment_runs/synchronized_real_pilot_fixed')
RAW = Path('tmp/sync_real_results/sync_real_pilot')


def main():
    began = time.perf_counter()
    old = json.loads(Path('experiment_runs/synchronized_real_pilot/protocol.json').read_text())
    output = Path('tmp/ecg_only_review'); output.mkdir(exist_ok=True)
    manifest = []
    for member in old['members']:
        stem = Path(member['name']).stem; pid = stem.split('_')[0]; lead = old['ecg_primary_lead'][pid]
        file = RAW/f'{stem}_reference_ecg.npz'
        beatfile = RAW/f'{stem}_lead{lead}_beats.csv'
        peaks = pd.read_csv(beatfile).beat_time_s.to_numpy()
        with np.load(file) as data:
            ecg = data[f'ecg{lead}']; fs = float(data['fs']); origin = float(data['start_s'])
        for page in range(4):
            a, b = page*60, min((page+1)*60, len(ecg)/fs)
            n = int(np.ceil((b-a)/10)); fig, axes = plt.subplots(n, 1, figsize=(16, max(3, n*1.8)), squeeze=False)
            for row, ax in enumerate(axes[:, 0]):
                start, end = a+row*10, min(b, a+(row+1)*10)
                cut = np.arange(round(start*fs), round(end*fs))
                t = origin+cut/fs; ax.plot(t, ecg[cut], lw=.65, color='black')
                selected = peaks[(peaks >= t[0])&(peaks < origin+end)]
                for peak in selected: ax.axvline(peak, lw=.7, alpha=.55, color='tab:red')
                ax.set_xlim(origin+start, origin+end); ax.set_xlabel('Released-record time (s)'); ax.set_ylabel('mV'); ax.grid(alpha=.25)
                ax.set_title(f'{stem} / primary ECG{lead} / {origin+start:.3f}–{origin+end:.3f}s / {len(selected)} auto peaks', fontsize=10)
            fig.suptitle('ECG-only technical review. Red: unchanged automatic R peaks. No radar information.', fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, .96]); path = output/f'{stem}_page{page+1}.png'
            fig.savefig(path, dpi=140); plt.close(fig)
            manifest.append({'record': stem, 'primary_lead': lead, 'page': page+1,
                             'start_s': origin+a, 'end_s': origin+b, 'file': str(path).replace('\\','/'),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (ROOT/'ecg_only_packet_manifest.json').write_text(json.dumps({'elapsed_seconds': time.perf_counter()-began,
        'radar_inputs_read': False, 'pages': manifest}, indent=2))
    print('Created', len(manifest), 'ECG-only pages in', time.perf_counter()-began, 'seconds')


if __name__ == '__main__': main()
