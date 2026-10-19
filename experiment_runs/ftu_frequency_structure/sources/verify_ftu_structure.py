"""Independent archived-spectrum checks: unwrap+SciPy instead of conjugate sums."""
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend

ROOT=Path('experiment_runs/ftu_frequency_structure')
SRC=Path('experiment_runs/ftu_adc_controlled')


def main():
    begin=time.perf_counter(); checks=[]; maxerr=0.; maskchecks=0; metricchecks=0
    metrics=pd.read_csv(ROOT/'record_metrics.csv')
    for path in sorted((ROOT/'spectra').glob('*.npz')):
        d=np.load(SRC/'signals'/path.name); a=np.load(path)
        ref=d['full_none_DC_roi'].transpose(1,2,0).reshape(128,1200)
        freq=np.fft.rfftfreq(4096,.05)*60; band=(freq>=45)&(freq<=150)
        np.testing.assert_array_equal(freq[band],a['frequency_bpm'])
        for arm in ['legacy_mean','legacy_none','full_none']:
            scale=.125 if arm.startswith('legacy') else 1.
            z=d[arm+'_DC_roi'].transpose(1,2,0).reshape(128,1200)
            for length in [512,768]:
                starts=a[f'starts_{length}']; common=a[f'common_{length}']
                assert (starts[common]>=256).all()
                k=np.flatnonzero(starts>=256)[0]; start=starts[k]
                for c,cell in enumerate(a['panel_cells']):
                    v=z[cell,start:start+length].astype(np.complex128)
                    # Absolute start differs from cumulative P2 only by a constant;
                    # detrend removes it, without treating phase origin as a discontinuity.
                    phase=detrend(np.unwrap(np.angle(v)),type='linear')
                    p=abs(np.fft.rfft(phase*np.hanning(length),4096)[band])**2;p/=p.sum()
                    stored=a[f'{arm}_{length}_phase_panel'][c,k]
                    if np.isfinite(stored).all():
                        err=float(abs(p-stored).max());maxerr=max(maxerr,err)
                        np.testing.assert_allclose(p,stored,atol=2e-7,rtol=1e-4)
                    for j,s in enumerate(starts):
                        v=z[cell,s:s+length].astype(np.complex128);b=ref[cell,s:s+length].astype(np.complex128)
                        expected=(np.isfinite(v).all() and (abs(v)>0).all()
                            and np.linalg.norm(v)>1e-6*scale*np.linalg.norm(b)
                            and np.count_nonzero(abs(v)<.01*scale*np.median(abs(ref[cell,:256])))/length<=.01)
                        assert expected==a[f'{arm}_{length}_valid_panel'][c,j]
                        maskchecks+=1
                # Verify every saved window label and finite-support intersection.
                ys=np.empty_like(a[f'labels_{length}'])
                for off in range(3):
                    for q,shift in enumerate([0,-20,-15,15,20]):
                        for j,s in enumerate(starts):
                            t=(np.arange(length)+s)*.05+off
                            if shift:t=np.remainder(t+shift,60)
                            ys[off,q,j]=np.mean(np.interp(t,d['reference_time'],d['reference_hr'],left=np.nan,right=np.nan))
                np.testing.assert_allclose(ys,a[f'labels_{length}'],atol=1e-10,equal_nan=True)
                np.testing.assert_array_equal(common,np.isfinite(ys).all(axis=(0,1))&(starts>=256))
                for off in range(3):
                    for q,shift in enumerate([0,-20,-15,15,20]):
                        row=metrics[(metrics.record==path.stem)&(metrics.arm==arm)&(metrics.length==length)
                                    &(metrics.kind=='panel')&(metrics.offset==off)&(metrics['shift']==shift)].iloc[0]
                        p=a[f'{arm}_{length}_phase_panel'][:,common]
                        valid=a[f'{arm}_{length}_valid_panel'][:,common]
                        y=ys[off,q,common]
                        if not len(y):
                            assert pd.isna(row.mae) and pd.isna(row.coverage)
                            continue
                        error=[];near=[];hit=[]
                        for c in range(8):
                            use=valid[c]&np.isfinite(p[c]).all(axis=-1)
                            if not use.any():continue
                            peak=a['frequency_bpm'][p[c,use].argmax(axis=-1)]
                            error.append(abs(peak-y[use]).mean());hit.append((abs(peak-y[use])<=5).mean())
                            near.append(np.mean([power[abs(a['frequency_bpm']-target)<=1200/length].sum()
                                                for power,target in zip(p[c,use],y[use])]))
                        for key,vals in [('mae',error),('near_power',near),('within5',hit)]:
                            expected=np.mean(vals) if vals else np.nan
                            np.testing.assert_allclose(row[key],expected,atol=2e-7,equal_nan=True)
                            metricchecks+=1
                checks.append(dict(record=path.stem,arm=arm,length=length,common_windows=int(common.sum())))
        # Continuous raw evidence, no z-score or phase-origin comparison.
        fig,axes=plt.subplots(3,8,figsize=(20,7),sharex=True)
        for c,cell in enumerate(a['panel_cells']):
            z=ref[cell].astype(np.complex128);amp=abs(z);t=np.arange(1200)/20
            axes[0,c].plot(t,amp,lw=.5);axes[0,c].axhline(.01*np.median(amp[:256]),color='r',lw=.6)
            axes[1,c].plot(t[1:],np.angle(z[1:]*z[:-1].conj()),lw=.4)
            axes[2,c].plot(t,np.unwrap(np.angle(z)),lw=.5)
            axes[0,c].set_title(f"B{a['panel_beams'][c]} R{a['roi'][c]}")
            axes[2,c].set_xlabel('seconds')
        axes[0,0].set_ylabel('raw amplitude');axes[1,0].set_ylabel('P2 increment / rad');axes[2,0].set_ylabel('unwrapped phase / rad')
        fig.suptitle(path.stem+' | full_none continuous raw traces; red=dropout threshold; phase is observed, not displacement')
        fig.tight_layout(rect=(0,0,1,.95));fig.savefig(ROOT/'figures'/f'{path.stem}_raw.png',dpi=120);plt.close(fig)
    m=pd.read_csv(ROOT/'record_metrics.csv'); gate=pd.read_csv(ROOT/'gate.csv')
    independent=[]
    for (rec,arm),g in m[(m.length==512)&(m.kind=='panel')&(m.arm!='legacy_mean')].groupby(['record','arm']):
        passes=[]
        for off in range(3):
            s=g[g.offset==off];v=s[s['shift']==0].iloc[0];w=s[s['shift']!=0]
            passes.append(bool(v.coverage>=.9 and v.within5>=.6 and all(v.near_power>=w.near_power+.01) and all(v.mae<w.mae)))
        expected=all(passes);stored=gate[(gate.record==rec)&(gate.arm==arm)]
        assert (stored.record_pass==expected).all()
        independent.append(dict(record=rec,arm=arm,record_pass=expected))
    result=dict(elapsed_seconds=time.perf_counter()-begin,spectrum_max_abs_error=maxerr,
                independent_spectrum_checks=checks,reliability_mask_checks=maskchecks,
                labels_and_common_windows_verified=True,primary_metric_checks=metricchecks,independent_gate=independent)
    (ROOT/'independent_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(spectrum_max_abs_error=maxerr,reliability_mask_checks=maskchecks,
                         passed_records={arm:sum(x['record_pass'] for x in independent if x['arm']==arm) for arm in ['legacy_none','full_none']}),indent=2))


if __name__=='__main__':main()
