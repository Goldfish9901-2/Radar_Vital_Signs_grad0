"""Independent synthesis and SciPy-periodogram verification of evaluation calibration."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import periodogram

ROOT=Path('experiment_runs/ftu_evaluation_controls')
SRC=Path('experiment_runs/ftu_reference_support_v2')


def main():
    begin=time.perf_counter();params=pd.read_csv(ROOT/'generation_parameters.csv');metrics=pd.read_csv(ROOT/'metrics.csv')
    support=pd.read_csv(SRC/'selected_support.csv');refs=pd.read_csv(SRC/'polar_reference_values.csv')
    maxsignal=0.;maxmetric=0.;synth_checks=0;metricchecks=0;actual={}
    for tag,pr in params.groupby('record'):
        d=np.load(ROOT/'signals'/f'{tag}.npz');ref=refs[refs.record==tag]
        t=ref.time_second.to_numpy();y=ref.hr_bpm.to_numpy();grid=np.arange(1200)*.05
        for _,row in pr.iterrows():
            phase=row.phase_rad;seed=int(row.seed);sigma=row.noise_sigma
            rng=np.random.default_rng(np.random.SeedSequence([seed,int(row.record_index)]))
            np.testing.assert_allclose(rng.uniform(0,2*np.pi),phase,atol=1e-14)
            noise=rng.normal(size=1200);np.testing.assert_array_equal(noise,d[f'noise_seed{seed}'])
            if row.generator=='positive':
                freq=interp1d(t,y,bounds_error=False,fill_value=np.nan)(grid+row.generation_offset)
            elif row.generator=='constant':
                freq=np.full(1200,y.mean());freq[grid>t[-1]]=np.nan
            else:
                centered=t-t.mean();slope=np.dot(centered,y-y.mean())/np.dot(centered,centered)
                freq=y.mean()+slope*(grid-t.mean());freq[grid>t[-1]]=np.nan
            valid=np.isfinite(freq);v=freq[valid]
            angle=np.concatenate([[phase],phase+np.cumsum((v[:-1]+v[1:])*.5/60*.05)*2*np.pi])
            expected=np.full(1200,np.nan);expected[valid]=np.cos(angle)+sigma*noise[valid]
            error=float(np.nanmax(abs(expected-d[row.signal_id])));maxsignal=max(maxsignal,error)
            np.testing.assert_allclose(expected,d[row.signal_id],atol=2e-10,equal_nan=True);synth_checks+=1
            mask=(metrics.record==tag)&(metrics['mode']==row['mode'])&(metrics.generator==row.generator)&(metrics.generation_offset==row.generation_offset)&(metrics.seed==seed)&(metrics.noise_sigma==sigma)
            selected=metrics[mask]
            for length,group in selected.groupby('length'):
                length=int(length);sup=support[(support.record==tag)&(support.length==length)].iloc[0]
                starts=np.array(json.loads(sup.all_offsets_common_starts),dtype=int)
                if not len(starts):
                    assert group.mae.isna().all();continue
                traces=np.stack([d[row.signal_id][s:s+length] for s in starts])
                hz,p=periodogram(traces,fs=20,window=np.hanning(length),nfft=4096,detrend='linear',axis=-1,scaling='spectrum')
                band=(hz*60>=45)&(hz*60<=150);bpm=hz[band]*60;p=p[:,band];p/=p.sum(axis=1,keepdims=True)
                peak=bpm[p.argmax(axis=1)]
                for index,r in group.iterrows():
                    labels=np.array([interp1d(t,y,bounds_error=False,fill_value=np.nan)(np.arange(s,s+length)/20+r.offset+r['shift']).mean() for s in starts])
                    power=np.array([v[abs(bpm-target)<=1200/length].sum() for v,target in zip(p,labels)])
                    values=dict(coverage=1.,mae=float(abs(peak-labels).mean()),within5=float((abs(peak-labels)<=5).mean()),near_power=float(power.mean()))
                    for key,value in values.items():
                        err=abs(value-r[key]);maxmetric=max(maxmetric,err)
                        np.testing.assert_allclose(value,r[key],atol=2e-10,rtol=1e-9);metricchecks+=1
                    actual[index]=values
    gates=pd.read_csv(ROOT/'score_rule_diagnostics.csv');gkeys=['record','mode','generator','generation_offset','seed','noise_sigma','length','offset']
    verifiedgates=0
    for _,g in gates.iterrows():
        subset=metrics
        for key in gkeys:subset=subset[subset[key]==g[key]]
        if int(g.common_windows)<5:
            assert not g.score_rule_pass;continue
        true=subset[subset['shift']==0].iloc[0];controls=subset[subset['shift']!=0]
        a=actual[true.name];b=[actual[i] for i in controls.index]
        passed=a['coverage']>=.9 and a['within5']>=.6 and all(a['near_power']-v['near_power']>=.01 and a['mae']<v['mae'] for v in b)
        assert passed==g.score_rule_pass;verifiedgates+=1
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert hashlib.sha256((SRC/'decision.json').read_bytes()).hexdigest()==protocol['source_files'][str(SRC/'decision.json').replace('\\','/')]
    assert json.loads((SRC/'decision.json').read_text())['evaluable_records']==0
    result=dict(elapsed_seconds=time.perf_counter()-begin,independent_signal_checks=synth_checks,
        maximum_signal_difference=maxsignal,independent_metric_checks=metricchecks,maximum_metric_difference=maxmetric,
        independent_score_gate_checks=verifiedgates,previous_0_of_6_preserved=True,new_radar_data_read=False)
    (ROOT/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
