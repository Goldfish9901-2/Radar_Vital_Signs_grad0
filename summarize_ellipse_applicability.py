"""Local verification, plots and descriptive report for frozen six-record audit."""
from pathlib import Path
import hashlib
import json
import shutil
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import linalg
from synchronized_ellipse_fixed import fit_calibration

OUT = Path('experiment_runs/ellipse_applicability')
SOURCE = Path('tmp/ellipse_applicability_results/ellipse_applicability')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mapped(x, c):
    return ((x-c['origin'])/c['scale']-c['center_normalized'])@c['transform']


def main():
    began = time.perf_counter()
    shutil.copytree(SOURCE, OUT, dirs_exist_ok=True)
    code = OUT/'source'; code.mkdir(exist_ok=True)
    for name in ['audit_ellipse_applicability.py','run_ellipse_applicability.py','synchronized_ellipse_fixed.py','run_synchronized_real_pilot.py']:
        shutil.copyfile(Path('tmp/ellipse_applicability_results/ellipse_audit_code')/name, code/name)
    for name in ['prepare_ellipse_applicability.py','kaggle/prepare_ellipse_applicability.py','summarize_ellipse_applicability.py']:
        target=code/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(name,target)
    shutil.copyfile('tmp/ellipse_applicability_results/ellipse_audit_code/audit_protocol.json',OUT/'executed_protocol.json')
    protocol=read(OUT/'audit_protocol.json')
    assert protocol==read(OUT/'executed_protocol.json')
    for name,h in protocol['source_hashes'].items():
        assert digest(code/name)==h
        assert Path(name).read_text()==(code/name).read_text()
    oldhash={}
    for folder in ['synchronized_real_pilot','synchronized_real_pilot_fixed']:
        root=OUT.parent/folder; manifest=read(root/'artifact_manifest.json')
        for name,h in manifest.items(): assert digest(root/name)==h,(folder,name)
        oldhash[folder]=len(manifest)
    oldacq={r['name']:r['sha256'] for r in read(OUT.parent/'synchronized_real_pilot/acquisition.json')}
    assert all(r['sha256']==oldacq[r['name']] for r in read(OUT/'acquisition.json'))
    records=read(OUT/'candidate_diagnostics.json'); blocks=read(OUT/'block_metrics.json')
    transfers=read(OUT/'transfer_metrics.json')
    plots=OUT/'plots'; plots.mkdir(exist_ok=True)
    errors=[]; candidates=[]; overview=[]; totalpairs=finite=proper=0
    cache={}
    for record in records:
        tag=record['record']; cloud=np.load(OUT/'clouds'/f'{tag}.npz'); x=cloud['iq']; t=cloud['time_s']
        assert float(cloud['native_fs'])==2000 and int(cloud['subsample_step'])==20
        models={}
        for fit in record['fits']:
            z=x[(t>=fit['first_s'])&(t<fit['last_s'])]
            assert len(fit['candidates'])==6
            totalpairs+=6
            try: _,_,center,transform,meta=fit_calibration(z); valid=True
            except ValueError: valid=False
            assert valid==fit['production_valid']
            for c in fit['candidates']:
                finite+=c['finite_generalized_eigenvalue']
                if not c['finite_generalized_eigenvalue']: continue
                assert 'eigenvector_real' in c
                v=np.array(c['conic_coefficients_normalized']); q=np.array([[v[0],v[1]/2],[v[1]/2,v[2]]])
                ctr=np.linalg.solve(q,-v[3:5]/2)
                errors.append(float(np.max(abs(ctr-c['center_normalized']))))
                candidates.append(dict(record=tag,fit=fit['fit'],candidate=c['candidate'],geometry_defined=c['geometry_defined'],radial_rmse=c['radial_rmse'],center_native=c['center_native'],semi_axes_native=c['axes_native'],quadratic_condition=c['quadratic_condition'],guard_failures='; '.join(c['guard_failures'])))
                if not c['geometry_defined']:
                    assert c['radial_rmse'] is None and c['axes_native'] is None and c['mapped_metrics'] is None
                    continue
                proper+=1; models[fit['fit']]=c
                shape=(q/(ctr@q@ctr-v[5]))/np.outer(fit['scale'],fit['scale'])
                errors.append(float(np.max(abs(shape-c['native_shape_matrix']))))
                errors.append(float(np.max(abs(1/np.sqrt(np.linalg.eigvalsh(shape))-c['axes_native']))))
                y=mapped(z,c); radius=np.hypot(y[:,0],y[:,1]); rmse=np.sqrt(np.mean((radius-1)**2))
                errors.append(abs(rmse-c['radial_rmse']))
                if valid:
                    assert np.allclose(center,c['center_normalized']) and np.allclose(transform,c['transform'])
        primary=models['cal_0_30']; mainfit=record['fits'][0]
        assert mainfit['production_valid']==next(r for r in protocol['records'] if r['record']==tag)['previous_primary_valid']
        cache[tag]=(x,t,models)
        overview.append(dict(record=tag,accepted=mainfit['production_valid'],radial_rmse=primary['radial_rmse'],arc95_deg=primary['mapped_metrics']['angular_95percent_arc_deg'],semi_axes_native=primary['axes_native'],shape_condition=primary['native_shape_condition']))
        # Common native axes across all six initial blocks; rejected ellipses diagnostic only.
        theta=np.linspace(0,2*np.pi,500); circle=np.c_[np.cos(theta),np.sin(theta)]
        def ellipse(c): return (circle@np.linalg.inv(c['transform'])+c['center_normalized'])*c['scale']+c['origin']
        e=ellipse(primary); initial=x[t<30]; allp=np.r_[initial,e]
        lo=allp.min(0); hi=allp.max(0); pad=(hi-lo)*.06
        fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
        for k,ax in enumerate(axes.flat):
            sel=(t>=k*5)&(t<(k+1)*5); ax.scatter(*x[sel].T,c=t[sel],s=3,cmap='viridis',vmin=0,vmax=30)
            ax.plot(*e.T,color='crimson',lw=1)
            r=np.linalg.norm(mapped(x[sel],primary),axis=1)
            ax.set_title(f'{k*5}-{(k+1)*5}s: radius RMSE={np.sqrt(np.mean((r-1)**2)):.3f}')
            ax.set(xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]),xlabel='I (native units)',ylabel='Q (native units)'); ax.set_aspect('equal',adjustable='box')
        fig.suptitle(f'{tag}: fixed 0-30s ellipse; '+('accepted' if mainfit['production_valid'] else 'REJECTED - diagnostic only'))
        fig.tight_layout(); fig.savefig(plots/f'{tag}_blocks.png',dpi=150); plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(16,4.8))
        axes[0].scatter(*x[t<60].T,c=t[t<60],s=2,cmap='viridis')
        for name,c in models.items():
            axes[0].plot(*ellipse(c).T,label=name,lw=1); axes[0].scatter(*c['center_native'],marker='x')
        axes[0].set(xlabel='I (native units)',ylabel='Q (native units)',title='0-60s; all fits diagnostic'); axes[0].legend(fontsize=7)
        r=np.linalg.norm(mapped(x,primary),axis=1)
        axes[1].plot(t,r,lw=.4); axes[1].axhline(1,color='red',lw=.8); axes[1].axvline(30,color='black',ls=':')
        axes[1].set(xlabel='Time (s)',ylabel='Mapped radius',title='Fixed initial ellipse, no HR')
        names=[f'cal_block{k}' for k in range(1,7)]+['post','later','score_whole']
        heat=np.array([[next(b['radial_rmse'] for b in blocks if b['record']==tag and b['fit']==name and b['evaluation']==target) for target in names] for name in models])
        im=axes[2].imshow(heat,aspect='auto'); axes[2].set_yticks(range(len(models)),models,fontsize=7); axes[2].set_xticks(range(len(names)),names,rotation=90,fontsize=7)
        axes[2].set_title('Radial RMSE (all diagnostic models)'); fig.colorbar(im,ax=axes[2])
        fig.suptitle(tag); fig.tight_layout(); fig.savefig(plots/f'{tag}_transfer.png',dpi=150); plt.close(fig)
    # Independent scalar radius/covariance/coverage calculations for every saved block.
    for b in blocks:
        x,t,models=cache[b['record']]; c=models[b['fit']]; z=mapped(x[(t>=b['first_s'])&(t<b['last_s'])],c)
        radius=np.hypot(z[:,0],z[:,1]); errors.append(abs(float(np.sqrt(np.mean((radius-1)**2)))-b['radial_rmse']))
        values=np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(z,rowvar=False,bias=True)),0))[::-1]
        errors.append(float(np.max(abs(values-b['mapped_centered_singular_values']))))
        angles=np.sort(np.mod(np.arctan2(z[:,1],z[:,0]),2*np.pi)); n=len(angles); count=int(np.ceil(.95*n)); ext=np.r_[angles,angles+2*np.pi]
        arc=min(ext[i+count-1]-ext[i] for i in range(n))*180/np.pi
        errors.append(abs(arc-b['angular_95percent_arc_deg']))
    for row in transfers:
        x,t,models=cache[row['record']]; a,b=row['pair'].split('->'); ca,cb=models[a],models[b]
        fits=next(r['fits'] for r in records if r['record']==row['record'])
        fa=next(f for f in fits if f['fit']==a); fb=next(f for f in fits if f['fit']==b)
        xa=x[(t>=fa['first_s'])&(t<fa['last_s'])]; xb=x[(t>=fb['first_s'])&(t<fb['last_s'])]
        def rmse(x,c):
            z=mapped(x,c); return np.sqrt(np.mean((np.hypot(z[:,0],z[:,1])-1)**2))
        for key,val in [('early_to_late_rmse',rmse(xb,ca)),('late_to_early_rmse',rmse(xa,cb))]: errors.append(abs(val-row[key]))
        delta=np.array(cb['center_native'])-ca['center_native']; change=np.sqrt(delta@np.array(ca['native_shape_matrix'])@delta)
        errors.append(abs(change-row['center_shift_in_early_radius_units']))
        shapechange=np.linalg.norm(np.log(linalg.eigvalsh(cb['native_shape_matrix'],ca['native_shape_matrix'])))
        errors.append(abs(shapechange-row['log_shape_change']))
        ya,yb=mapped(xa,ca),mapped(xb,ca)
        ia=np.floor(np.mod(np.arctan2(ya[:,1],ya[:,0]),2*np.pi)*36/(2*np.pi)).astype(int)
        ib=np.floor(np.mod(np.arctan2(yb[:,1],yb[:,0]),2*np.pi)*36/(2*np.pi)).astype(int)
        ra,rb=np.hypot(ya[:,0],ya[:,1]),np.hypot(yb[:,0],yb[:,1])
        diffs=[abs(np.median(ra[ia==i])-np.median(rb[ib==i])) for i in range(36) if sum(ia==i)>=10 and sum(ib==i)>=10]
        assert len(diffs)==row['matched_angular_bins']
        errors.append(abs(float(np.median(diffs))-row['matched_angle_radial_change']))
        penalty=max(rmse(xb,ca)-rmse(xb,cb),rmse(xa,cb)-rmse(xa,ca))
        errors.append(abs(penalty-row['max_transfer_penalty']))
        supported=len(diffs)>=6 and row['minimum_arc95_deg']>=180
        drift=supported and fa['production_valid'] and fb['production_valid'] and np.median(diffs)>=.1 and penalty>=.1 and (change>=.25 or shapechange>=.25)
        stable=supported and np.median(diffs)<=.05 and penalty<=.05 and change<=.1 and shapechange<=.1
        assert row['classification']==('time_dependent_mismatch_supported' if drift else 'stable_geometry_supported' if stable else 'inconclusive')
    assert len(records)==6 and totalpairs==144 and finite==72 and proper==24
    assert max(errors)<1e-8,max(errors)
    execution=read(OUT/'execution.json'); assert not execution['hr_predictions_generated'] and not execution['ecg_or_hr_arrays_decoded'] and not execution['other27_subjects_read']
    pd.DataFrame(candidates).to_csv(OUT/'finite_candidates.csv',index=False)
    pd.DataFrame(overview).to_csv(OUT/'primary_summary.csv',index=False)
    validation=dict(status='passed',record_count=6,fits=24,all_eigenpairs=144,finite_candidates=72,proper_ellipses=24,undefined_nonellipse_candidates=48,block_metrics=len(blocks),transfer_pairs=len(transfers),max_absolute_numeric_discrepancy=max(errors),original_artifacts_unchanged=oldhash,acquisition_sha256_match=True,executed_protocol_equal=True,production_source_unchanged=True,elapsed_seconds=time.perf_counter()-began,scope='Independent algebra/radius/coverage/shape checks and immutable production replay; not physiological validation')
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2))
    manifest={str(p.relative_to(OUT)).replace('\\','/'):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
    assert all(digest(OUT/name)==h for name,h in manifest.items())
    print(json.dumps(validation,indent=2)); print(pd.DataFrame(overview).to_string(index=False))


if __name__=='__main__': main()
