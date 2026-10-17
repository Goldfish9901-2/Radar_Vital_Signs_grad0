"""Frozen 2x2 synthetic spectrum/readout experiment; no fitting on import."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL = dict(
    arms={'A':dict(scale=True,band=False),'B':dict(scale=False,band=False),
          'C':dict(scale=True,band=True),'D':dict(scale=False,band=True)},
    seeds=[42,43,44],epochs=60,patience=12,batch_size=64,lr=.0003,weight_decay=.0001,
    model='129->64->32->1 GELU dropout .15; unchanged',loss='SmoothL1 beta .5',
    scheduler='ReduceLROnPlateau factor .5 patience 4; validation centered RMSE',clip=1.,
    preprocessing='original per-window zscore preserved; optional train-only per-bin StandardScaler; then mask',
    band_hz=[.5,2.75],mask='inclusive bin-center frequencies; all other bins zero AFTER optional scaling',
    training_data='exact previous 768 signals; RNG 20261017; validation unchanged 192 signals',
    checkpoint_selection='minimum validation centered RMSE separately per arm/seed',
    configuration_selection='lowest mean validation centered RMSE across three seeds; alphabetic tie break',
    evaluation='previous evaluation is development evidence only; fresh frozen final set generated AFTER configuration selection',
    final_seed=20261018,final_groups_per_grid=24,signals_per_group=8,
    final_grids={'on_grid':'0.75+.05*k k=0..35','between_grid':'0.7625/.7875+.05*k k=0..34'},
    generation='uniform grid draw, phase uniform [0,2pi], amplitude uniform [.5,2], relative Gaussian noise std uniform [0,.3]',
    metrics='raw and group-centered RMSE/r/population std ratio; MAE; center over full artificial group BEFORE stratification',
    strata=dict(frequency_edges_hz=[.75,1.25,1.75,2.25,2.50001],noise_edges=[0,.1,.2,.300001]),
    contrasts='paired A-B,A-C,C-D,B-D; positive RMSE difference means second arm improves; report each seed, no seed-as-subject inference',
    success_reference='report old reference RMSE<=4.6875 and r>=.9 descriptively; no real-data training in this experiment',
    scope='synthetic readout only; no causal claim that low-variance noise bins explain all errors',
)


def generate(final=False):
    from src.features.edacm import zscore_1d
    from src.features.representations import frequency_features
    rng=np.random.default_rng(PROTOCOL['final_seed'] if final else 20261017)
    train=.75+.05*np.arange(36)
    between=np.concatenate([.7625+.05*np.arange(35),.7875+.05*np.arange(35)])
    settings=([('final_on_grid',24,train),('final_between_grid',24,between)] if final else
              [('train',96,train),('val',24,.775+.05*np.arange(35)),('development',24,between)])
    rows=[];features=[]
    for split,count,grid in settings:
        for group in range(count):
            for window in range(8):
                hz=rng.choice(grid)
                amp,phase,noise=rng.uniform(.5,2),rng.uniform(0,2*np.pi),rng.uniform(0,.3)
                signal=amp*(np.sin(2*np.pi*hz*np.arange(256)/20+phase)+noise*rng.normal(size=256))
                features.append(frequency_features(zscore_1d(signal)[None])[0][0])
                rows.append(dict(sample_tag=f'{split}_{group}',split=split,window_start=window,
                                 label_bpm=60*hz,frequency_hz=hz,amplitude=amp,phase_rad=phase,
                                 relative_noise_std=noise,absolute_noise_std=amp*noise))
    return pd.DataFrame(rows),np.asarray(features)


def metrics(y,p,cy,cp):
    def pair(a,b,prefix):
        sa,sb=np.std(a),np.std(b)
        return {prefix+'rmse':float(np.sqrt(np.mean((a-b)**2))),
                prefix+'r':float(np.corrcoef(a,b)[0,1]) if sa>1e-10 and sb>1e-10 else None,
                prefix+'std_ratio':float(sb/sa) if sa>1e-10 else None}
    return dict(n=len(y),mae=float(np.mean(abs(y-p))),**pair(y,p,''),**pair(cy,cp,'centered_'))


def score(frame,pred):
    y=frame.label_bpm.to_numpy()
    f=pd.DataFrame(dict(y=y,p=pred,group=frame.sample_tag.to_numpy()))
    c=f[['y','p']]-f.groupby('group')[['y','p']].transform('mean')
    return metrics(y,np.asarray(pred),c.y.to_numpy(),c.p.to_numpy())


def transform(x,mean,scale,arm):
    spec=PROTOCOL['arms'][arm]
    result=(x.astype(float)-mean)/scale if spec['scale'] else x.astype(float).copy()
    if spec['band']:
        hz=np.fft.rfftfreq(256,1/20)
        result[:,(hz<.5)|(hz>2.75)]=0
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    import random,shutil,time
    assert Path('/kaggle/working').is_dir(),'Kaggle-only model fitting'
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler
    start=time.perf_counter()
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True);assert torch.cuda.is_available()
    root=Path(__file__).resolve().parents[2]
    out=Path('/kaggle/working/ftu_spectrum_factorial');out.mkdir()
    shutil.copyfile(root/'source_manifest.json',out/'source_manifest.json')
    (out/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    f,replayed=generate()
    x=np.load(root/'reference_features.npz')['x_freq']
    np.testing.assert_allclose(replayed,x,atol=2e-6,rtol=1e-5)
    f.to_csv(out/'development_inputs.csv',index=False)
    np.savez_compressed(out/'development_features.npz',x_freq=x)
    masks={s:(f.split==s).to_numpy() for s in ['train','val','development']}
    y=f.label_bpm.to_numpy();mean=float(y[masks['train']].mean());std=float(y[masks['train']].std())
    scaler=StandardScaler().fit(x[masks['train']].astype(float))
    np.savez(out/'scaler.npz',mean=scaler.mean_,scale=scaler.scale_)
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda')
    target=tensor((y-mean)/std);ids=torch.as_tensor(np.flatnonzero(masks['train']),device='cuda')
    def model():return nn.Sequential(nn.Linear(129,64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,32),nn.GELU(),nn.Dropout(.15),nn.Linear(32,1)).cuda()
    history=[];summary=[];predictions=[];init_hashes={}
    for arm in PROTOCOL['arms']:
        tx=tensor(transform(x,scaler.mean_,scaler.scale_,arm))
        for seed in PROTOCOL['seeds']:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            net=model();name=f'{arm}_seed{seed}'
            init_hashes[name]=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in net.parameters())).hexdigest()
            optimizer=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001)
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=4)
            lossfn=nn.SmoothL1Loss(beta=.5);generator=torch.Generator(device='cuda').manual_seed(seed)
            best=float('inf');stale=0
            for epoch in range(1,61):
                net.train();order=ids[torch.randperm(len(ids),generator=generator,device='cuda')]
                for batch in order.split(64):
                    optimizer.zero_grad();loss=lossfn(net(tx[batch]).squeeze(1),target[batch]);loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
                net.eval()
                with torch.no_grad():pred=(net(tx).squeeze(1)*std+mean).cpu().numpy()
                scores={s:score(f[masks[s]],pred[masks[s]]) for s in ['train','val']}
                for s,m in scores.items():history.append(dict(arm=arm,seed=seed,epoch=epoch,split=s,**m))
                value=scores['val']['centered_rmse'];scheduler.step(value)
                if value<best:
                    best=value;stale=0
                    torch.save(dict(state_dict=net.state_dict(),epoch=epoch,arm=arm,seed=seed,
                                    scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,label_mean=mean,label_std=std),out/f'{name}.pt')
                else:stale+=1
                if stale>=12:break
            saved=torch.load(out/f'{name}.pt',map_location='cuda',weights_only=False)
            net.load_state_dict(saved['state_dict']);net.eval()
            with torch.no_grad():pred=(net(tx).squeeze(1)*std+mean).cpu().numpy()
            for s,mask in masks.items():summary.append(dict(arm=arm,seed=seed,split=s,best_epoch=saved['epoch'],epochs=epoch,**score(f[mask],pred[mask])))
            fp=f.copy();fp['pred_bpm']=pred;fp['arm']=arm;fp['seed']=seed;predictions.append(fp)
            print(name,'best_epoch',saved['epoch'],'validation',best,flush=True)
    # Select using validation only BEFORE generating or evaluating the fresh final set.
    scores=pd.DataFrame(summary).query("split == 'val'").groupby('arm').centered_rmse.mean().sort_index()
    selected=scores.idxmin()
    selection=dict(selected_arm=selected,validation_mean_centered_rmse=scores.to_dict())
    (out/'selection.json').write_text(json.dumps(selection,indent=2))
    final,fx=generate(final=True);final.to_csv(out/'final_inputs.csv',index=False)
    np.savez_compressed(out/'final_features.npz',x_freq=fx)
    for arm in PROTOCOL['arms']:
        tx=tensor(transform(fx,scaler.mean_,scaler.scale_,arm))
        for seed in PROTOCOL['seeds']:
            net=model();saved=torch.load(out/f'{arm}_seed{seed}.pt',map_location='cuda',weights_only=False)
            net.load_state_dict(saved['state_dict']);net.eval()
            with torch.no_grad():pred=(net(tx).squeeze(1)*std+mean).cpu().numpy()
            for s in final.split.unique():
                mask=(final.split==s).to_numpy()
                summary.append(dict(arm=arm,seed=seed,split=s,best_epoch=saved['epoch'],**score(final[mask],pred[mask])))
            fp=final.copy();fp['pred_bpm']=pred;fp['arm']=arm;fp['seed']=seed;predictions.append(fp)
    for seed in PROTOCOL['seeds']:assert len({init_hashes[f'{a}_seed{seed}'] for a in PROTOCOL['arms']})==1
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    pd.concat(predictions).to_csv(out/'predictions.csv',index=False)
    (out/'execution.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-start,
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),initialization_hashes=init_hashes),indent=2))
    (out/'artifact_manifest.json').write_text(json.dumps({p.name:digest(p) for p in out.iterdir() if p.is_file()},indent=2))
    shutil.make_archive('/kaggle/working/000_ftu_spectrum_factorial','zip',out)
