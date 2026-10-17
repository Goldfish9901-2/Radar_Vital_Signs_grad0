"""Frozen same-visit cross-capture MLP diagnostic; fitting is Kaggle-only."""
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def score(frame, pred):
    y = frame.label_bpm.to_numpy()
    tmp = pd.DataFrame(dict(y=y, p=pred, record=frame.sample_tag.to_numpy()))
    centered = tmp[['y', 'p']] - tmp.groupby('record')[['y', 'p']].transform('mean')
    a, b = centered.y.to_numpy(), centered.p.to_numpy()
    return dict(mae=float(np.mean(abs(y-pred))),
                centered_rmse=float(np.sqrt(np.mean((a-b)**2))),
                centered_r=float(np.corrcoef(a,b)[0,1]) if a.std()>1e-10 and b.std()>1e-10 else None,
                centered_r2=float(1-np.sum((a-b)**2)/np.sum(a*a)) if np.sum(a*a)>1e-10 else None,
                centered_std_ratio=float(b.std()/a.std()) if a.std()>1e-10 else None)


def synthetic():
    from src.features.edacm import zscore_1d
    from src.features.representations import frequency_features
    rng = np.random.default_rng(20261017)
    rows, features = [], []
    grids = [0.75+.05*np.arange(36), .775+.05*np.arange(35),
             np.concatenate([.7625+.05*np.arange(35), .7875+.05*np.arange(35)])]
    for split, count, grid in zip(['train','val','evaluation'], [96,24,24], grids):
        for record in range(count):
            for window in range(8):
                hz = rng.choice(grid)
                amp, phase, noise = rng.uniform(.5,2), rng.uniform(0,2*np.pi), rng.uniform(0,.3)
                signal = amp*(np.sin(2*np.pi*hz*np.arange(256)/20+phase)+noise*rng.normal(size=256))
                features.append(frequency_features(zscore_1d(signal)[None])[0][0])
                rows.append(dict(sample_tag=f'{split}_{record}', group_key='synthetic', split=split,
                                 window_start=window, label_bpm=60*hz))
    return pd.DataFrame(rows), np.asarray(features)


def run(data_hash):
    assert Path('/kaggle/working').is_dir(), 'Training must run on Kaggle'
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler
    from src.training.ftu_position_protocol import load_arrays
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True, warn_only=True)
    assert torch.cuda.is_available()
    root = Path(__file__).resolve().parents[2]
    protocol = json.loads((root/'protocol.json').read_text())
    out = Path('/kaggle/working/ftu_record_learnability'); out.mkdir()
    for name in ['protocol.json','record_split.csv','source_manifest.json']:
        shutil.copyfile(root/name, out/name)
    archive, = Path('/kaggle/input').rglob('ftu_timebase_edacm_v2.zip')
    assert sha(archive)==data_hash
    summary=[]; histories=[]; predictions=[]; permutations=[]

    def fit(frame, features, domain, seed, shuffle_seed=None):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        masks={s:(frame.split==s).to_numpy() for s in ['train','val','evaluation']}
        scaler=StandardScaler().fit(features[masks['train']].astype(float))
        y=frame.label_bpm.to_numpy(); target=y.copy()
        if shuffle_seed is not None:
            rng=np.random.default_rng(shuffle_seed)
            for tag, indices in frame[masks['train']].groupby('sample_tag').groups.items():
                ids=np.asarray(list(indices)); donors=rng.permutation(ids)
                target[ids]=y[donors]
                assert np.array_equal(np.sort(target[ids]),np.sort(y[ids]))
                for i,d in zip(ids,donors):
                    permutations.append(dict(seed=seed,shuffle_seed=shuffle_seed,sample_tag=tag,index=int(i),donor=int(d)))
        mean, std=float(y[masks['train']].mean()),float(y[masks['train']].std())
        tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
        x=tensor(scaler.transform(features)); target=tensor((target-mean)/std)
        ids=torch.as_tensor(np.flatnonzero(masks['train']),device='cuda')
        model=nn.Sequential(nn.Linear(129,64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,32),
                            nn.GELU(),nn.Dropout(.15),nn.Linear(32,1)).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=4)
        lossfn=nn.SmoothL1Loss(beta=.5)
        name=f'{domain}_seed{seed}_'+('true' if shuffle_seed is None else f'shuffle{shuffle_seed}')
        best=float('inf'); stale=0
        # Independent order generator keeps paired minibatch order identical even if dropout consumes RNG.
        generator=torch.Generator(device='cuda').manual_seed(seed)
        for epoch in range(1,protocol['epochs']+1):
            model.train()
            order=ids[torch.randperm(len(ids),generator=generator,device='cuda')]
            for batch in order.split(protocol['batch_size']):
                optimizer.zero_grad(); loss=lossfn(model(x[batch]).squeeze(1),target[batch]); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            model.eval()
            with torch.no_grad(): pred=(model(x).squeeze(1)*std+mean).cpu().numpy()
            scores={s:score(frame[masks[s]],pred[masks[s]]) for s in ['train','val']}
            for s,m in scores.items(): histories.append(dict(run=name,epoch=epoch,split=s,**m))
            value=scores['val']['centered_rmse']; scheduler.step(value)
            if value<best:
                best=value; stale=0
                torch.save(dict(state_dict=model.state_dict(),epoch=epoch,scaler_mean=scaler.mean_,
                                scaler_scale=scaler.scale_,label_mean=mean,label_std=std,run=name),out/f'{name}.pt')
            else: stale+=1
            if stale>=protocol['patience']: break
        saved=torch.load(out/f'{name}.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(saved['state_dict']); model.eval()
        with torch.no_grad(): pred=(model(x).squeeze(1)*std+mean).cpu().numpy()
        final={}
        for s,mask in masks.items():
            final[s]=score(frame[mask],pred[mask])
            summary.append(dict(run=name,domain=domain,seed=seed,shuffle_seed=shuffle_seed,split=s,
                                best_epoch=saved['epoch'],epochs=epoch,**final[s],
                                mean_baseline_mae=float(abs(y[mask]-mean).mean()),
                                median_baseline_mae=float(abs(y[mask]-np.median(y[masks['train']])).mean())))
        f=frame.copy(); f['pred_bpm']=pred; f['run']=name; predictions.append(f)
        print(name,final,flush=True)
        return final

    sf,sx=synthetic(); assert sx.shape==(1152,129)
    sf.to_csv(out/'synthetic_inputs.csv',index=False)
    np.savez_compressed(out/'synthetic_features.npz',x_freq=sx)
    gate=True
    for seed in protocol['seeds']:
        result=fit(sf,sx,'synthetic',seed)
        gate=gate and all(result[s]['centered_rmse']<=4.6875 and
                         result[s]['centered_r'] is not None and result[s]['centered_r']>=.9
                         for s in ['val','evaluation'])
    if gate:
        frame,_,features,_=load_arrays(archive)
        split=pd.read_csv(root/'record_split.csv')
        selected=frame.sample_tag.isin(split.sample_tag)
        xf=features[selected]; frame=frame[selected].drop(columns='split').merge(
            split[['sample_tag','split']],on='sample_tag',how='left',validate='many_to_one')
        frame=frame.rename(columns={'label_heart_rate':'label_bpm'})
        frame=frame[['sample_tag','group_key','window_start','split','label_bpm']].reset_index(drop=True)
        assert len(frame)==1728 and frame.groupby('sample_tag').size().eq(8).all()
        np.savez_compressed(out/'real_features.npz',x_freq=xf)
        frame.to_csv(out/'real_inputs.csv',index=False)
        for seed in protocol['seeds']:
            fit(frame,xf,'real',seed)
            for shuffle_seed in protocol['shuffle_seeds']: fit(frame,xf,'real',seed,shuffle_seed)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(histories).to_csv(out/'history.csv',index=False)
    pd.concat(predictions).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(permutations).to_csv(out/'permutations.csv',index=False)
    (out/'execution.json').write_text(json.dumps(dict(synthetic_gate_pass=gate,elapsed_sec=time.perf_counter()-start,
        data_sha256=data_hash,torch_version=torch.__version__,gpu=torch.cuda.get_device_name()),indent=2))
    (out/'artifact_manifest.json').write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file()},indent=2))
    shutil.make_archive('/kaggle/working/000_ftu_record_learnability','zip',out)
