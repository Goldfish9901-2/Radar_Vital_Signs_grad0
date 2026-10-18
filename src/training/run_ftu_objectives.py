"""Paired absolute versus within-record dynamic objectives; fixed60 epochs."""
import hashlib,json,random,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
from src.training.ftu_position_protocol import load_arrays
from src.training.ftu_spectrum_factorial import score as base_score

PROTOCOL=dict(seeds=[42,43,44],shuffle_seeds=[101,202,303],epochs=60,
    objectives=['absolute','dynamic'],batch_records=8,windows_per_record=8,
    model='fresh MLP129-64-32-1 GELU dropout .15; no synthetic pretraining',
    preprocessing='unchanged D: original window normalization, no per-bin scaler, same 29-bin mask, no renormalization',
    optimizer='AdamW lr .0003 weight_decay .0001; CONSTANT learning rate for all60 epochs, no scheduler or early stopping',
    loss='both SmoothL1 beta .5, clip1, identical global training label mean/std; dynamic centers prediction and target separately inside each complete training record',
    batching='8 complete records x8 windows; final batch4 records; same initialization/dropout RNG and per-epoch record order across targets and shuffles',
    selection='minimum correct-label validation record-centered RMSE; best checkpoint only for evaluation',
    split='unchanged216 captures/1728 windows,108/54/54 captures, original six training subjects',
    shuffle='same fixed within-record permutations101/202/303 as prior run; validation/evaluation labels correct',
    calibration='dynamic output offset unconstrained; raw HR metrics diagnostic only, no claim of absolute calibration; evaluation means only for scoring',
    budget='exactly60 epochs per run,24 runs; save best and epoch60 checkpoints; epoch60 predictions train/val only, never choose via evaluation',
    baseline='unchanged training-global mean/median and same29-bin original-spectrum peak',
    scope='same-subject same-visit cross-triggered-capture; p03 incomplete radar-start logs; overlap precludes formal permutation significance',
    interpretation=['train and unseen-capture improvement with consistent true-over-shuffle advantage supports objective bottleneck',
                    'train-only improvement supports dynamic fitting but insufficient cross-capture stability',
                    'no train improvement OR evaluation close to shuffle: end MLP tuning and audit continuous frontend/time correspondence'],
    historical_comparison='new absolute arm also changes batching, fixes60 epochs and constant lr versus previous early-stop/plateau run; not a one-factor early-stop ablation')



def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def score(f,p):
    result=base_score(f,p)
    centered=f.label_bpm-f.groupby('sample_tag').label_bpm.transform('mean')
    variance=float(np.mean(centered.to_numpy()**2))
    result['centered_r2']=1-result['centered_rmse']**2/variance if variance>1e-10 else None
    return result


def load_data(archive,split_path):
    f,xt,xf,_=load_arrays(archive);s=pd.read_csv(split_path)
    ids=np.flatnonzero(f.sample_tag.isin(s.sample_tag));x=xf[ids]
    g=f.iloc[ids].copy();assert (g.split=='train').all()
    g['split']=g.sample_tag.map(s.set_index('sample_tag').split)
    g=g.rename(columns={'label_heart_rate':'label_bpm'})
    g=g[['sample_tag','group_key','window_start','split','label_bpm']].reset_index(drop=True)
    for tag,indices in g.groupby('sample_tag',sort=True).groups.items():
        record_ids=ids[np.asarray(indices)]
        data=np.concatenate([xt[record_ids].ravel(),xf[record_ids].ravel()]).astype(np.float32)
        # split was originally hashed using time block followed by frequency block.
        expected=s.set_index('sample_tag').loc[tag,'feature_sha256']
        assert hashlib.sha256(data.tobytes()).hexdigest()==expected,tag
    assert len(g)==1728 and g.groupby('sample_tag').size().eq(8).all()
    assert g.groupby('split').size().to_dict()==dict(evaluation=432,train=864,val=432)
    return g,x


def run(data_hash):
    assert Path('/kaggle/working').is_dir(),'Kaggle-only training'
    import torch
    from torch import nn
    start=time.perf_counter();torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True);assert torch.cuda.is_available()
    root=Path(__file__).resolve().parents[2];out=Path('/kaggle/working/ftu_objectives');out.mkdir()
    for name in ['record_split.csv','source_manifest.json']:shutil.copyfile(root/name,out/name)
    (out/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2))
    archive,=Path('/kaggle/input').rglob('ftu_timebase_edacm_v2.zip');assert sha(archive)==data_hash
    f,raw=load_data(archive,root/'record_split.csv');f.to_csv(out/'inputs.csv',index=False)
    hz=np.fft.rfftfreq(256,1/20);band=(hz>=.5)&(hz<=2.75);assert band.sum()==29
    x=raw.copy();x[:,~band]=0
    np.savez_compressed(out/'features.npz',original=raw,masked=x,mask=band)
    y=f.label_bpm.to_numpy();masks={s:(f.split==s).to_numpy() for s in ['train','val','evaluation']}
    mean=float(y[masks['train']].mean());std=float(y[masks['train']].std())
    base=f.copy();base['peak_bpm']=hz[np.flatnonzero(band)[raw[:,band].argmax(1)]]*60
    base['mean_bpm']=mean;base['median_bpm']=np.median(y[masks['train']]);base.to_csv(out/'baselines.csv',index=False)
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda')
    tx=tensor(x)
    records=np.asarray([np.asarray(idx) for _,idx in f[masks['train']].groupby('sample_tag',sort=True).groups.items()])
    assert records.shape==(108,8)
    record_ids=torch.as_tensor(records,device='cuda')
    batch_hashes={};last_predictions=[]
    summary=[];history=[];predictions=[];permutations=[];init={}
    for objective in PROTOCOL['objectives']:
        for seed in PROTOCOL['seeds']:
            for shuffle in [None]+PROTOCOL['shuffle_seeds']:
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                target=y.copy();name=f'{objective}_seed{seed}_'+('true' if shuffle is None else f'shuffle{shuffle}')
                if shuffle is not None:
                    rng=np.random.default_rng(shuffle)
                    for tag,indices in f[masks['train']].groupby('sample_tag',sort=True).groups.items():
                        idx=np.asarray(indices);donors=rng.permutation(idx);target[idx]=y[donors]
                        np.testing.assert_array_equal(np.sort(target[idx]),np.sort(y[idx]))
                        for i,j in zip(idx,donors):permutations.append(dict(run=name,index=int(i),donor=int(j)))
                target=tensor((target-mean)/std)
                net=nn.Sequential(nn.Linear(129,64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,32),nn.GELU(),nn.Dropout(.15),nn.Linear(32,1)).cuda()
                init[name]=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in net.parameters())).hexdigest()
                opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001)
                lossfn=nn.SmoothL1Loss(beta=.5);gen=torch.Generator(device='cuda').manual_seed(seed)
                best=float('inf');batch_digest=hashlib.sha256()
                for epoch in range(1,61):
                    net.train();order=torch.randperm(len(record_ids),device='cuda',generator=gen)
                    batch_digest.update(order.cpu().numpy().tobytes());total_loss=0.
                    for record_batch in order.split(8):
                        batch=record_ids[record_batch]
                        opt.zero_grad();yp=net(tx[batch.flatten()]).reshape(len(batch),8);yt=target[batch]
                        if objective=='dynamic':
                            yp=yp-yp.mean(dim=1,keepdim=True);yt=yt-yt.mean(dim=1,keepdim=True)
                        loss=lossfn(yp,yt);loss.backward()
                        nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                        total_loss+=float(loss.detach())*batch.numel()
                    net.eval()
                    with torch.no_grad():pred=(net(tx).squeeze(1)*std+mean).cpu().numpy()
                    scores={s:score(f[masks[s]],pred[masks[s]]) for s in ['train','val']}
                    for s,m in scores.items():history.append(dict(run=name,epoch=epoch,split=s,train_objective_loss=total_loss/864,lr=.0003,**m))
                    value=scores['val']['centered_rmse']
                    if value<best:
                        best=value
                        torch.save(dict(state_dict=net.state_dict(),epoch=epoch,label_mean=mean,label_std=std,mask=band,run=name),out/f'{name}.pt')

                batch_hashes[name]=batch_digest.hexdigest()
                torch.save(dict(state_dict=net.state_dict(),epoch=60,label_mean=mean,label_std=std,mask=band,run=name),out/f'{name}_epoch60.pt')
                fp=f[masks['train']|masks['val']].copy();fp['run']=name;fp['pred_bpm']=pred[masks['train']|masks['val']];last_predictions.append(fp)
                saved=torch.load(out/f'{name}.pt',map_location='cuda',weights_only=False)
                net.load_state_dict(saved['state_dict']);net.eval()
                with torch.no_grad():pred=(net(tx).squeeze(1)*std+mean).cpu().numpy()
                for s,mask in masks.items():summary.append(dict(run=name,split=s,best_epoch=saved['epoch'],epochs=epoch,**score(f[mask],pred[mask])))
                fp=f.copy();fp['run']=name;fp['pred_bpm']=pred;predictions.append(fp)
                print(name,'epoch',saved['epoch'],'validation centered RMSE',best,flush=True)
    for seed in PROTOCOL['seeds']:assert len({v for k,v in init.items() if f'_seed{seed}_' in k})==1
    for seed in PROTOCOL['seeds']:assert len({v for k,v in batch_hashes.items() if f'_seed{seed}_' in k})==1
    pd.concat(last_predictions).to_csv(out/'epoch60_predictions_train_val.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    pd.concat(predictions).to_csv(out/'predictions.csv',index=False);pd.DataFrame(permutations).to_csv(out/'permutations.csv',index=False)
    (out/'execution.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-start,initialization_hashes=init,batch_order_hashes=batch_hashes,
        data_sha256=data_hash,torch_version=torch.__version__,gpu=torch.cuda.get_device_name()),indent=2))
    (out/'artifact_manifest.json').write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file()},indent=2))
    shutil.make_archive('/kaggle/working/000_ftu_objectives','zip',out)
