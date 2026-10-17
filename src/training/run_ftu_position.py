"""Kaggle A-D experiments. No search or retraining based on test outcomes."""
import hashlib,json,os,random,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
from src.training.ftu_position_protocol import PROTOCOL,load_arrays,choose_record,artificial_spectra
from src.training.ftu_tcn_protocol import donor_indices


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def run(data_hash, checkpoint_hash):
    assert Path('/kaggle/working').is_dir(),'Training must run on Kaggle'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from torch import nn
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    import sklearn
    from src.models.tcn import TCNConfig,TCNHeartRateModel
    from src.features.representations import frequency_features
    from src.features.edacm import zscore_1d
    assert torch.cuda.is_available()
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True,warn_only=True)
    def seed():
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    seed();start=time.perf_counter()
    archives=list(Path('/kaggle/input').rglob('ftu_timebase_edacm_v2.zip'));assert len(archives)==1
    paths=list(Path('/kaggle/input').rglob('ftu_tcn_p1/model/best.pt'));assert len(paths)==1
    assert sha(archives[0])==data_hash and sha(paths[0])==checkpoint_hash
    frame,xt,xf,manifest=load_arrays(archives[0]);y=frame.label_heart_rate.to_numpy(dtype=np.float64)
    masks={s:(frame.split==s).to_numpy() for s in ['train','val','test']}
    tag,record=choose_record(frame)
    test_indices=np.flatnonzero(masks['test'])
    donors=donor_indices(frame[masks['test']][['group_key','sample_tag','window_start']])
    out=Path('/kaggle/working/ftu_position_abcd');out.mkdir()
    (out/'input_manifest.csv').write_bytes(manifest)
    write(out/'protocol.json',dict(PROTOCOL,input_archive_sha256=data_hash,checkpoint_sha256=checkpoint_hash,
        selected_training_record=tag,selected_training_record_hr_std=float(y[record].std()),
        torch_version=torch.__version__,numpy_version=np.__version__,sklearn_version=sklearn.__version__,
        gpu=torch.cuda.get_device_name(),deterministic_algorithms='warn_only'))
    checkpoint=torch.load(paths[0],map_location='cuda',weights_only=False)
    frozen=TCNHeartRateModel(TCNConfig(**checkpoint['config'])).cuda().eval()
    frozen.load_state_dict(checkpoint['model_state'])
    mean,std=checkpoint['label_stats']['mean'],checkpoint['label_stats']['std']
    def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device='cuda')
    # A: fixed real time-domain contexts, artificial peak translated over bins.
    contexts=frame[masks['train']].sort_values(['sample_tag','window_start']).groupby('group_key').head(1).index
    arows=[];embeddings={}
    with torch.no_grad():
        for kind in PROTOCOL['A']['shapes']:
            spectrum=artificial_spectra(kind)
            embeddings[kind]=frozen.freq_branch(tensor(spectrum[:,None])).cpu().numpy()
            for index in contexts:
                tx=np.repeat(xt[index:index+1],129,axis=0)
                p=(frozen(tensor(tx),tensor(spectrum[:,None]))*std+mean).cpu().numpy()
                for k,value in enumerate(p):
                    arows.append(dict(condition=kind,context=frame.iloc[index].sample_tag,bin=k,
                        frequency_bpm=k*20/256*60,pred_bpm=float(value)))
        bins=PROTOCOL['A']['paired_sine_bins']
        sine=np.asarray([zscore_1d(np.sin(2*np.pi*k*np.arange(256)/256)) for k in bins])[:,None]
        frequencies=np.concatenate([frequency_features(s)[0][None] for s in sine])
        p=(frozen(tensor(sine),tensor(frequencies))*std+mean).cpu().numpy()
        for k,value in zip(bins,p):
            arows.append(dict(condition='paired_sine',context='synthetic_both_branches',bin=k,
                frequency_bpm=k*20/256*60,pred_bpm=float(value)))
    pd.DataFrame(arows).to_csv(out/'A_position_response.csv',index=False)
    np.savez_compressed(out/'A_embeddings.npz',**embeddings)
    write(out/'A_input_range.json',dict(train_freq_min=float(xf[masks['train']].min()),
        train_freq_max=float(xf[masks['train']].max()),
        synthetic_single_bin_max=float(artificial_spectra('single_bin').max()),
        warning='Fixed-time artificial spectra break branch consistency and can be OOD; paired sine is also synthetic.',
        theoretical_tcn_receptive_field=181))
    # B: train-only scaling and validation-only alpha selection.
    scaler=StandardScaler().fit(xf[masks['train']].astype(np.float64))
    x=scaler.transform(xf.astype(np.float64));candidates=[];best=None
    for alpha in PROTOCOL['B']['alphas']:
        ridge=Ridge(alpha=alpha).fit(x[masks['train']],y[masks['train']])
        mae=float(abs(ridge.predict(x[masks['val']])-y[masks['val']]).mean())
        candidates.append(dict(alpha=alpha,val_mae=mae))
        if best is None or mae<best[0]:best=(mae,alpha,ridge)
    ridge=best[2];ridge_pred=ridge.predict(x)
    pd.DataFrame(candidates).to_csv(out/'B_alpha_selection.csv',index=False)
    np.savez(out/'B_ridge.npz',coef=ridge.coef_,intercept=ridge.intercept_,alpha=best[1],
             scaler_mean=scaler.mean_,scaler_scale=scaler.scale_)
    np.testing.assert_allclose(ridge_pred,x@ridge.coef_+ridge.intercept_,atol=1e-8)
    # C: one position-sensitive MLP, no convolution or pooling.
    seed();c=PROTOCOL['C']
    mlp=nn.Sequential(nn.Linear(129,64),nn.GELU(),nn.Dropout(c['dropout']),
        nn.Linear(64,32),nn.GELU(),nn.Dropout(c['dropout']),nn.Linear(32,1)).cuda()
    xgpu=tensor(x);norm_y=tensor((y-mean)/std)
    train_ids=tensor(np.flatnonzero(masks['train'])).long();val_ids=tensor(np.flatnonzero(masks['val'])).long()
    optimizer=torch.optim.AdamW(mlp.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=4)
    lossfn=nn.SmoothL1Loss(beta=.5);history=[];best_val=float('inf');stale=0
    for epoch in range(1,c['epochs']+1):
        mlp.train();order=train_ids[torch.randperm(len(train_ids),device='cuda')]
        for ids in order.split(c['batch_size']):
            optimizer.zero_grad();loss=lossfn(mlp(xgpu[ids]).squeeze(1),norm_y[ids]);loss.backward()
            nn.utils.clip_grad_norm_(mlp.parameters(),1.);optimizer.step()
        mlp.eval()
        with torch.no_grad():pred=mlp(xgpu).squeeze(1)*std+mean
        val_mae=float(abs(pred[val_ids]-tensor(y[masks['val']])).mean())
        history.append(dict(epoch=epoch,val_mae=val_mae,train_mae=float(abs(pred[train_ids]-tensor(y[masks['train']])).mean())))
        scheduler.step(val_mae)
        if val_mae<best_val:
            best_val=val_mae;stale=0
            torch.save(dict(state_dict=mlp.state_dict(),architecture=c,label_mean=mean,label_std=std,
                scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,epoch=epoch),out/'C_mlp.pt')
        else:stale+=1
        print('C epoch',epoch,'validation MAE',val_mae,flush=True)
        if stale>=c['patience']:break
    write(out/'C_history.json',history)
    saved=torch.load(out/'C_mlp.pt',map_location='cuda',weights_only=False);mlp.load_state_dict(saved['state_dict'])
    mlp.eval()
    with torch.no_grad():mlp_pred=(mlp(xgpu).squeeze(1)*std+mean).cpu().numpy()
    rows=[]
    for name,pred in [('ridge',ridge_pred),('mlp',mlp_pred)]:
        for i,row in frame.iterrows():
            rows.append(dict(model=name,condition='original',split=row.split,group_key=row.group_key,
                sample_tag=row.sample_tag,window_start=row.window_start,label_bpm=y[i],pred_bpm=float(pred[i])))
        for i,donor in zip(test_indices,test_indices[donors]):
            row=frame.iloc[i]
            rows.append(dict(model=name,condition='subject_swap',split='test',group_key=row.group_key,
                sample_tag=row.sample_tag,window_start=row.window_start,label_bpm=y[i],pred_bpm=float(pred[donor])))
    pd.DataFrame(rows).to_csv(out/'BC_predictions.csv',index=False)
    # D: deliberately overfit only the preselected training record.
    seed();d=PROTOCOL['D'];cfg=dict(checkpoint['config']);cfg['dropout']=0.
    overfit=TCNHeartRateModel(TCNConfig(**cfg)).cuda()
    dx=tensor(xt[record]);df=tensor(xf[record,None]);dm=float(y[record].mean());ds=float(y[record].std())
    assert ds>0
    target=tensor((y[record]-dm)/ds)
    optimizer=torch.optim.Adam(overfit.parameters(),lr=d['lr'],weight_decay=0.)
    best_error=float('inf');consecutive=0;dhistory=[]
    for step in range(1,d['max_steps']+1):
        overfit.train();optimizer.zero_grad();prediction=overfit(dx,df)
        loss=((prediction-target)**2).mean();loss.backward();optimizer.step()
        if step%d['check_every']==0 or step==1:
            with torch.no_grad():
                train_pred=overfit(dx,df)*ds+dm
                overfit.eval();eval_pred=overfit(dx,df)*ds+dm
                error=float(abs(eval_pred-tensor(y[record])).mean())
                train_error=float(abs(train_pred-tensor(y[record])).mean())
            dhistory.append(dict(step=step,train_mode_mae=train_error,eval_mode_mae=error))
            if error<best_error:
                best_error=error
                torch.save(dict(state_dict=overfit.state_dict(),config=cfg,label_mean=dm,label_std=ds,
                                step=step,sample_tag=tag),out/'D_tcn.pt')
            consecutive=consecutive+1 if error<d['stop_mae_bpm'] else 0
            if consecutive>=d['consecutive_checks']:break
    saved_d=torch.load(out/'D_tcn.pt',map_location='cuda',weights_only=False)
    overfit.load_state_dict(saved_d['state_dict']);overfit.eval()
    with torch.no_grad():dp=(overfit(dx,df)*ds+dm).cpu().numpy()
    drows=frame.iloc[record][['sample_tag','group_key','window_start']].copy()
    drows['label_bpm']=y[record];drows['pred_bpm']=dp;drows.to_csv(out/'D_predictions.csv',index=False)
    pd.DataFrame(dhistory).to_csv(out/'D_history.csv',index=False)
    np.savez_compressed(out/'D_inputs.npz',x_time=xt[record],x_freq=xf[record],label=y[record])
    write(out/'summary.json',dict(elapsed_sec=time.perf_counter()-start,B_alpha=best[1],
        C_best_epoch=saved['epoch'],C_epochs=len(history),D_best_step=saved_d['step'],D_steps=step,
        D_best_eval_mae=best_error,D_fit_threshold=.1,D_pass=best_error<.1,
        selected_training_record=tag,training_runs={'ridge_candidates':7,'MLP':1,'single_record_TCN':1},
        label_stats=dict(mean=mean,std=std),parameters_C=sum(p.numel() for p in mlp.parameters()),
        parameters_D=sum(p.numel() for p in overfit.parameters())))
    source=Path(__file__).resolve().parents[2]/'source_manifest.json'
    shutil.copyfile(source,out/'source_manifest.json')
    write(out/'artifact_manifest.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    shutil.make_archive('/kaggle/working/000_ftu_position_abcd','zip',out)
