"""Preregistered positional feature diagnostics; imports no training framework."""
import io
import zipfile
import numpy as np
import pandas as pd

PROTOCOL = dict(seed=42, split='unchanged historical 6/2/2',
    A=dict(bins=list(range(129)), shapes=['single_bin','gaussian_sigma2'],
           contexts='lexicographically first training window per training subject',
           paired_sine_bins=list(range(10,33)), fs_hz=20, n=256,
           note='Synthetic response is OOD sensitivity, not HR accuracy or proof of causal feature use.'),
    B=dict(alphas=[.01,.1,1.,10.,100.,1000.,10000.], scaler='per-bin training-only StandardScaler',
           selection='minimum validation MAE; ascending-alpha tie break'),
    C=dict(widths=[129,64,32,1], activation='GELU', dropout=.15, epochs=30, patience=8,
           lr=3e-4, weight_decay=1e-4, batch_size=64, loss='SmoothL1 beta=.5',
           optimizer='AdamW', scheduler='ReduceLROnPlateau factor=.5 patience=4',
           clip_grad_norm=1., selection='minimum validation MAE',
           scaler='same training-only per-bin StandardScaler as Ridge'),
    D=dict(selection='training record with maximum population HR std; lexicographic tie break',
           architecture='fresh dual-branch TCN with P1 dimensions; dropout=0 for memorization',
           max_steps=1500, check_every=25, stop_mae_bpm=.1, consecutive_checks=3,
           lr=.001, weight_decay=0., loss='MSE', batch='all 8 windows',
           label_normalization='selected record mean and population std',
           selection_checkpoint='minimum record eval-mode MAE, training windows only',
           note='Memorization sanity test, not validation or generalization'),
    negative_control='B/C: same p01/p08 matched record/window bijection, no HR matching')


def load_arrays(archive):
    with zipfile.ZipFile(archive) as z:
        prefix='training_exports_ftu_edacm_v2/'
        manifest_bytes=z.read(prefix+'manifest.csv')
        frame=pd.read_csv(io.BytesIO(manifest_bytes))
        xt=[];xf=[]
        for row in frame.itertuples():
            name=prefix+'windows/'+row.window_path.split('/windows/',1)[1]
            with np.load(io.BytesIO(z.read(name))) as a:
                xt.append(a['x_time']);xf.append(a['x_freq'][0])
        xt=np.asarray(xt,dtype=np.float32);xf=np.asarray(xf,dtype=np.float32)
        assert xt.shape==(3520,1,256) and xf.shape==(3520,129)
        assert np.isfinite(xt).all() and np.isfinite(xf).all()
        return frame,xt,xf,manifest_bytes


def choose_record(frame):
    train=frame[frame.split=='train']
    variation=train.groupby('sample_tag',sort=True).label_heart_rate.std(ddof=0)
    tag=variation.idxmax()
    indices=np.flatnonzero((frame.sample_tag==tag).to_numpy())
    assert len(indices)==8 and (frame.iloc[indices].split=='train').all()
    return tag,indices


def artificial_spectra(kind):
    if kind=='single_bin':
        x=np.eye(129,dtype=np.float64)
    elif kind=='gaussian_sigma2':
        x=np.exp(-.5*((np.arange(129)[:,None]-np.arange(129)[None,:])/2.)**2)
    else:raise ValueError(kind)
    return ((x-x.mean(axis=1,keepdims=True))/x.std(axis=1,keepdims=True)).astype(np.float32)
