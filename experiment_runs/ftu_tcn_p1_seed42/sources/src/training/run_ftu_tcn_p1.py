"""Kaggle-only single TCN training and frozen-checkpoint inference."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile
import numpy as np
import pandas as pd
from src.training.ftu_tcn_protocol import PROTOCOL, donor_indices


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(expected_archive_hash):
    if not Path('/kaggle/working').is_dir():
        raise RuntimeError('Training is authorized on Kaggle only')
    start=time.perf_counter()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from src.training import train_model
    from src.models.factory import create_model
    from src.training.datasets import build_datasets, resolve_window_path
    from torch.utils.data import DataLoader
    assert torch.cuda.is_available(), 'GPU required for this frozen budget'
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(PROTOCOL['seed']);np.random.seed(PROTOCOL['seed'])
    archives=list(Path('/kaggle/input').rglob('ftu_timebase_edacm_v2.zip'))
    assert len(archives)==1,archives
    assert digest(archives[0])==expected_archive_hash
    data=Path('/kaggle/working/frozen_data');data.mkdir()
    with zipfile.ZipFile(archives[0]) as z: z.extractall(data)
    export=data/'training_exports_ftu_edacm_v2'
    out=Path('/kaggle/working/ftu_tcn_p1');out.mkdir()
    manifest=pd.read_csv(export/'manifest.csv')
    assert len(manifest)==3520
    assert manifest.groupby('group_key').split.nunique().max()==1
    for split,expected in [('train',PROTOCOL['train_subjects']),('val',PROTOCOL['val_subjects']),('test',PROTOCOL['test_subjects'])]:
        actual={int(s.rsplit('/',1)[1]) for s in manifest[manifest.split==split].group_key}
        assert actual==set(expected),(split,actual)
    test=manifest[manifest.split=='test'].reset_index(drop=True)
    donors=donor_indices(test[['group_key','sample_tag','window_start']])
    mapping=test[['group_key','sample_tag','window_start']].copy()
    for col in ['group_key','sample_tag','window_start']:
        mapping['donor_'+col]=test[col].to_numpy()[donors]
    mapping.to_csv(out/'donor_mapping.csv',index=False)
    hashes={}
    for row in manifest.itertuples():
        path=resolve_window_path(row.window_path,export)
        hashes[path.relative_to(export).as_posix()]=digest(path)
        with np.load(path) as z:
            assert z['x_time'].shape==(1,256) and z['x_freq'].shape==(1,129)
            assert np.isfinite(z['x_time']).all() and np.isfinite(z['x_freq']).all()
    (out/'input_window_hashes.json').write_text(json.dumps(hashes,indent=2))
    protocol=dict(PROTOCOL,input_archive_sha256=expected_archive_hash,
        manifest_sha256=digest(export/'manifest.csv'),numpy_version=np.__version__,
        torch_version=torch.__version__,cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),deterministic_algorithms="enabled_warn_only",
        source_manifest_sha256=digest(Path(__file__).resolve().parents[2]/'source_manifest.json'))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    shutil.copyfile(export/'manifest.csv',out/'input_manifest.csv')
    p=PROTOCOL
    sys.argv=['train_model','--model','tcn','--export-dir',str(export),'--output-dir',str(out/'model'),
        '--datasets','FTU','--epochs',str(p['max_epochs']),'--patience',str(p['patience']),
        '--batch-size',str(p['batch_size']),'--num-workers','0','--seed',str(p['seed']),
        '--lr',str(p['lr']),'--weight-decay',str(p['weight_decay']),'--dropout',str(p['dropout']),
        '--hidden-channels',str(p['hidden_channels']),'--num-blocks',str(p['num_blocks']),
        '--kernel-size',str(p['kernel_size'])]
    train_model.main()
    checkpoint_path=out/'model/best.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cuda',weights_only=False)
    checkpoint_hash=digest(checkpoint_path)
    model=create_model('tcn',checkpoint['config']).to('cuda').eval()
    model.load_state_dict(checkpoint['model_state'])
    train,val,test_ds,stats=build_datasets(export,{'FTU'})
    assert abs(stats.mean-checkpoint['label_stats']['mean'])<1e-7
    rows=[];original_test=None
    for split,ds in [('train',train),('val',val),('test',test_ds)]:
        predictions=[]
        with torch.no_grad():
            for batch in DataLoader(ds,batch_size=64,shuffle=False,num_workers=0):
                predictions.extend((model(batch['x_time'].cuda(),batch['x_freq'].cuda())*stats.std+stats.mean).cpu().tolist())
        assert len(predictions)==len(ds)
        for row,pred in zip(ds.rows,predictions):
            rows.append(dict(split=split,condition='original',group_key=row['group_key'],
                sample_tag=row['sample_tag'],window_start=int(row['window_start']),
                label_bpm=float(row['label_heart_rate']),pred_bpm=pred,checkpoint_sha256=checkpoint_hash))
        if split=='test':original_test=np.asarray(predictions)
    # Run the swapped inputs through the same checkpoint, then verify against
    # the corresponding donor predictions (model is sample-independent in eval).
    swapped=[]
    with torch.no_grad():
        for start_idx in range(0,len(donors),64):
            batch=[test_ds[int(i)] for i in donors[start_idx:start_idx+64]]
            xt=torch.stack([b['x_time'] for b in batch]).cuda()
            xf=torch.stack([b['x_freq'] for b in batch]).cuda()
            swapped.extend((model(xt,xf)*stats.std+stats.mean).cpu().tolist())
    np.testing.assert_allclose(swapped,original_test[donors],atol=1e-4,rtol=0)
    for row,pred in zip(test_ds.rows,swapped):
        rows.append(dict(split='test',condition='subject_swap',group_key=row['group_key'],
            sample_tag=row['sample_tag'],window_start=int(row['window_start']),
            label_bpm=float(row['label_heart_rate']),pred_bpm=pred,checkpoint_sha256=checkpoint_hash))
    pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
    (out/'validation.json').write_text(json.dumps(dict(seed=p['seed'],training_runs=1,
        checkpoint_sha256=checkpoint_hash,selected_epoch=int(checkpoint['epoch']),
        epochs_completed=len(json.loads((out/'model/history.json').read_text())),
        predictions=len(rows),subject_swap_is_bijection=True,both_branches_swapped=True,
        donor_predictions_reproduced=True,elapsed_sec=time.perf_counter()-start),indent=2))
    source_root=Path(__file__).resolve().parents[2]
    shutil.copyfile(source_root/'source_manifest.json',out/'source_manifest.json')
    hashes={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
    (out/'artifact_manifest.json').write_text(json.dumps(hashes,indent=2))
    shutil.make_archive('/kaggle/working/000_ftu_tcn_p1','zip',out)
