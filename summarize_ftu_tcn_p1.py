"""Local post-hoc validation and tracking audit of the one frozen Kaggle TCN run."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
import pandas as pd
from src.training.audit_predictions import metrics
from src.training.ftu_tcn_protocol import donor_indices


def digest(data):
    return hashlib.sha256(data).hexdigest()


def finite_or_none(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def summarize():
    started=time.perf_counter()
    out=Path('experiment_runs/ftu_tcn_p1_seed42');out.mkdir(parents=True,exist_ok=True)
    downloaded=Path('tmp/ftu_tcn_p1_result')
    with zipfile.ZipFile(downloaded/'results.zip') as z:
        expected=json.loads(z.read('artifact_manifest.json'))
        for name,value in expected.items():
            assert digest(z.read(name))==value,name
        for name in z.namelist():
            if name.endswith('/'):continue
            target=(out/name).resolve();assert target.is_relative_to(out.resolve())
            target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(z.read(name))
    (out/'remote_artifact_manifest.json').write_bytes((out/'artifact_manifest.json').read_bytes())
    protocol=json.loads((out/'protocol.json').read_text())
    validation=json.loads((out/'validation.json').read_text())
    config=json.loads((out/'model/run_config.json').read_text())
    source=Path('experiment_runs/ftu_timebase_edacm_v2/full_artifacts.zip')
    assert digest(source.read_bytes())==protocol['input_archive_sha256']
    manifest=pd.read_csv(out/'input_manifest.csv')
    assert digest((out/'input_manifest.csv').read_bytes())==protocol['manifest_sha256']
    hashes=json.loads((out/'input_window_hashes.json').read_text())
    with zipfile.ZipFile(source) as z:
        assert z.read('training_exports_ftu_edacm_v2/manifest.csv')==(out/'input_manifest.csv').read_bytes()
        for path,value in hashes.items():
            assert digest(z.read('training_exports_ftu_edacm_v2/'+path))==value
    assert len(hashes)==3520
    assert config['config']['time_channels']==config['config']['freq_channels']==1
    assert config['config']['use_frequency_domain']
    assert config['args']['seed']==42 and config['args']['epochs']==30
    assert digest((out/'model/best.pt').read_bytes())==validation['checkpoint_sha256']
    history=json.loads((out/'model/history.json').read_text())
    assert len(history)==validation['epochs_completed']<=30
    assert min(history,key=lambda r:r['val']['mae_bpm'])['epoch']==validation['selected_epoch']
    frame=pd.read_csv(out/'predictions.csv')
    assert len(frame)==4224 and np.isfinite(frame[['label_bpm','pred_bpm']]).all().all()
    assert set(frame.checkpoint_sha256)=={validation['checkpoint_sha256']}
    keys=['split','group_key','sample_tag','window_start']
    original=frame[frame.condition=='original'].copy()
    assert not original.duplicated(keys).any()
    joined=original.merge(manifest[keys+['label_heart_rate']],on=keys,how='outer',indicator=True,validate='one_to_one')
    assert (joined._merge=='both').all()
    np.testing.assert_allclose(joined.label_bpm,joined.label_heart_rate,atol=1e-5,rtol=0)
    test=original[original.split=='test'].reset_index(drop=True)
    swapped=frame[frame.condition=='subject_swap'].reset_index(drop=True)
    assert test[keys].equals(swapped[keys])
    np.testing.assert_allclose(test.label_bpm,swapped.label_bpm,atol=0,rtol=0)
    donors=donor_indices(test[['group_key','sample_tag','window_start']])
    np.testing.assert_allclose(swapped.pred_bpm,test.pred_bpm.to_numpy()[donors],atol=1e-4,rtol=0)
    constants={'train_mean':float(manifest[manifest.split=='train'].label_heart_rate.to_numpy(dtype=np.float32).mean()),
               'train_median':float(manifest[manifest.split=='train'].label_heart_rate.median())}
    np.testing.assert_allclose(constants['train_mean'],config['label_stats']['mean'],atol=1e-6)
    tables=[];record_rows=[]
    for (split,condition), g in frame.groupby(['split','condition']):
        tables.append(dict(scope='overall',group='all',split=split,condition=condition,**metrics(g,constants)))
        for subject,h in g.groupby('group_key'):
            tables.append(dict(scope='subject',group=subject,split=split,condition=condition,**metrics(h,constants)))
            centered=h.copy()
            for col in ['label_bpm','pred_bpm']:
                centered[col]-=centered.groupby('sample_tag')[col].transform('mean')
            tables.append(dict(scope='subject_record_centered',group=subject,split=split,condition=condition,
                               **metrics(centered,{})))
        for tag,h in g.groupby('sample_tag'):
            record_rows.append(dict(sample_tag=tag,group_key=h.group_key.iloc[0],split=split,condition=condition,
                                    **metrics(h,constants)))
        centered=g.copy()
        for col in ['label_bpm','pred_bpm']:
            centered[col]-=centered.groupby('sample_tag')[col].transform('mean')
        tables.append(dict(scope='record_centered',group='all',split=split,condition=condition,**metrics(centered,{})))
    for split,g in original.groupby('split'):
        for name,value in constants.items():
            base=g.copy();base.pred_bpm=value
            tables.append(dict(scope='overall',group='all',split=split,condition=name,**metrics(base,constants)))
    summary=pd.DataFrame(tables);records=pd.DataFrame(record_rows)
    summary.to_csv(out/'tracking_summary.csv',index=False);records.to_csv(out/'record_metrics.csv',index=False)
    overall=metrics(test,constants);control=metrics(swapped,constants)
    reported=json.loads((out/'model/summary.json').read_text())['test']['mae_bpm']
    assert abs(reported-overall['mae'])<1e-4
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=out/'plots';plots.mkdir(exist_ok=True)
    plotdata=test.copy();plotdata['swap']=swapped.pred_bpm.to_numpy()
    for subject,g in plotdata.groupby('group_key'):
        groups=list(g.groupby('sample_tag',sort=True))
        fig,axes=plt.subplots(11,4,figsize=(16,24),squeeze=False)
        for ax,(tag,h) in zip(axes.flat,groups):
            h=h.sort_values('window_start');t=(h.window_start+128)/20
            ax.plot(t,h.label_bpm,label='Reference')
            ax.plot(t,h.pred_bpm,label='TCN')
            ax.plot(t,h.swap,alpha=.7,label='Subject swap')
            ax.axhline(constants['train_mean'],ls='--',color='gray',label='Train mean')
            ax.set_title(tag,fontsize=7);ax.tick_params(labelsize=7)
        axes.flat[0].legend(fontsize=6)
        fig.supxlabel('Window center (seconds)');fig.supylabel('HR (bpm)')
        fig.tight_layout();fig.savefig(plots/('trajectories_'+subject.rsplit('/',1)[1]+'.png'),dpi=130);plt.close(fig)
    tag=test.groupby('sample_tag').label_bpm.std().idxmax()
    h=plotdata[plotdata.sample_tag==tag].sort_values('window_start')
    fig,ax=plt.subplots(figsize=(9,4));t=(h.window_start+128)/20
    for name,col in [('Reference','label_bpm'),('TCN','pred_bpm'),('Subject swap','swap')]:ax.plot(t,h[col],label=name)
    ax.axhline(constants['train_mean'],color='gray',ls='--',label='Train mean')
    ax.set(xlabel='Window center (s)',ylabel='HR (bpm)',title=tag+' (largest reference variation)')
    ax.legend();fig.tight_layout();fig.savefig(plots/'largest_hr_variation.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    limit=max(1,float(np.max(np.abs(np.r_[test.pred_bpm-test.label_bpm,
        swapped.pred_bpm-test.label_bpm,constants['train_mean']-test.label_bpm]))))
    for name,pred in [('TCN',test.pred_bpm),('Subject swap',swapped.pred_bpm),('Train mean',constants['train_mean'])]:
        axes[0].hist(pred-test.label_bpm,bins=np.linspace(-limit,limit,41),alpha=.45,label=name)
    axes[0].legend();axes[0].set(xlabel='Signed error (bpm)',ylabel='Windows')
    axes[1].scatter(test.label_bpm,test.pred_bpm,s=8,alpha=.4)
    axes[1].plot([test.label_bpm.min(),test.label_bpm.max()],[test.label_bpm.min(),test.label_bpm.max()],'k--')
    axes[1].set(xlabel='Reference HR',ylabel='TCN HR')
    fig.tight_layout();fig.savefig(plots/'errors.png',dpi=150);plt.close(fig)
    original_center=summary[(summary.split=='test')&(summary.scope=='record_centered')&(summary.condition=='original')].iloc[0]
    swap_center=summary[(summary.split=='test')&(summary.scope=='record_centered')&(summary.condition=='subject_swap')].iloc[0]
    result=dict(local_elapsed_sec=time.perf_counter()-started,all_3520_input_hashes_match=True,
        predictions_match_manifest=True,same_checkpoint=True,donor_predictions_match=True,
        constants=constants,test=overall,subject_swap=control,delta_mae=control['mae']-overall['mae'],
        within_record_r=finite_or_none(original_center.pearson_r),within_record_r_swap=finite_or_none(swap_center.pearson_r),
        within_record_r_var=finite_or_none(original_center.r_var),within_record_r_var_swap=finite_or_none(swap_center.r_var),
        training_runs=1)
    (out/'local_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    shutil.copyfile(downloaded/'kernel.log',out/'kernel.log')
    shutil.copyfile('tmp/ftu_tcn_protocol_validation.json',out/'local_protocol_validation.json')
    shutil.copyfile('tmp/ftu_tcn_p1_upload/kernel-metadata.json',out/'kernel-metadata.json')
    sources=json.loads((out/'source_manifest.json').read_text())['files']
    for name,value in sources.items():
        data=Path(name).read_bytes();assert digest(data)==value
        dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    for name in ['kaggle/prepare_ftu_tcn_p1.py','summarize_ftu_tcn_p1.py','validate_ftu_tcn_protocol.py']:
        dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
    lines=['# FTU P1：单种子双分支 TCN','',
        '**本轮未观察到可靠的记录内动态心率跟踪；不满足扩展到更多模型或跨域实验的证据要求。**',
        '输出确实随输入变化，但与真实记录内 HR 变化近乎不相关。这个结论只针对本次固定配置、划分和预算，不能证明 FTU 不可学习。', '',
        '固定种子 42、最多 30 epochs、早停耐心值 8；仅训练一次，按验证集 MAE 选 checkpoint。',
        f"实际训练 {validation['epochs_completed']} epochs，选择 epoch {validation['selected_epoch']}。",
        '最早 epoch 被验证集 MAE 选中；后续训练没有改善该选择指标。未按测试集相关性改选 checkpoint。',
        '输入为已验证的 EDACM-only v2；单通道时域和频域，保持历史 6/2/2 受试者划分。',
        '', '| 测试条件 | MAE (bpm) | 全局 R_var |','|---|---:|---:|',
        f"| TCN | {overall['mae']:.4f} | {overall['r_var']:.4f} |",
        f"| 跨受试者双分支替换 | {control['mae']:.4f} | {control['r_var']:.4f} |",
        f"| 训练均值 | {overall['train_mean_mae']:.4f} | 0 |",
        f"| 训练中位数 | {overall['train_median_mae']:.4f} | 0 |",'',
        'R_var 定义为标准差比 σ(pred)/σ(HR)，不是方差之比。全局统计不能单独说明动态跟踪。',
        '', '| 记录内去均值后汇总 | Pearson r | R_var |','|---|---:|---:|',
        f"| 原始 | {original_center.pearson_r:.4f} | {original_center.r_var:.4f} |",
        f"| 替换 | {swap_center.pearson_r:.4f} | {swap_center.r_var:.4f} |",'',
        f"替换 ΔMAE = {result['delta_mae']:+.4f} bpm；受试者与逐记录指标见 tracking_summary.csv / record_metrics.csv。",
        '', '## 协议和边界','',
        '- p01 与 p08 按记录后缀、窗口位置一一交换；不参考 HR、误差或模型输出，两条分支同步替换。',
        '- 原始与负对照使用同一 best.pt；原标签保持不变。替换预测已与对应供体预测核对。',
        '- 替换是双射，因此整体预测分布与全局 R_var 不变；重点比较标签对应关系、MAE 和记录内相关性。',
        '- 保留条件与时间位置的替换是保守诊断，共享条件变化可能使替换后仍有相关性。',
        '- 测试仅两名已参与探索的受试者；每条记录八个重叠窗口。不能据此推断总体泛化或显著性。',
        '- 雷达起始偏移仍假定为 0 秒；结果不是硬件同步验证。',
        '- 固定随机种子，确定性算法采用 warn_only；不承诺跨硬件位级一致。',
        '- AdamW、SmoothL1(beta=0.5)、梯度裁剪 1.0、ReduceLROnPlateau(factor=0.5, patience=4)；未用测试结果调参。',
        '', '## 产物与复核','',
        '- model/best.pt：验证集选择的 checkpoint；final.pt 保存相同最佳权重及测试指标。',
        '- predictions.csv：全部训练/验证/测试原始预测及测试负对照，共 4,224 行。',
        '- plots/：两名测试受试者的全部记录轨迹、最大真实 HR 变化记录及误差图。',
        '- protocol.json、model/run_config.json、source_manifest.json、input_window_hashes.json：完整配置及来源。',
        '- 独立核对 3,520 个输入窗口哈希、划分与标签、checkpoint 哈希和替换映射。',
        f"- Kaggle 全流程 {validation['elapsed_sec']:.2f} 秒；本地统计与校验 {result['local_elapsed_sec']:.2f} 秒。",'',
        '[Kaggle 实验](https://www.kaggle.com/code/goldfish9901/ftu-edacm-tcn-p1-seed42)','']
    lines += ['## 各测试受试者的记录内动态指标','',
              '| 受试者 | 条件 | 记录去均值 r | 记录去均值 R_var | 逐记录 r 中位数 |',
              '|---|---|---:|---:|---:|']
    for row in summary[(summary.split=='test')&(summary.scope=='subject_record_centered')].itertuples():
        selected=records[(records.group_key==row.group)&(records.condition==row.condition)]
        lines.append(f'| {row.group} | {row.condition} | {row.pearson_r:.4f} | {row.r_var:.4f} | {selected.pearson_r.median():.4f} |')
    lines.append('')
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    artifacts={str(p.relative_to(out)).replace('\\','/'):digest(p.read_bytes())
               for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    (out/'artifact_manifest.json').write_text(json.dumps(artifacts,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':summarize()
