"""Independent local analysis and archive for frozen A-D experiment outputs."""
import hashlib,json,shutil,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.training.audit_predictions import metrics
from src.training.ftu_position_protocol import load_arrays,choose_record
from src.training.ftu_tcn_protocol import donor_indices


def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2),encoding='utf-8')


def main():
    started=time.perf_counter();out=Path('experiment_runs/ftu_position_abcd');out.mkdir(parents=True,exist_ok=True)
    downloaded=Path('tmp/ftu_position_result')
    with zipfile.ZipFile(downloaded/'results.zip') as z:
        expected=json.loads(z.read('artifact_manifest.json'))
        for name,digest in expected.items():assert sha(z.read(name))==digest,name
        for name in z.namelist():
            if name.endswith('/'):continue
            target=(out/name).resolve();assert target.is_relative_to(out.resolve())
            target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(z.read(name))
    shutil.copyfile(out/'artifact_manifest.json',out/'remote_artifact_manifest.json')
    protocol=json.loads((out/'protocol.json').read_text());remote=json.loads((out/'summary.json').read_text())
    archive=Path('experiment_runs/ftu_timebase_edacm_v2/full_artifacts.zip')
    assert sha(archive.read_bytes())==protocol['input_archive_sha256']
    assert sha(Path('experiment_runs/ftu_tcn_p1_seed42/model/best.pt').read_bytes())==protocol['checkpoint_sha256']
    manifest,xt,xf,raw=load_arrays(archive);assert raw==(out/'input_manifest.csv').read_bytes()
    predictions=pd.read_csv(out/'BC_predictions.csv')
    assert len(predictions)==8448 and np.isfinite(predictions[['label_bpm','pred_bpm']]).all().all()
    keys=['split','group_key','sample_tag','window_start']
    for name,g in predictions.groupby('model'):
        orig=g[g.condition=='original']
        joined=orig.merge(manifest[keys+['label_heart_rate']],on=keys,how='outer',indicator=True,validate='one_to_one')
        assert (joined._merge=='both').all()
        np.testing.assert_allclose(joined.label_bpm,joined.label_heart_rate,atol=1e-6)
        test=orig[orig.split=='test'].reset_index(drop=True)
        swap=g[g.condition=='subject_swap'].reset_index(drop=True)
        assert test[keys].equals(swap[keys])
        np.testing.assert_array_equal(test.label_bpm,swap.label_bpm)
        donors=donor_indices(test[['group_key','sample_tag','window_start']])
        np.testing.assert_allclose(swap.pred_bpm,test.pred_bpm.to_numpy()[donors],atol=1e-7)
    # Independent closed-form inference; no model fitting happens locally.
    with np.load(out/'B_ridge.npz') as z:
        train=xf[manifest.split=='train'].astype(np.float64)
        np.testing.assert_allclose(z['scaler_mean'],train.mean(axis=0),atol=1e-8)
        np.testing.assert_allclose(z['scaler_scale'],train.std(axis=0),atol=1e-8)
        ridge=((xf.astype(np.float64)-z['scaler_mean'])/z['scaler_scale'])@z['coef']+z['intercept']
        expected=predictions[(predictions.model=='ridge')&(predictions.condition=='original')].pred_bpm.to_numpy()
        np.testing.assert_allclose(ridge,expected,atol=1e-7,rtol=0)
        coef=z['coef'].copy()
    selection=pd.read_csv(out/'B_alpha_selection.csv')
    assert selection.loc[selection.val_mae.idxmin(),'alpha']==remote['B_alpha']
    history=json.loads((out/'C_history.json').read_text())
    assert min(history,key=lambda r:r['val_mae'])['epoch']==remote['C_best_epoch']
    tag,ids=choose_record(manifest);assert tag==protocol['selected_training_record']==remote['selected_training_record']
    d=pd.read_csv(out/'D_predictions.csv');assert set(d.sample_tag)=={tag}
    np.testing.assert_allclose(d.label_bpm,manifest.iloc[ids].label_heart_rate,atol=1e-7)
    with np.load(out/'D_inputs.npz') as z:
        np.testing.assert_array_equal(z['x_time'],xt[ids]);np.testing.assert_array_equal(z['x_freq'],xf[ids])
    np.testing.assert_allclose(abs(d.pred_bpm-d.label_bpm).mean(),remote['D_best_eval_mae'],atol=1e-5)
    # Reference TCN predictions are frozen P1 evidence, not a new training run.
    prior=pd.read_csv('experiment_runs/ftu_tcn_p1_seed42/predictions.csv');prior['model']='frozen_tcn'
    allpred=pd.concat([predictions,prior],ignore_index=True)
    constants={'train_mean':float(manifest[manifest.split=='train'].label_heart_rate.to_numpy(dtype=np.float32).mean()),
               'train_median':float(manifest[manifest.split=='train'].label_heart_rate.median())}
    tables=[];record_metrics=[]
    for (model,split,condition),g in allpred.groupby(['model','split','condition']):
        tables.append(dict(model=model,split=split,condition=condition,scope='overall',group='all',**metrics(g,constants)))
        centered=g.copy()
        for col in ['label_bpm','pred_bpm']:centered[col]-=centered.groupby('sample_tag')[col].transform('mean')
        tables.append(dict(model=model,split=split,condition=condition,scope='record_centered',group='all',**metrics(centered,{})))
        for group,h in centered.groupby('group_key'):
            tables.append(dict(model=model,split=split,condition=condition,scope='subject_record_centered',group=group,**metrics(h,{})))
        for tag_,h in g.groupby('sample_tag'):
            record_metrics.append(dict(model=model,split=split,condition=condition,sample_tag=tag_,group_key=h.group_key.iloc[0],**metrics(h,constants)))
    table=pd.DataFrame(tables);table.to_csv(out/'tracking_summary.csv',index=False)
    pd.DataFrame(record_metrics).to_csv(out/'record_metrics.csv',index=False)
    response=pd.read_csv(out/'A_position_response.csv');arows=[]
    for (condition,context),g in response.groupby(['condition','context']):
        for band,h in [('all',g),('HR_bins_10_to_32',g[g.bin.between(10,32)])]:
            assert len(h)>1
            arows.append(dict(condition=condition,context=context,band=band,
                output_min=float(h.pred_bpm.min()),output_max=float(h.pred_bpm.max()),
                output_span=float(h.pred_bpm.max()-h.pred_bpm.min()),
                slope_bpm_per_bpm=float(np.polyfit(h.frequency_bpm,h.pred_bpm,1)[0])))
    astats=pd.DataFrame(arows);astats.to_csv(out/'A_response_summary.csv',index=False)
    with np.load(out/'A_embeddings.npz') as z:
        embedding=[]
        for name in z.files:
            h=z[name][10:33];anchor=h[6]
            embedding.append(dict(condition=name,hr_band_max_distance_from_bin16=float(np.linalg.norm(h-anchor,axis=1).max()),
                anchor_norm=float(np.linalg.norm(anchor))))
    write(out/'A_embedding_summary.json',embedding)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=out/'plots';plots.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,condition in zip(axes,['single_bin','gaussian_sigma2','paired_sine']):
        for context,h in response[response.condition==condition].groupby('context'):
            ax.plot(h.frequency_bpm,h.pred_bpm,label=context[:3])
        ax.axvspan(45,150,color='gray',alpha=.15)
        ax.set(title=condition,xlabel='Input peak frequency (bpm)',ylabel='Frozen TCN output (bpm)')
        if condition!='paired_sine':ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(plots/'A_peak_response.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].semilogx(selection.alpha,selection.val_mae,'o-');axes[0].set(xlabel='Ridge alpha',ylabel='Validation MAE')
    axes[1].plot(np.arange(129)*20/256*60,coef);axes[1].axvspan(45,150,color='gray',alpha=.15)
    axes[1].set(xlabel='Frequency (bpm)',ylabel='Ridge coefficient (standardized bins)')
    fig.tight_layout();fig.savefig(plots/'B_ridge.png',dpi=150);plt.close(fig)
    dh=pd.read_csv(out/'D_history.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].semilogy(dh.step,dh.train_mode_mae,label='train mode');axes[0].semilogy(dh.step,dh.eval_mode_mae,label='eval mode')
    axes[0].axhline(.1,color='gray',ls='--');axes[0].legend();axes[0].set(xlabel='Step',ylabel='Record MAE (bpm)')
    axes[1].plot((d.window_start+128)/20,d.label_bpm,'o-',label='Reference')
    axes[1].plot((d.window_start+128)/20,d.pred_bpm,'x--',label='TCN memorization')
    axes[1].set(title=tag,xlabel='Window center (s)',ylabel='HR (bpm)');axes[1].legend()
    fig.tight_layout();fig.savefig(plots/'D_record_fit.png',dpi=150);plt.close(fig)
    # Every test record, no error-selected success examples.
    for subject,g in manifest[manifest.split=='test'].groupby('group_key'):
        fig,axes=plt.subplots(11,4,figsize=(16,24))
        for ax,(tag_,h) in zip(axes.flat,g.groupby('sample_tag',sort=True)):
            h=h.sort_values('window_start');t=(h.window_start+128)/20
            ax.plot(t,h.label_heart_rate,label='Reference')
            for model in ['frozen_tcn','ridge','mlp']:
                p=allpred[(allpred.model==model)&(allpred.condition=='original')&(allpred.sample_tag==tag_)].sort_values('window_start')
                ax.plot(t,p.pred_bpm,label=model)
            ax.set_title(tag_,fontsize=7);ax.tick_params(labelsize=7)
        axes.flat[0].legend(fontsize=6);fig.supxlabel('Window center (s)');fig.supylabel('HR (bpm)')
        fig.tight_layout();fig.savefig(plots/('BC_trajectories_'+subject.rsplit('/',1)[1]+'.png'),dpi=130);plt.close(fig)
    testtable=table[(table.split=='test')&(table.scope=='overall')]
    centered=table[(table.split=='test')&(table.scope=='record_centered')]
    lines=['# FTU 频率位置与拟合诊断 A–D','',
        '固定已有 6/2/2 划分；模型拟合全部在 Kaggle，测试集不参与参数或 checkpoint 选择。',
        '', '**本轮没有恢复可靠的跨受试者动态心率跟踪；也不能把失败仅归因于 TCN 的全局池化。**',
        'A 显示当前 checkpoint 的峰位置响应很弱但非严格为零；B/C 保留绝对频率位置后仍缺少动态跟踪证据；D 证明这套输入和 TCN 在专门的记忆设置下能够拟合八个训练样本。',
        '', '## A：冻结 TCN 的峰位置响应','',
        '固定时域分支为每名训练受试者的首个窗口，移动标准化单峰；另用时频一致的正弦输入对照。',
        '人工频谱可能分布外，固定真实时域还会破坏分支一致性。这是响应诊断，不是心率估计准确率。',
        'TCN 理论感受野为 181，频域长度为 129；全局池化不能直接推出严格的位置不变性。','',
        '| 输入 | HR 区间输出跨度范围 (bpm) | 响应斜率范围 (输出 bpm / 输入 bpm) |','|---|---:|---:|']
    for condition,g in astats[astats.band=='HR_bins_10_to_32'].groupby('condition'):
        lines.append(f'| {condition} | {g.output_span.min():.4f}–{g.output_span.max():.4f} | {g.slope_bpm_per_bpm.min():.4f}–{g.slope_bpm_per_bpm.max():.4f} |')
    lines+=['','## B/C：位置敏感模型','',
        f"Ridge 验证集选择 alpha={remote['B_alpha']}；MLP 为 129→64→32→1，单种子 42，实际 {remote['C_epochs']} epochs，选中 epoch {remote['C_best_epoch']}。",
        'Ridge 选中预先固定网格的上界，提示验证集更偏好强正则收缩；没有根据本轮结果扩大搜索。',
        '129 维输入是现有导出的 Hann/log/窗口标准化频谱，随后按训练集逐 bin 标准化；不是原始复数 FFT。',
        '两模型使用相同训练集 scaler；未卷积、池化或重排频率位置。','',
        '| 模型 | 条件 | 测试 MAE | 记录去均值 r | 记录去均值 R_var |','|---|---|---:|---:|---:|']
    for r in testtable.itertuples():
        c=centered[(centered.model==r.model)&(centered.condition==r.condition)].iloc[0]
        lines.append(f'| {r.model} | {r.condition} | {r.mae:.4f} | {c.pearson_r:.4f} | {c.r_var:.4f} |')
    base=metrics(prior[prior.split=='test'].query("condition == 'original'"),constants)
    lines += ['',f"常数基线：训练均值 MAE={base['train_mean_mae']:.4f}，训练中位数 MAE={base['train_median_mae']:.4f} bpm。",
        'MLP 的 MAE 改善不等于跟踪改善：换成另一名受试者的输入后误差没有变大，记录内相关性仍接近零。',
        'R_var 为标准差比；跨受试者替换使用既定 p01/p08 双射，边际预测分布不变。','',
        '## D：训练记录过拟合','',
        f"选择 {tag}（训练集中标签标准差最大，8 个窗口，std={protocol['selected_training_record_hr_std']:.4f} bpm）。",
        f"最佳 eval-mode MAE={remote['D_best_eval_mae']:.6f} bpm，step={remote['D_best_step']}；共运行 {remote['D_steps']} 步。",
        '这是新初始化的同规模双分支 TCN：关闭 dropout、无权重衰减、全批量 MSE、按该记录标准化标签。',
        '因此它检验小样本记忆与接口/优化可行性；并非原 P1 常规训练设置的等价复现，也不证明跨记录或跨受试者泛化。','',
        '## 证据与边界','',
        '- 保存 A 全 bin 响应与频域嵌入、Ridge 系数与 scaler、MLP/单记录 TCN checkpoint、逐窗口预测及轨迹图。',
        '- 输入归档、冻结 TCN checkpoint、远端产物和运行源码 SHA256 已核对。',
        '- 本地独立重算全部 Ridge 预测，核对验证集选择、全部预测与 manifest 配对及替换映射。',
        '- 仅两个已参与探索的测试受试者、单个 MLP 种子；同步仍是假设。低 MAE 不能替代记录内动态跟踪证据。',
        f"- Kaggle 总计算时间 {remote['elapsed_sec']:.2f} 秒；本地分析 {time.perf_counter()-started:.2f} 秒。",'',
        '[Kaggle A–D 实验](https://www.kaggle.com/code/goldfish9901/ftu-position-abcd-diagnostics)','']
    lines += ['## 原始测试预测的分布与分受试者检查','',
              '| 模型 | 预测均值 | 预测标准差 | 全局 R_var |','|---|---:|---:|---:|']
    for r in testtable[testtable.condition=='original'].itertuples():
        lines.append(f'| {r.model} | {r.pred_mean:.4f} | {r.pred_std:.4f} | {r.r_var:.4f} |')
    lines += ['', '| 模型 | 受试者 | 记录去均值 r | 记录去均值 R_var |','|---|---|---:|---:|']
    for r in table[(table.split=='test')&(table.scope=='subject_record_centered')&(table.condition=='original')].itertuples():
        lines.append(f'| {r.model} | {r.group} | {r.pearson_r:.4f} | {r.r_var:.4f} |')
    lines.append('')
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write(out/'local_validation.json',dict(input_hashes_match=True,manifest_matches=True,
        ridge_predictions_recomputed=True,val_selection_verified=True,donor_mapping_verified=True,
        selected_record_verified=True,elapsed_sec=time.perf_counter()-started))
    shutil.copyfile(downloaded/'kernel.log',out/'kernel.log')
    shutil.copyfile('tmp/ftu_position_local_validation.json',out/'local_protocol_validation.json')
    shutil.copyfile('tmp/ftu_position_upload/kernel-metadata.json',out/'submitted_kernel_metadata.json')
    source=json.loads((out/'source_manifest.json').read_text())['files']
    for name,digest in source.items():
        data=Path(name).read_bytes();assert sha(data)==digest,name
        dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    for name in ['kaggle/prepare_ftu_position.py','summarize_ftu_position.py','validate_ftu_position.py']:
        dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
    write(out/'artifact_manifest.json',{p.relative_to(out).as_posix():sha(p.read_bytes())
        for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})
    print(testtable[['model','condition','mae','r_var']].to_string(index=False))
    print(centered[['model','condition','pearson_r','r_var']].to_string(index=False))
    print(json.dumps(remote))


if __name__=='__main__':main()
