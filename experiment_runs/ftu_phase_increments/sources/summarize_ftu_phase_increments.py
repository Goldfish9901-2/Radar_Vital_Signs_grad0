"""Independently verify saved increments/readouts and summarize paired evidence."""
import json,time,hashlib,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

start=time.perf_counter();root=Path('experiment_runs/ftu_phase_increments');source=Path('experiment_runs/ftu_target_audit')
r=pd.read_csv(root/'readouts.csv');r=r[r.scored];q=pd.read_csv(root/'increment_quality.csv');m=pd.read_csv(root/'metrics.csv');records=pd.read_csv(root/'record_metrics.csv')
assert r.groupby(['arm','method','kind']).size().eq(324).all()
hz=np.fft.rfftfreq(256,.05);band=(hz>=.5)&(hz<=2.75)
max_error=0.;identity_error=0.
for path in sorted((root/'signals').glob('*.npz')):
    with np.load(path) as new,np.load(source/'signals'/path.name) as old:
        np.testing.assert_array_equal(new['scored'],old['scored'])
        for arm in ['A','B']:
            for key in ['bins','weights']:np.testing.assert_array_equal(new[arm+'_'+key],old[arm+'_'+key])
            z=old[arm+'_z'].astype(np.complex128)
            # Independent angle difference + principal wrapping, not conjugate-product code.
            angle=np.angle(z[...,1:])-np.angle(z[...,:-1]);angle=(angle+np.pi)%(2*np.pi)-np.pi
            p0=old[arm+'_delta'][...,1:];ratio=abs(z[...,:-1])/abs(z[...,1:])
            identity_error=max(identity_error,float(abs(p0-ratio*np.sin(angle)/(1+1e-6/abs(z[...,1:])**2)).max()))
            for method,target in [('P1',np.sin(angle)),('P2',angle)]:
                err=float(abs(new[f'{arm}_{method}_delta'][...,1:]-target).max());max_error=max(max_error,err);assert err<2e-6
            for method in ['P0','P1','P2']:
                for kind in ['single0','single1','single2','fused']:
                    traces=new[f'{arm}_{method}_signal'] if kind=='fused' else new[f'{arm}_{method}_phase'][:,int(kind[-1])]
                    peak=hz[band][(abs(np.fft.rfft(traces*np.hanning(256)))**2)[:,band].argmax(axis=1)]*60
                    g=r[(r.sample_tag==path.stem)&(r.arm==arm)&(r.method==method)&(r.kind==kind)].sort_values('window_start')
                    np.testing.assert_array_equal(peak[new['scored']],g.peak_bpm)
                    np.testing.assert_allclose(g.label_bpm,[np.nanmean(old['mapped_reference_hr'][s:s+256]) for s in new['starts'][new['scored']]],atol=1e-12)
paired=[]
for arm in ['A','B']:
    for kind in ['fused','single0','single1','single2']:
        base=records[(records.arm==arm)&(records.kind==kind)&(records.method=='P0')].set_index('sample_tag')
        subbase=m[(m.arm==arm)&(m.kind==kind)&m.subject.str.startswith('p')&(m.subject!='pooled')&(m.method=='P0')].set_index('subject')
        # Subject keys may be dataset-prefixed: use actual six keys instead of naming convention.
        subjects=sorted(r.subject.unique());subbase=m[(m.arm==arm)&(m.kind==kind)&(m.method=='P0')].set_index('subject').loc[subjects]
        for method in ['P1','P2']:
            a=records[(records.arm==arm)&(records.kind==kind)&(records.method==method)].set_index('sample_tag').loc[base.index]
            s=m[(m.arm==arm)&(m.kind==kind)&(m.method==method)].set_index('subject').loc[subjects]
            for metric,direction in [('mae',-1),('centered_r2',1),('near_reference_power_fraction',1)]:
                diff=a[metric]-base[metric];sd=s[metric]-subbase[metric]
                keep=diff.index!='p05_Distance_40_cm_r1'
                paired.append(dict(arm=arm,kind=kind,method=method,metric=metric,mean_change=float(diff.mean()),
                    records_improved=int((direction*diff>1e-12).sum()),subjects_improved=int((direction*sd>1e-12).sum()),
                    mean_change_without_prior_max_spike_record=float(diff[keep].mean())))
pd.DataFrame(paired).to_csv(root/'paired_changes.csv',index=False)
qs=q.groupby(['arm','method']).agg(max_abs_delta=('abs_delta_max','max'),median_window_target_max=('abs_delta_max','median'),
    median_cumulative_span=('cumulative_span','median'),max_cumulative_span=('cumulative_span','max'),median_abs_slope=('cumulative_slope',lambda x:abs(x).median())).reset_index()
qs.to_csv(root/'numerical_summary.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(13,4))
subjects=sorted(r.subject.unique())
for ax,metric in zip(axes,['mae','centered_r2','near_reference_power_fraction']):
    for arm in ['A','B']:
        for method in ['P0','P1','P2']:
            g=m[(m.arm==arm)&(m.method==method)&(m.kind=='fused')].set_index('subject').loc[subjects]
            ax.plot(range(6),g[metric],'.-' if arm=='A' else '.--',label=arm+' '+method)
    ax.set(xticks=range(6),xticklabels=[s.rsplit('/',1)[-1] for s in subjects],title=metric);ax.legend(fontsize=6)
fig.tight_layout();fig.savefig(root/'figures'/'subject_comparison.png',dpi=140);plt.close(fig)
with np.load(root/'positive_control.npz') as z:
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    for ax,name in zip(axes,['constant','slow','drop']):
        ax.plot(z['true_delta'],color='black',label='true phase')
        for method in ['P0','P1','P2']:ax.plot(z[name+'_'+method],label=method,alpha=.7)
        ax.set(title=name,xlabel='Sample',ylabel='Increment (rad)');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'figures'/'positive_control.png',dpi=140);plt.close(fig)
def md(df):
    # No optional tabulate dependency.
    def fmt(v):return f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
pooled=m[(m.kind=='fused')&(m.subject=='pooled')][['arm','method','mae','centered_rmse','centered_r2','centered_r','near_reference_power_fraction']]
singles=m[(m.kind!='fused')&(m.subject=='pooled')][['arm','method','kind','mae','centered_r','near_reference_power_fraction']]
pair=pd.DataFrame(paired);f=pair[pair.kind=='fused']
control=pd.read_csv(root/'positive_control.csv');execution=json.loads((root/'execution.json').read_text())
report=f'''# FTU frozen-target phase increment audit

54 条原训练采集、六名受试者，A/B 身份及权重完全复用上轮。每个条件 324 个共同评分窗口；没有重新选目标、调整频带、搜索延时或训练模型。

## 结果与决策

去除幅度比显著限制了异常增量，但未恢复可靠的记录内 HR 跟踪。P1/P2 对固定峰值 MAE 有小幅改善，仍约 31–33 bpm；去均值 R² 均远小于零、相关性接近零。按预先给出的决策，**不重导出、不重训，继续向上游审计复数 RDA 的相位含义、构建方式与参考时间轴**。这不是证明真实信号中不存在心跳信息。

融合信号汇总（bpm；相关性与 R² 是合并记录内去均值后的指标）：

{md(pooled)}

记录内参考标准差只有约 1.177 bpm，因此巨大负 R² 表明峰值起伏远大于标签变化，不能只凭相关性符号判断成功。窗口重叠且仅六人，不按独立窗口做显著性检验。

## 数值正对照与真实增量

{md(control)}

正对照使用完全相同的已知相位轨迹，最大绝对增量小于 1 rad；P1 校验 sin(真实增量)，P2 校验真实增量。恒幅、缓变和骤降均通过 2e-6 容差。P0 在骤降时受幅度比显著放大。零幅度使相邻两个增量未定义，三种方法均显式保留 NaN，不进入后续处理；首样本零仅是窗口边界约定。真实数据共同有效窗口仍为 324，无额外排除。

P0 满足 delta = (前幅度/当前幅度) × sin(相角增量) / (1+epsilon/当前幅度²)。独立重算的最大浮点残差 {identity_error:.8g}；P1/P2 独立相角差校验最大误差 {max_error:.8g}。这证明幅度变化会进入现有公式，不证明它是 HR 失败的唯一原因。

{md(qs)}

累计跨度和斜率仅是异常趋势诊断，不是真实位移。P1 有界于 ±1，P2 有界于 ±π；上限下降部分来自公式本身，不能作为生理信息改善证据。

## 配对收益及单目标/融合

下表变化均为新方法减 P0；改善计数按 MAE 越小、R²/参考附近功率越大计算。最后一列排除上轮已识别的最大异常记录，仅作敏感性分析，主结果保留全部 54 条。

{md(f.drop(columns=['kind']))}

此处 mean_change 是逐记录指标差的等权均值，尤其记录 R² 的平均变化不等于上表合并去均值 R² 的变化；低变化记录的分母很小，不能把这列的巨大数值解释为实质恢复。

三个单目标的合并结果：

{md(singles)}

全部三个单目标及融合的统计在 metrics.csv、record_metrics.csv、paired_changes.csv；没有按 HR 挑最佳单目标。六张示例图按每位受试者固定最小选样哈希取记录，分别显示三个单目标和融合。参考附近功率沿用旧定义（参考 ±1 个频率 bin 的功率除以固定 HR 频带功率），只作事后评分。

![六名受试者配对结果](figures/subject_comparison.png)

![解析正对照](figures/positive_control.png)

## 复现与边界

P0 的增量、累积、去趋势、融合、标准化及频谱逐项核对旧 NPZ，容差 2e-5；实际误差见 p0_reproduction.csv。最初严格逐位校验在频谱出现约 1e-7 级平台差异后停止，记录容差后重跑；没有改变科学协议。P0 峰值和标签完全复现旧表。输入 56 个文件哈希通过上轮清单验证，保存所有新增量/累积/相位/融合/频谱及冻结身份权重。

P2 是复数观测的主值相角增量，不自动等于胸壁位移。低幅度噪声、多散射体、混叠、导出相干性及未实测同步仍未排除。沿用 frame/20+0s 时间假设，未利用 HR 选择目标、峰或延时。该实验是六名已知受试者的记录诊断，不是跨受试者泛化评价。

本地 uv CPU 主运行 {execution['elapsed_sec']:.2f} 秒，无训练。协议及代码与结果在本次一起归档，未做结果前独立提交，不宣称时间戳预注册。完整可重运行命令：`uv run --locked --no-sync python run_ftu_phase_increments.py`，然后运行 `summarize_ftu_phase_increments.py`。
'''
(root/'REPORT.md').write_text(report,encoding='utf-8')
plan=Path('PLAN.md');text=plan.read_text(encoding='utf-8');marker='## 2026-10-18 冻结目标相位增量对照'
if marker not in text:
    plan.write_text(text+'\n\n'+marker+'\n\n54条记录 A/B × P0/P1/P2 无训练对照完成。幅度比去除压低异常增量，但融合峰值 MAE 仍约31–33 bpm、记录内 R² 远小于零，未恢复可靠参考对应。不重训；继续上游复数RDA相位含义/构建与参考时间轴审计。详见 [报告](experiment_runs/ftu_phase_increments/REPORT.md)。\n',encoding='utf-8')
shutil.copyfile(__file__,root/'sources'/Path(__file__).name)
(root/'validation.json').write_text(json.dumps(dict(elapsed_sec=time.perf_counter()-start,independent_angle_difference_max_error=max_error,
    p0_amplitude_identity_max_error=identity_error,all_saved_peaks_recomputed=True,labels_recomputed=True,identities_weights_exact=True,
    scored_windows_per_condition=324,scored_records=54),indent=2))
(root/'artifact_manifest.json').write_text(json.dumps({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'},indent=2))
print(pooled.to_string(index=False));print(f.to_string(index=False));print(qs.to_string(index=False))
