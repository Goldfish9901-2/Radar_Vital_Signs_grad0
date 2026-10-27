"""Report per-record increments and their conditional spread; no aggregate pass quota."""
import hashlib
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path('experiment_runs/ftu_residual_calibration')


def fmt(v):return 'NA' if pd.isna(v) else f'{v:.4f}' if isinstance(v,float) else str(v)


def table(df,cols):
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,r in df.iterrows():lines.append('| '+' | '.join(fmt(r[c]) for c in cols)+' |')
    return '\n'.join(lines)


def spread(row,name):
    return 'NA' if pd.isna(row[name+'_median']) else f"{row[name+'_median']:.3f} [{row[name+'_q05']:.3f}, {row[name+'_q95']:.3f}]"


def main():
    s=pd.read_csv(ROOT/'record_summary.csv');w=pd.read_csv(ROOT/'window_readouts.csv');prec=pd.read_csv(ROOT/'precision_annotations.csv')
    execution=json.loads((ROOT/'execution.json').read_text());validation=json.loads((ROOT/'validation.json').read_text())
    labels={'dynamic_unevaluable_at_this_fft_precision':'当前FFT精度下动态不可评价（操作性标记）',
        'synthetic_stable_increment':'合成配对残差增益稳定',
        'no_stable_residual_evidence_at_this_readout':'此读出未显示稳定残差恢复',
        'trend_control':'趋势对照','insufficient_common_support':'共同窗口不足','no_centered_dynamic_target':'目标无中心化动态'}
    rows=[]
    for _,r in s[(s.length==512)&(s.alpha==1)&(s.readout=='fft')].iterrows():
        rows.append(dict(record=r.record[:3],sigma=r.noise_sigma,R_rms=r.target_centered_rms,
            delta_MAE=spread(r,'mae_gain'),delta_MAE_vs_known_trend=spread(r,'known_trend_mae_gain'),
            residual_R2=spread(r,'residual_r2'),residual_RMSE=spread(r,'centered_residual_rmse'),
            interpretation=labels[r.annotation]))
    paired=pd.DataFrame(rows);paired.to_csv(ROOT/'report_alpha1.csv',index=False)
    levels=[]
    for _,r in s[(s.length==512)&(s.readout=='fft')&(s.noise_sigma==.25)].iterrows():
        levels.append(dict(record=r.record[:3],alpha=r.alpha,R_rms=r.target_centered_rms,
            delta_MAE=spread(r,'mae_gain'),residual_R2=spread(r,'residual_r2'),interpretation=labels[r.annotation]))
    levels=pd.DataFrame(levels)
    decomposition=[]
    for tag,g in w[(w.length==512)&(w.alpha==1)&(w.noise_sigma==0)&(w.seed==20261026)].groupby('record'):
        r=g.target_hr-g.trend_hr
        decomposition.append(dict(record=tag[:3],windows=len(g),target_mean=float(g.target_hr.mean()),
            target_window_std=float(g.target_hr.std(ddof=0)),trend_window_std=float(g.trend_hr.std(ddof=0)),
            residual_mean=float(r.mean()),residual_window_std=float(r.std(ddof=0))))
    decomposition=pd.DataFrame(decomposition);decomposition.to_csv(ROOT/'target_decomposition.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,tag in zip(axes.ravel(),sorted(s.record.unique())):
        a=w[(w.record==tag)&(w.length==512)&(w.alpha==1)&(w.noise_sigma==.25)].copy()
        a['fft_delta']=a.fft_hr-a.fft_trend_hr
        a['oracle_delta']=a.ideal_hr-a.ideal_trend_hr
        for col in ['fft_delta','oracle_delta']:
            a[col]=a[col]-a.groupby('seed')[col].transform('mean')
        target=a[a.seed==20261026].copy();r=target.target_hr-target.trend_hr;r-=r.mean()
        grouped=a.groupby('center_s').fft_delta
        xs=sorted(a.center_s.unique());median=grouped.median().reindex(xs);lo=grouped.quantile(.05).reindex(xs);hi=grouped.quantile(.95).reindex(xs)
        ax.fill_between(xs,lo,hi,alpha=.25,color='tab:blue',label='FFT conditional5-95%')
        ax.plot(xs,median,'o-',ms=3,color='tab:blue',label='FFT paired residual median')
        ax.plot(target.center_s,r,'k--',lw=1.7,label='true residual = phase oracle')
        ax.axhline(0,color='gray',lw=.5);ax.set_title(tag[:3]);ax.set_xlabel('window center / s');ax.set_ylabel('centered residual / bpm')
    axes[0,0].legend(fontsize=7);fig.suptitle('Development synthetic calibration | alpha=1,sigma=.25 | beyond whole-record linear trend')
    fig.tight_layout();fig.savefig(ROOT/'residual_trajectories.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for ax,tag in zip(axes.ravel(),sorted(s.record.unique())):
        a=s[(s.record==tag)&(s.length==512)&(s.noise_sigma==.25)]
        for readout,color in [('ideal','black'),('fft','tab:blue')]:
            g=a[a.readout==readout].sort_values('alpha')
            ax.fill_between(g.alpha,g.mae_gain_q05,g.mae_gain_q95,alpha=.2,color=color)
            ax.plot(g.alpha,g.mae_gain_median,'o-',color=color,label=readout+' paired gain')
        ax.axhline(0,color='gray',ls=':');ax.set_title(tag[:3]);ax.set_xticks([0,.5,1,2]);ax.set_xlabel('residual multiplier alpha');ax.set_ylabel('MAE gain over same-readout trend / bpm')
    axes[0,0].legend(fontsize=8);fig.suptitle('Conditional5-95% spread over32paired phase/noise realizations | no biological confidence intervals')
    fig.tight_layout();fig.savefig(ROOT/'alpha_response.png',dpi=130);plt.close(fig)
    requirements=dict(status='No independent validation dataset accessed or designated by posthoc resplit',
        development_records=sorted(s.record.unique()),freeze_required_before_validation=True,
        requirements=['Untouched records from subjects not used in rule/model development preferred; external dataset acceptable',
            'Same-acquisition radar/reference trigger or measured clock mapping with uncertainty',
            'Raw RR/ECG or explicitly traceable HR generation, filtering, window anchor and latency',
            'Freeze target, frequency readout, record inclusion, primary metrics and conditional uncertainty convention before examining validation scores',
            'Retain all difficult controls and prespecified records; analyze at record/subject level; no window-level sample inflation'],
        existing_FTU_resplit_is_not_independent=True,prior_controls_and_0_01_unchanged=True)
    (ROOT/'independent_validation_requirements.json').write_text(json.dumps(requirements,indent=2))
    noisy=paired[paired.sigma==.25];noiseless=paired[paired.sigma==0]
    precision=prec[(prec.length==512)].copy();precision['record']=precision.record.str[:3]
    report=f'''# 全窗平均HR与趋势之外动态：有限开发校准

完成日期2026-10-27。**全窗平均目标的已知相位理想读数通过；FFT的趋势外增益具有记录依赖，不能概括成普遍动态跟踪能力。** α=1时p03/p04/p05在本合成实验中有稳定配对残差增益；p02被预定操作性精度规则标记为当前FFT下动态不可评价；p07/p09未显示稳定残差恢复。全部六条仍完整报告，不以5/6计数作为出口，不排除p04/p09。

这不是新雷达结果、不是独立验证。原参考准入0/6、旧移位与0.01门槛均保持。研究主张继续限定为待真实验证的窗口平均HR及频谱结构；合成可读出不证明实际回波含有同样结构。

## 结果前固定的目标和相位读数

协议`5623878`。保留以前的全窗逐帧算术平均：

`Y_alpha(s,L) = (1/L) sum[k=s..s+L-1] h_alpha[k]`，`h_alpha=m+alpha*r`，`m`为整条参考样本对时间的OLS线性趋势，`r=HR-m`。只使用现有参考范围内插值，无外推或循环。

已知生成相位定义在帧边界：`phi[k+1]-phi[k]=2*pi*h_alpha[k]/(60*fs)`，所以

`ideal(s,L)=60*(phi[s+L]-phi[s])/(2*pi*(L/fs))=Y_alpha(s,L)`。

这里L个采样间隔对应L个目标样本，而不是用首末样本之间L−1个间隔代替全窗。为与旧的离散目标精确对应，本轮采用每0.05秒频率保持的已知生成器；上一轮使用梯形相位积分，因此**不能把跨轮差异全部归于一个读数改动**。本轮两个读数共享同一个生成信号，内部比较仍配对。

理想读数使用潜在已知相位，不从含噪余弦估计相位；它是目标定义/读出上限的检查，不能当作可部署算法或真实Polar同步准确的证明。最大窗均值误差 **{validation['maximum_phase_mean_error_bpm']:.3g}bpm**，低于预定1e-9容差。

FFT保持已有逐窗去趋势、Hann、4096点显示FFT、45–150bpm全局最大峰。原生bin为2.34375/1.5625bpm（512/768帧），零填充网格0.29296875bpm，不是额外物理分辨率。相位理想值和FFT各自与同一个全窗目标比较。

## 配对设计和相对趋势的增量

固定α=0、0.5、1、2；**只有α=1对应观测参考残差幅度**，放大或缩小均只用于检测能力诊断。单位幅度余弦加噪声σ=0或0.25，32个固定种子20261026–20261057；每记录/种子跨α和噪声水平共享初相位和标准高斯序列。前3个随机对与上轮一致，但生成器相位积分约定按上文改变。

`M`为窗均趋势，`R_alpha=Y_alpha-M`。对每个读数分别定义：

- 配对误差增益：`MAE(pred_0,Y_alpha)-MAE(pred_alpha,Y_alpha)`，正值有利于包含残差的信号。
- 另报告相对已知趋势`M`的误差增益，避免把FFT趋势读出本身的偏差藏起来。
- 配对残差估计：`D_alpha=pred_alpha-pred_0`，与`R_alpha`比较；同时报告原始偏差/RMSE。
- 动态残差R²：对同一记录共同窗口内的`D_alpha`和`R_alpha`各自去均值后，`1-SSE/sum(R_centered²)`。它比较的是**相对趋势后的变化**，而非原始HR总体相关性，也不会把静态残差均值修正称为动态恢复。

窗支持保持原冻结交集：主512帧每条12–14个；768帧仍只有0–1个，不计算有意义的动态R²或拿来替代主分析。保留所有旧−20/−15/−10/−5秒对照与0/1/2秒偏移的FFT诊断，见`unchanged_shift_diagnostics.csv`；没有删除难通过的移位或调整0.01。它们是保留的次要诊断，本轮新问题不重新以旧5/6门判定。

## 不确定性和精度标记的含义

下表每格为中位数`[q05,q95]`，来自每条记录32个**配对相位/噪声**条件，表示条件90%敏感性范围，**不是受试者置信区间**。不对重叠窗口bootstrap，不将种子或窗口计为新的生物样本。参考HR计算/平滑、真实同步及测量误差仍未知，不包含在这些范围内。

结果前固定了FFT操作性精度标记：`max(显示网格/sqrt(12), alpha0跨种子中心化读出抖动的q95)`。网格项约0.08457bpm是此离散读出的报告约定，**不是原生Fourier分辨率、严格误差下界或生理可辨识性定理**。目标中心化残差RMS低于该标尺标为“当前FFT精度下动态不可评价”，仍保存全部原始误差/R²；它不证明参考没有动态，也不用于排除记录或救回5/6。

对于高于标尺的情况，q05(配对MAE增益)>0且q05(残差R²)>0只标注“合成配对增益稳定”；否则报告当前读出未显示稳定证据，而非雷达算法失败。理想读数另行检查，不套用FFT标尺。

## α=1：逐记录结果

含噪σ=.25，所有误差单位bpm；R_rms为目标中心化残差RMS，R²无单位：

{table(noisy,['record','R_rms','delta_MAE','delta_MAE_vs_known_trend','residual_RMSE','residual_R2','interpretation'])}

无噪声条件，用于区分读出偏差和噪声敏感性：

{table(noiseless,['record','R_rms','delta_MAE','residual_RMSE','residual_R2','interpretation'])}

理想读数在所有有足够支持、非零残差的条件下恢复已知残差至数值精度；它也说明本合成构造中的残差不是数学上不存在。FFT在p07可降低全窗误差，却未稳定恢复中心化残差；**降低平均误差与恢复细节动态是两个结果**。p02的极负R²受小目标分母放大，主要按残差RMS/RMSE与精度标记解释，不把其数值大小当作生理失败程度。

![相对固定全记录趋势的中心化残差](residual_trajectories.png)

## 先解释“趋势之外”到底指什么

`m`拟合整条一分钟参考，评分仅覆盖后部共同窗口。全记录OLS残差全局正交于线性趋势，**不保证在这个局部子区间也正交或只包含快速变化**。p03/p04/p05的目标窗间标准差都小于残差标准差，残差可以主要表现为对全记录趋势的局部抵消；尤其p04的原HR窗均值近乎平坦。这些合成增益不能直接宣称为快速心搏细节恢复。没有在结果后另拟合局部趋势来改变主指标。

{table(decomposition,['record','windows','target_mean','target_window_std','trend_window_std','residual_mean','residual_window_std'])}

## 固定倍率响应（σ=.25）

所有记录与倍率均保留。α=2的成功只说明对更强人工残差的检测能力，不能升级α=1或真实雷达结论；α=0是趋势对照，残差R²不定义。

{table(levels,['record','alpha','R_rms','delta_MAE','residual_R2','interpretation'])}

![倍率与配对误差增量](alpha_response.png)

## 明确出口与独立验证边界

目标/相位均值检查已通过，本轮有限实验到此完成。对p03/p04/p05，只能报告已知合成信号下、此固定全记录趋势定义的条件性增益；对p02/p07/p09，不给出稳定FFT细节残差恢复主张。没有所有记录必须过门的新要求，也没有通过排除p04/p09或降低阈值改变旧结论。

**现有FTU真实研究主张继续限定为窗口平均HR与频谱结构。** 这既不是声称真实雷达的平均HR能力已验证，也不是声称原回波没有心跳；本轮没有真实信号测试。

六条记录及其开发受试者已反复使用，不能重划分成“独立验证”。当前没有建立未参与规则开发的封存验证集。若未来采用其他FTU记录/受试者，必须有未查看、未参与开发的证据；优先取得真正未触及受试者的新记录或另一数据集，要求同次触发/时钟映射、原始RR/ECG或可追溯HR生成及延迟，再在看其评分前冻结目标、读出与报告规则。`independent_validation_requirements.json`保存这些要求。本轮不取新数据、不追加模型/参数搜索、不把本开发校准称为独立验证。

此前论文确认的Beat→Flow→CSV链路、原始RR/ECG及HR平滑/同步的未知项仍未解决；32个条件随机种子不能弥补这些生理参考证据的缺失。

## 本地执行、核验和产物

主计算 **{execution['elapsed_seconds']:.3f}s**；独立核验 **{validation['elapsed_seconds']:.3f}s**，均为本地uv，无训练。独立使用中心化闭式回归、另一种相位累计及SciPy periodogram重算：{validation['signals_checked']}条合成信号、{validation['window_readouts_checked']}个窗读数、{validation['paired_metric_checks']}项配对指标、{validation['conditional_quantile_checks']}项条件分位数，以及精度标尺与全部旧移位覆盖。最大trace差{validation['maximum_trace_difference']:.3g}，最大指标差{validation['maximum_metric_difference']:.3g}。旧decision哈希保持。

`signals/*.npz`含参考、m/r、α、真实边界相位、所有配对噪声和trace；`generation_parameters.csv`含随机参数；`window_readouts.csv`含目标/趋势、理想/FFT及配对趋势读数；`paired_metrics.csv`、`record_summary.csv`、`precision_annotations.csv`保留所有分解指标和标记。全部协议、代码、来源哈希、执行时间及产物哈希随报告归档。
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    (ROOT/'sources').mkdir(exist_ok=True)
    for name in ['prepare_ftu_residual_calibration.py','audit_ftu_residual_calibration.py','verify_ftu_residual_calibration.py','summarize_ftu_residual_calibration.py','audit_ftu_structure.py']:
        shutil.copyfile(name,ROOT/'sources'/name)
    hashes={str(f.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(ROOT.rglob('*')) if f.is_file() and f.name!='artifact_manifest.json'}
    (ROOT/'artifact_manifest.json').write_text(json.dumps(hashes,indent=2));print('Wrote report and',len(hashes),'artifact hashes')


if __name__=='__main__':main()
