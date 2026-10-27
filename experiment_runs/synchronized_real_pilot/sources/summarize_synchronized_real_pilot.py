"""Describe one frozen real-data run without selecting better configurations."""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('experiment_runs/synchronized_real_pilot')
DATA = Path('tmp/sync_real_compact/sync_real_pilot')


def table(frame, columns):
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append('| '+' | '.join('NA' if pd.isna(v) else f'{v:.4f}' if isinstance(v, float) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def main():
    for name in ['acquisition.json', 'records.json', 'errors.json', 'execution.json', 'record_metrics.csv', 'window_predictions.csv']:
        shutil.copyfile(DATA/name, ROOT/name)
    for path in DATA.glob('*_beats.csv'): shutil.copyfile(path, ROOT/path.name)
    (ROOT/'signals').mkdir(exist_ok=True)
    for path in DATA.glob('*_phase20.npz'): shutil.copyfile(path, ROOT/'signals'/path.name)
    frame = pd.read_csv(ROOT/'window_predictions.csv')
    metrics = pd.read_csv(ROOT/'record_metrics.csv')
    records = json.loads((ROOT/'records.json').read_text())
    errors = json.loads((ROOT/'errors.json').read_text())
    execution = json.loads((ROOT/'execution.json').read_text())
    validation = json.loads((ROOT/'validation.json').read_text())
    qa = pd.read_csv(ROOT/'reference_qa.csv')
    primary = metrics[metrics.reference_offset_s == 0].copy()
    primary['MAE_gain_vs_mean'] = primary.mean_oracle_mae-primary.mae
    primary['change_gain_vs_constant'] = primary.constant_change_rmse-primary.change_rmse
    primary['residual_gain_vs_trend'] = primary.trend_detrended_rmse-primary.detrended_rmse
    primary.to_csv(ROOT/'primary_effects.csv', index=False)
    subject = primary.groupby('subject').agg(records=('scenario', 'size'),
        mae=('mae', 'mean'), mean_oracle_mae=('mean_oracle_mae', 'mean'),
        trend_oracle_mae=('trend_oracle_mae', 'mean'), change_rmse=('change_rmse', 'mean'),
        constant_change_rmse=('constant_change_rmse', 'mean'), detrended_rmse=('detrended_rmse', 'mean'),
        trend_detrended_rmse=('trend_detrended_rmse', 'mean'),
        reference_residual_std=('reference_residual_std', 'mean'), ecg_lead_disagreement_rmse=('ecg_lead_disagreement_rmse', 'mean')).reset_index()
    subject.to_csv(ROOT/'subject_summary.csv', index=False)
    selected = frame[frame.reference_offset_s == 0]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), squeeze=False)
    subjects = ['GDN0002', 'GDN0007', 'GDN0006']
    for row, pid in enumerate(subjects):
        for col, scenario in enumerate(['Resting', 'Valsalva']):
            ax = axes[row, col]
            g = selected[(selected.subject == pid)&(selected.scenario == scenario)].sort_values('start_s')
            if len(g):
                valid = g.radar_valid & np.isfinite(g.reference)
                t = g.center_s-g.start_s.min()
                ax.plot(t, g.reference, label='ECG primary', color='black')
                ax.plot(t, g.secondary_lead_reference, label='ECG second lead', color='gray', linestyle=':')
                ax.plot(t, g.prediction.where(valid), label='Frozen phase FFT', color='tab:blue')
                good = g[valid]
                if len(good) >= 3:
                    x = good.center_s.to_numpy(); y = good.reference.to_numpy()
                    ax.plot(x-g.start_s.min(), np.polyval(np.polyfit(x, y, 1), x), label='Offline local trend', color='tab:orange', linestyle='--')
                ax.set_ylabel('HR (bpm)'); ax.set_xlabel('Window center relative to scoring start (s)')
            else: ax.text(.5, .5, 'Protocol/interface gate failed\nNo replacement', ha='center', transform=ax.transAxes)
            ax.set_title(pid+' / '+('Rest' if col == 0 else 'Post-Valsalva recovery')); ax.grid(alpha=.25)
    handles, labels = next((ax.get_legend_handles_labels() for ax in axes.flat if ax.get_legend_handles_labels()[0]))
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .965), ncol=4)
    fig.suptitle('Frozen real-data pilot: no HR-guided range, peak, delay or interval selection', y=.995)
    fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(ROOT/'trajectories.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    report = f'''# 同步真实数据：冻结相位/FFT小型试验

2026-10-27。结果前协议提交`46fc91b`。本轮不再使用旧六条FTU记录做合成校准；首次读取公开GUARDIAN临床数据的三个指定受试者、两个固定场景。**当前冻结读数没有显示超过常数/趋势的收益；同时自动参考存在明显双导联不一致，不能把所有分数当作可靠的真实HR性能。** 自动R峰尚未人工裁定，剩余时钟误差无独立数值界，因此也不能宣布细节动态跟踪已确证。

GDN0002两条记录的双导联窗均HR差RMSE约0.005–0.006bpm，其主读数MAE为13.90/11.73bpm，明显劣于同区间均值oracle4.07/1.42bpm，窗间变化和残差误差也更大。这支持此固定读数未显示增量信息，不能推广成所有雷达前端失败。GDN0006/GDN0007的双导联HR差分别约7–9和24bpm，超过参考残差幅度；这些记录的自动参考尚未确立。保留所有描述性分数，不换主导联、调检测器或删除记录来改善成绩。

## 数据、同步与适用范围

来源：[数据论文](https://www.nature.com/articles/s41597-020-00629-5)、[官方数据v2](https://doi.org/10.6084/m9.figshare.12186516.v2)、[官方代码](https://gitlab.com/sven_schellenberger/scidata_phase1)。官方将Gold-code同步序列同时采到雷达与参考系统，依据该序列优化重采样比并校正延时；不通过HR找延时。发布文件已删除同步序列，逐记录剩余时钟误差不能从当前文件独立重测。0秒为主分析，±0.5秒为冻结敏感性，**不是已知误差界，不选最好偏移**。

按首个官方分片ID的固定哈希选GDN0002/GDN0007/GDN0006，选择不查看标签或雷达。原六条FTU和此前已探索Nature2026均不作为新验证；本轮仅有3个独立受试者，其他27名未访问。本地仓库审计未发现这些记录既往分析，这是可追溯的操作性未使用证据，不声称有外部登记证明。

静息在同一仰卧体位；恢复段取Valsalva首轮结束后30–210秒，避开操作本身和下一轮操作。它仍受呼吸/血流动力学恢复影响，不能等同于单独操纵HR，场景说明也不能证明实际没有意外身体运动。不拼接记录。原始数据本地缺失，按运行要求在私有Kaggle CPU取数与计算。

## 固定前端与参考

CW I/Q没有Range/Beam。只用记录开始30秒做直接最小二乘椭圆校准，然后P2主值共轭相位增量累加、抗混叠降采样20Hz。固定校准失败规则，不改算法或换记录；此CW适配器**不同于FTU ADC→RDA，也不是官方加密MATLAB elreko的复现**。每窗25.6秒、步长2秒，线性去趋势、周期Hann、4096点FFT与45–150bpm全局峰；无HR引导的选点/选峰。

参考用发布同步后的原始ECG，固定NeuroKit2 0.2.12自动R峰，主导联按官方源码既定映射，副导联只作诊断。完整窗口目标为每个RR区间的60/RR按时间积分再除窗长，时间锚点为中心；异常RR0.3–2秒之外、缺失值/边界或覆盖不足使该窗口无效，不编辑搏动、填零或外推。两个导联的评分段原始ECG、全部R峰保存在私有Kaggle输出和本地tmp；GitHub保存逐窗参考、R峰与原文件CRC/SHA256。自动峰的正确性不能由目标积分的数学一致性代替。

## 覆盖与失败

指定6条，共下载357,437,510字节，逐成员核对官方ZIP CRC32并记录SHA256，未声称已校验整个2GB分片的MD5。完成{execution['records_completed']}条、失败{execution['records_failed']}条。失败记录保留且不补选：

```json
{json.dumps(errors, ensure_ascii=False, indent=2)}
```

所有偏移取共同参考有效支持，相同雷达窗口与相同有效掩码；相邻变化只计算相邻且连续2秒的窗口。覆盖率见逐记录表，不能把只有较少有效窗口的记录视为完整180秒证据。

## 逐记录主分析（0秒）

误差单位bpm。MAE_gain_vs_mean、change_gain_vs_constant、residual_gain_vs_trend均为“对应对照误差减雷达误差”，正值有利于雷达。常数均值/中位数及OLS时间趋势均使用**同一评分区间标签**离线拟合，是较强的oracle诊断，不是可部署基线。分别对预测/参考在该相同区间去线性趋势，避免上一轮全记录趋势在局部区间抵消的问题。

{table(primary, ['subject','scenario','windows_scored','windows_total','mae','mean_oracle_mae','trend_oracle_mae','MAE_gain_vs_mean','change_gain_vs_constant','residual_gain_vs_trend','detrended_r2'])}

## 参考变化与双导联诊断

{table(primary, ['subject','scenario','reference_std','reference_residual_std','ecg_lead_disagreement_rmse','prediction_std','detrended_rmse','trend_detrended_rmse'])}

残差变化很小、双导联不一致或时钟误差尚不能约束时，只描述均值/频谱或标记动态未获证实，不把低误差当成功。双导联一致仍不能排除共同检测偏差；未知同步误差也不包含在双导联差内。每条原始参考及自动峰须质量审查后才可提升主张。

按评分起点后60秒起固定5秒展示两个原始导联及冻结检测峰，没有根据误差挑片段或改峰。50ms匹配只是诊断容差，不是已验证的时钟/标注误差界：

{table(qa, ['subject','scenario','ECG1_peaks','ECG2_peaks','ECG1_matched_within_50ms_fraction','ECG2_matched_within_50ms_fraction'])}

![原始ECG与冻结R峰快照](ecg_reference_snapshots.png)

## 逐受试者汇总

每名受试者内对已完成场景等权，再在受试者间等权；场景缺失须按覆盖表理解。没有窗口级显著性检验、种子扩样或5/6通过配额。

{table(subject, ['subject','records','mae','mean_oracle_mae','trend_oracle_mae','change_rmse','constant_change_rmse','detrended_rmse','trend_detrended_rmse'])}

![所有冻结记录轨迹](trajectories.png)

## 固定偏移敏感性

{table(metrics, ['subject','scenario','reference_offset_s','windows_scored','mae','detrended_rmse','detrended_r2'])}

## 核验与出口

Kaggle主任务（取数+处理）{execution['elapsed_seconds']:.3f}秒；本地uv环境独立数值核验{validation['elapsed_seconds']:.3f}秒。使用分数搏动计数差独立重算参考窗均值，使用SciPy periodogram重算FFT，另用闭式线性投影重算对照/残差误差；{validation['window_readouts_checked']}个读数，最大参考差{validation['maximum_reference_error_bpm']:.3g}bpm、FFT差{validation['maximum_fft_error_bpm']:.3g}bpm、汇总差{validation['maximum_summary_error']:.3g}。核验通过只证明实现一致，不证明ECG检测、同步剩余误差或生理有效性。

这次一次性试验按冻结配置结束，不在观察结果后追加前端/窗口/阈值搜索，也不扩展受试者追逐通过数。下一决策须同时依据原始误差、窗间变化、残差增量、覆盖及参考质量；条件不足时保留“尚未验证超过常数/趋势的信息”的结论。既有FTU0/6保持，本轮不得用新数据成绩倒改旧结论。
'''
    (ROOT/'REPORT.md').write_text(report, encoding='utf-8')
    source_names = ['run_synchronized_real_pilot.py','verify_synchronized_real_pilot.py',
                    'summarize_synchronized_real_pilot.py','inspect_synchronized_reference.py',
                    'probe_synchronized_dataset.py','prepare_synchronized_real_pilot.py',
                    'kaggle/prepare_synchronized_real_pilot.py']
    (ROOT/'sources').mkdir(exist_ok=True)
    for name in source_names:
        target = ROOT/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    (ROOT/'source_hashes.json').write_text(json.dumps({name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in source_names}, indent=2))
    decision = {'new_radar_records': 6, 'new_subjects': 3, 'training': False, 'old_ftu_synthetic_calibration': False,
                'real_tracking_success_claim': False, 'fixed_readout_increment_demonstrated': False,
                'reference_disagreement_subjects': ['GDN0006', 'GDN0007'],
                'reference_manually_adjudicated': False, 'clock_error_bound_established': False,
                'next_prerequisite': 'Audit ECG reference independently and establish synchronization uncertainty before stronger dynamic claims; preserve other subjects untouched',
                'no_posthoc_detector_frontend_or_subject_replacement': True}
    (ROOT/'decision.json').write_text(json.dumps(decision, indent=2))
    manifest = {str(p.relative_to(ROOT)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name != 'artifact_manifest.json'}
    (ROOT/'artifact_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(primary[['subject','scenario','mae','MAE_gain_vs_mean','change_gain_vs_constant','residual_gain_vs_trend']].to_string(index=False))


if __name__ == '__main__': main()
