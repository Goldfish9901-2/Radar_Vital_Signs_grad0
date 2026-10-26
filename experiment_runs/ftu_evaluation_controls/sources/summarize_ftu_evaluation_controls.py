"""Summarize frozen synthetic calibration and scoped synchronization provenance."""
import hashlib
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path('experiment_runs/ftu_evaluation_controls')


def table(df,cols):
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,r in df.iterrows():
        lines.append('| '+' | '.join('NA' if pd.isna(r[c]) else f'{r[c]:.4f}' if isinstance(r[c],float) else str(r[c]) for c in cols)+' |')
    return '\n'.join(lines)


def main():
    m=pd.read_csv(ROOT/'metrics.csv');g=pd.read_csv(ROOT/'score_rule_diagnostics.csv')
    w=pd.read_csv(ROOT/'window_metrics.csv');tally=pd.read_csv(ROOT/'noiseless_tally.csv')
    execution=json.loads((ROOT/'execution.json').read_text());v=json.loads((ROOT/'validation.json').read_text())
    a=m[(m['mode']=='primary')&(m.length==512)&(m.offset==0)&(m['shift']==0)]
    means=a.groupby(['generator','noise_sigma'])[['mae','within5','near_power','centered_rmse','std_ratio']].mean().reset_index()
    p=g[(g['mode']=='primary')&(g.length==512)&(g.noise_sigma==0)&(g.generator=='positive')&(g.seed==20261026)].copy()
    p['record']=p.record.str[:3]
    matched=g[(g['mode']=='matched_secondary')&(g.length==512)&(g.noise_sigma==0)].groupby(['seed','offset']).score_rule_pass.sum().rename('passed_records').reset_index()
    robustness=g[(g['mode']=='primary')&(g.length==512)&(g.noise_sigma==.25)].groupby(['generator','seed']).agg(offset_passes=('score_rule_pass','sum')).reset_index()
    recordpositive=a[(a.noise_sigma==0)&(a.generator=='positive')&(a.seed==20261026)].copy();recordpositive['record']=recordpositive.record.str[:3]
    wm=w[(w['mode']=='primary')&(w.generator=='positive')&(w.noise_sigma==0)&(w.length==512)&(w.offset==0)&(w['shift']==0)]
    readout=[]
    for tag,d in wm.groupby('record'):
        readout.append(dict(record=tag[:3],peak_vs_mean=float(abs(d.peak_bpm-d.mean_generation_hr).mean()),
            peak_vs_hann_energy_mean=float(abs(d.peak_bpm-d.hann_energy_mean_generation_hr).mean())))
    pd.DataFrame(readout).to_csv(ROOT/'readout_alignment.csv',index=False)
    colors=dict(positive='tab:blue',constant='tab:orange',trend='tab:green')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,tag in zip(axes.ravel(),sorted(a.record.unique())):
        rows=w[(w.record==tag)&(w['mode']=='primary')&(w.noise_sigma==0)&(w.length==512)&(w.offset==0)&(w['shift']==0)&(w.seed==20261026)]
        for kind,vals in rows.groupby('generator'):
            ax.plot(vals.center_s,vals.peak_bpm,'o-',ms=3,lw=1,label=kind,color=colors[kind])
        vals=rows[rows.generator=='positive'];ax.plot(vals.center_s,vals.label_hr,'k--',lw=1.6,label='window mean HR')
        ax.set_title(tag[:3]);ax.set_xlabel('segment center / seconds');ax.set_ylabel('bpm')
        ax.ticklabel_format(useOffset=False)
    axes[0,0].legend(fontsize=8);fig.suptitle('Synthetic only: noiseless,512frames,+0s,seed20261026 | global band peak vs window-mean HR')
    fig.tight_layout();fig.savefig(ROOT/'frequency_readout.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for ax,tag in zip(axes.ravel(),sorted(g.record.unique())):
        vals=g[(g.record==tag)&(g['mode']=='primary')&(g.generator=='positive')&(g.length==512)&(g.noise_sigma==0)]
        for seed,part in vals.groupby('seed'):
            ax.plot(part.offset,part.minimum_power_advantage,'o-',label=str(seed),alpha=.7)
        ax.axhline(.01,color='black',ls='--',label='frozen threshold');ax.set_title(tag[:3])
        ax.set_xticks([0,1,2]);ax.set_xlabel('assumed reference offset / s');ax.set_ylabel('true minus strongest control power')
    axes[0,0].legend(fontsize=7);fig.suptitle('Synthetic positive: minimum power advantage over EACH of4fixed shifts; no radar scores')
    fig.tight_layout();fig.savefig(ROOT/'power_advantages.png',dpi=130);plt.close(fig)
    evidence=[dict(question='同次采集触发与时钟映射',status='部分过程说明；逐记录关系未解决',
        source='https://arxiv.org/html/2405.12659v1',sections='VI-A and VI-C2',
        evidence='One operator ran mmWave Studio; another ensured synchronized measurements via Polar Beat. Reference recorded with Beat, saved/accessed via Flow.',
        missing='Shared trigger/event IDs, clock discipline, precise per-record start mapping, transport latency and uncertainty are not specified in the examined sections. Flow synchronization must not be equated with radar/Polar clock synchronization.'),
        dict(question='原始RR间期/ECG是否存在',status='传感器具备能力；本次采集/发布未确认',
        source='https://raw.githubusercontent.com/polarofficial/polar-ble-sdk/master/documentation/products/PolarH10.md',
        evidence='Current SDK documents HR and RR intervals in ms and ECG130Hz; dataset paper describes minute-long time+HR CSVs, not released raw ECG/RR.',
        missing='Whether raw RR/ECG were recorded by the actual Beat/app/firmware version, retained, or separately released; current device capability does not prove historical acquisition. RR interval is distinct from respiratory-rate reference.'),
        dict(question='HR计算/平滑与导出',status='导出路径已明确；算法、时窗和平滑/延迟未知',
        source='https://arxiv.org/html/2405.12659v1',sections='VI-C2',
        evidence='Beat to Flow website Diary Export session CSV, approximately2KB, initial metadata3rows then time+integer HR; stops from00:00:59 to00:01:02.',
        missing='Instantaneous60/RR vs averaged beats, effective filter/window, artifact rejection, resampling, temporal anchor and latency are not given in the examined sources. A1Hz CSV is not evidence of unsmoothed beat-to-beat HR.'),
        dict(question='Polar Flow官方导出说明',status='可确认导出选项；不确认本数据集HR处理',
        source='https://support.polar.com/en/export-training-sessions-flow',
        evidence='Documents Diary Export, CSV/TCX/FIT/GPX options and HR export; examined page does not specify this dataset HR computation/smoothing.',
        missing='Need actual acquisition settings, app and firmware versions and data-owner confirmation.')]
    (ROOT/'provenance'/'evidence.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False),encoding='utf-8')
    questions=dict(purpose='Concrete follow-up questions only; no contact/message sent',questions=[
        'For each frozen record, how were radar and Polar Beat recording starts/stops coordinated? Is there a shared trigger/event, host/device time log, or measured offset/uncertainty?',
        'Were original R-R intervals or raw ECG acquired for those exact sessions? If so, where are files and device timestamps, and how do they map to the exported CSV/session IDs?',
        'Which H10 firmware, Beat/Flow versions, recording modes and export options were used?',
        'How were reported HR values computed from beats: averaging/filter length, artifact rejection, resampling, temporal anchor and effective delay?',
        'Do longer same-session reference records exist beyond the minute-long CSVs, without concatenating independent captures?'])
    (ROOT/'provenance'/'followup_questions.json').write_text(json.dumps(questions,indent=2))
    report=f'''# FTU 评估灵敏度正负对照（开发用合成校准）

**现有完整评分规则未通过此次灵敏度校准。** 无噪声、已知HR驱动的正对照，在正确时间原点下每个种子仅4/6条通过，要求0/1/2秒全部通过时为3/6，未达预定5/6。恒频/趋势对照在正确原点下均未通过完整规则，但所有记录的±5bpm命中率仍为100%。因此不能靠该命中率或小MAE证明动态跟踪，也不能把理想正对照的失败归咎于雷达bin。

此前参考准入**0/6结论保持不变**；本轮只校准评分器，未读取新雷达数组，未重跑旧雷达谱图，未训练或扩大信号区域。合成分数不是雷达性能。

## 冻结协议与合成方法

结果前协议`a37e447`。六条既有参考和共同窗口保持不变，20Hz、512/768帧、Hann、4096点FFT显示、45–150bpm。直接调用`audit_ftu_structure.spectrum/scalar_metrics`，保持逐窗线性去趋势、最大band峰、±一个原生Fourier bin参考附近功率定义。512/768原生bin为2.34375/1.5625bpm；0.29296875bpm零填充网格不是额外分辨率。这里检验的是representation级周期信号与评分，**不是ADC/RDA/胸壁相位恢复**。

三类信号均为`cos(φ0+2π∫f(t)/60 dt)`：正对照`f=线性插值HR(t)`；恒频`f=整条参考HR算术均值`；趋势`f=整条参考HR对时间的OLS直线`，去除残差。积分采用20Hz梯形规则，不外推、不跨缺口。记录均值和趋势故意使用整条参考，是带标签信息的诊断对照，不是可部署基线。趋势本来就可能与HR相关；它检验的是细节动态是否有额外价值，不能称为与HR独立的纯噪声。

三个预定种子20261026/27/28，单位幅度余弦；噪声标准差0和0.25。每条记录/种子跨生成器配对初相位和同一高斯噪声。保存180条合成trace、生成频率、噪声和完整参数。主实验固定生成原点0，报告所有三个评分偏移；次分析仅对正对照按+1/+2秒重新生成并用对应偏移评分，以区分已知原点错配和评分灵敏度，不能替代主分析。没有搜索初相位、噪声、频点或延时。

512帧使用此前冻结的共同窗口（每条12–14个），对照−20/−15/−10/−5秒，所有偏移与对照同窗口。768帧仍仅0–1个共同窗口，不允许作为通过判断或缩短窗长补救。

## 评分规则与准入门分开报告

评分器通过条件仍为：覆盖率≥.9、±5bpm峰值命中率≥.6、参考附近功率比**每个**对照至少高.01、MAE小于**每个**对照；至少5共同窗口。`score_rule_pass`是这些条件的合成诊断，**不绕过此前标签可辨别性准入门**。`formal_real_data_pass`始终False。

此次校准预定要求：无噪声正对照每个种子在正确原点及全偏移判定均≥5/6；恒频/趋势在正确原点不得有任一记录通过。只做开发校准，不选择替代阈值或删除难通过对照。

## 无噪声完整规则通过数

{table(tally,['generator','seed','offset0_passes','all_offsets_passes'])}

正对照按匹配原点另行生成的次分析：

{table(matched,['seed','offset','passed_records'])}

匹配原点后仍只有4/6，说明问题不只是把已知0秒信号拿去对+1/+2秒假设评分。正确原点失败的正对照是p04/p09；主分析+2秒下p05另因功率优势不足而失败。恒频在所有单偏移均无通过；趋势在+2秒有1条记录（每个种子相同）通过单偏移规则，但无记录通过全偏移完整规则。因此本次直接证据是**灵敏度不足**；不能宣称完整规则已出现恒频/趋势假阳性。

## 单指标为何会“看起来很好”

主512帧、正确原点，先逐记录/种子计算再等权平均；命中率是比例：

{table(means,['generator','noise_sigma','mae','within5','near_power','centered_rmse','std_ratio'])}

所有条件的命中率均1.0。恒频无噪声MAE约2.096bpm、预测动态标准差比0，却也全部命中；趋势MAE约0.839bpm。不能把这种小MAE/命中率解释为细节动态恢复。分布收缩、慢趋势和宽容差在此都可能产生表观良好成绩。

正对照逐记录动态诊断（种子20261026；其余种子同向结论）：

{table(recordpositive,['record','mae','centered_rmse','r','std_ratio','near_power'])}

例如p04的正对照MAE仅约0.1745bpm，峰值与窗均值的相关性却约−0.5855。它是已知HR驱动信号，不是没有生理对应；微小窗均值变化与有限谱峰读数不完全一致，会使相关性/方差比变得不稳定。p09恒频MAE约0.3705bpm，略低于正对照约0.3786bpm，差异远小于原生bin。不能以此声称恒频模型学会了动态。

![合成峰值与窗均值](frequency_readout.png)

## 窗均值与最大谱峰并非同一统计量

正对照中`f(t)`由参考确定，窗内信号可能是变频波形。全窗算术平均HR不必等于Hann谱最大峰；Hann还改变了时间权重。另保存Hann平方权重的频率均值，仅用于解释，并没有据此替换评分标签或选择新规则：

{table(pd.DataFrame(readout),['record','peak_vs_mean','peak_vs_hann_energy_mean'])}

原规则正对照的逐记录/偏移最小功率与MAE优势（相对最难超过的固定对照）：

{table(p,['record','offset','minimum_power_advantage','minimum_mae_advantage','power_pass','mae_pass','score_rule_pass'])}

![理想正对照的最小功率优势](power_advantages.png)

负优势不是生成频率没有采用HR，而是当前窗口统计/峰读数/对照难度组合未稳定识别已知对应。0.25噪声结果完整保存，没有根据结果调噪声或阈值。重叠窗口不作为独立样本；三个种子是相位/噪声重复，不是新增受试者。

## 同步与参考生成资料：具体查到了什么

1. **触发关系：**[数据论文VI-A](https://arxiv.org/html/2405.12659v1)描述一名操作员运行mmWave Studio，另一名操作员通过Polar Beat确保同步测量。这提供人工协调采集的过程说明，但未给出共享硬件触发、共同事件ID、逐记录时钟映射及偏差不确定度。VI-C2的“Polar Flow synchronization”描述记录保存/访问链路，不能据此认定雷达与Polar时钟同步成立。
2. **原始RR/ECG：**[Polar官方H10 SDK产品文档](https://raw.githubusercontent.com/polarofficial/polar-ble-sdk/master/documentation/products/PolarH10.md)列出RR间期（ms）和130Hz ECG能力。论文VI-C2只描述约2KB、时间+HR的分钟CSV；目前没有本次六采集实际录制/保存/发布RR或ECG的证据。当前SDK能力不证明2023年Beat/固件记录了这些流。呼吸频率参考中的RR缩写不等于心搏R–R间期。
3. **HR导出与平滑：**论文明确Beat→Flow网站Diary→Export session CSV，前三行为元数据，随后为时间与HR，停止时间59–62秒。这与已有原CSV一致。[Polar官方Flow导出说明](https://support.polar.com/en/export-training-sessions-flow)也说明CSV/TCX/FIT等导出选项。检查到的资料未规定本数据集HR究竟为60/RR、多少拍/多少秒均值、伪迹剔除、平滑方式、时间锚点或延迟。**1Hz输出不等于瞬时、未平滑的逐搏HR。**

原网页、正文、URL/重定向及SHA256保存在`provenance/`。初次旧Polar链接转到404，正确链接随后一次超时再成功；失败请求不作为算法证据。未重下雷达、未把顶层清单/QuickOpen缺项当作不存在证明，也未联系作者。`provenance/followup_questions.json`列出可向数据持有者核对的具体问题：同次触发/时间映射、原始RR/ECG及session匹配、Beat/Flow/固件版本、HR计算平滑与延迟、是否有同次更长参考。

## 决策

保留上一轮0/6；本轮合成校准也未达预定标准，暂停真实动态跟踪主张、更多bin搜索及训练。下一步若设计新规则，应仅在明确标记的开发数据上研究窗口统计、参考时间锚点和动态信息增益，再冻结到独立记录验证；不能用本轮结果删除困难对照或宣称新规则已获独立验证。本轮没有自动重设计/上线新评分规则。

若最终仅能取得短时、1Hz、微小变化的参考，仍可限定条件研究平均HR及频谱结构；动态跟踪需要更合适的同步参考验证数据。这不证明原始回波缺少心跳，也不把传感器能力当作已取得的参考信号。

## 执行与独立核验

本地uv主诊断{execution['elapsed_seconds']:.3f}s，独立核验{v['elapsed_seconds']:.3f}s。验证180条合成信号、{v['independent_metric_checks']}项指标、{v['independent_score_gate_checks']}项可评分门控；独立梯形累加与SciPy periodogram重算，最大trace差{v['maximum_signal_difference']:.3g}，指标差{v['maximum_metric_difference']:.3g}。此前decision文件哈希保持不变。未进行NN训练或读取新雷达数据。所有配置、生成参数、trace、逐窗结果、源代码与哈希归档。
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    (ROOT/'sources').mkdir(exist_ok=True)
    for name in ['prepare_ftu_evaluation_controls.py','audit_ftu_evaluation_controls.py','verify_ftu_evaluation_controls.py','query_ftu_evaluation_provenance.py','summarize_ftu_evaluation_controls.py','audit_ftu_structure.py']:
        shutil.copyfile(name,ROOT/'sources'/name)
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}
    (ROOT/'artifact_manifest.json').write_text(json.dumps(hashes,indent=2));print('Wrote report and',len(hashes),'artifact hashes')


if __name__=='__main__':main()
