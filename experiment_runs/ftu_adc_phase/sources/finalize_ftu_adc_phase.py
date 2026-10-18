"""Publish observed ADC-phase evidence and update the research decision trail."""
import hashlib,json,shutil,subprocess
from pathlib import Path
import pandas as pd
from audit_ftu_adc_phase import digest
root=Path('experiment_runs/ftu_adc_phase')
gate=json.loads((root/'synthetic_gate.json').read_text());prov=json.loads((root/'provenance/provenance.json').read_text());validation=json.loads((root/'synthetic_validation.json').read_text())
assert not gate['gate_passed'] and prov['numerical_reproduction_passed']
metrics=pd.read_csv(root/'synthetic_metrics.csv');stages=pd.read_csv(root/'stage_phase_metrics.csv')
source_manifest=json.loads((root/'source_manifest.json').read_text())
for name,h in source_manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
prior=subprocess.check_output(['git','show','4154b82:experiment_runs/ftu_adc_phase/protocol.json'])
assert json.loads(prior)==json.loads((root/'protocol.json').read_text())
def table(df):
    def fmt(x):return 'undefined' if pd.isna(x) else f'{x:.7g}' if isinstance(x,float) else str(x)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
report=f'''# FTU ADC→complex RDA phase fidelity and source audit

## Decision

本轮确认了一个会抹掉已知微动相位的导出环节：**帧内 chirp 均值消除**。在受控单目标 ADC 中，帧内恒相位目标完全归零；按条件性 PRI 加入帧内连续微动后，完整 Doppler FFT 的 DC 仍归零，legacy 截断残差则不再忠实保留相位。按结果前冻结的停止规则，**暂停六条真实记录的 HR 2×2 评分、54 条扩展和模型训练**。本轮仍完成一条真实原始记录的来源复现，该操作只验证导出来源，没有评价 HR 或挑选目标。

协议/六条哈希选样/源码快照先提交于 `4154b82`，随后才执行。没有根据结果改门槛、频带、选样、延时或模型。

## Historical-source evidence

复现记录：`{prov['sample_tag']}`，按原六名训练受试者的固定哈希规则选样后取第一人。读取完整原始 bin，用归档候选代码及当前 legacy 默认链独立生成全部 **{prov['frames']} 帧**，并复算全记录 range 能量中心。

- 原始 bin 字节数：{prov['raw_bytes']}；SHA-256：`{prov['raw_sha256']}`。
- 历史 NPZ SHA-256：`{prov['historical_rda_sha256']}`，与上轮 source_hashes 一致。
- 重新估计 range 中心 **{prov['recomputed_global_range_center']}**，历史 sidecar 中为 **{prov['saved_global_range_center']}**。
- 每帧最大相对 L2 误差 **{prov['max_relative_l2']:.8g}**；最大绝对误差 **{prov['max_abs_error']:.8g}**（未标准化 FFT 单位）。逐帧误差见 provenance/reproduction_per_frame.csv。
- 两份 loader 对同一原始数据输出数组一致；归档转换与当前 legacy 转换数值一致。

这些证据将本条实际导出与“归档 loader → 64 个抽样 chirp → chirp_mean → 8 点 Doppler FFT → angle FFT → 全记录 range crop”的数值行为联系起来。**sidecar 没有生成命令、原始输入哈希或代码 commit，仍不能声称找回了唯一的历史生成脚本/命令；本次原始哈希是新计算的。**也没有将单记录复现直接当作其余 53 条的来源证明。

## Actual configuration and coordinates

原 sidecar 的输入尺寸为 `1200×4×128×250`，输出 `1200×8×16×8` complex64。六条固定哈希选样中的五条有保存的配置日志，p03 日志缺失，未用其他记录冒充；原文和最后配置行在 config_evidence/ 与 configuration_evidence.json。

现有日志显示 ChannelConfig 参数 `3,15`、ChirpConfig TX mask `3`、四 RX；不能据此把四接收通道当作八通道 TDM 虚拟阵列。I/Q 的现行解释为 chirp→RX 块→I1 I2 Q1 Q2，归档和现行 loader 一致。本轮 int16 打包/实际 loader 解包往返验证了这个**声明布局的实现**；未取得独立官方读取器或已知硬件注入数据，物理布局正确性仍待确认。

合成使用 20 Hz 帧率、128 chirp、250 ADC 点、4 RX；按 ProfileConfig idle=700、ramp=5000、10ns 单位解释，条件性 PRI 为 57µs。日志/API 单位解释及 p03 的实际设置仍属边界，不能称实测时间戳。相位为 0.2 sin(2π·1.2t)，模拟帧内常量及按上述 PRI 连续变化两种情形，ADC 幅度1000，int16 量化后通过实际 loader 解包。单目标 range12、固定 boresight beam8，无 HR 参与选点。

历史 chirp 索引是 `linspace(0,127,64,dtype=int)`，63 个间隔中 62 个为2、1个为3；不是严格等间隔。故不把相同数组下标当相同非零物理 Doppler：本轮比较严格共有的 **物理 DC**（两者都是 shifted index4），并保存实际 chirp 时间上的频率响应。完整64点后取8点保留的是更窄低频范围，并非全频压缩；8点 FFT 只使用选中序列前8项（原始索引0,2,…,14），但其减去的均值由全部64项决定。

## Synthetic 2×2 results

门槛事先固定：每帧幅度大于无 clutter 理论相干增益的 1e-6、累计相位 RMSE≤0.01 rad、固定频带峰值误差≤3 bpm。相位统一与每帧 chirp0 的已知相位比较，不根据结果补偿平均采样时刻。对极低幅度明确给出 undefined，不从数值残差强行解释相位/HR。

{table(metrics)}

关闭均值消除的四个 case×FFT 条件均保留72 bpm，累计相位 RMSE 为 0.000024–0.006634 rad；连续微动的较大误差包含不同 chirp 积分时刻相对 chirp0 的确定性偏差，不能全算作相位损坏。开启均值消除的四个条件全部未通过：恒相位在 clutter 阶段即为零；连续微动经全64点 FFT 的 DC 归零；legacy 残差累计相位 RMSE **2.22518 rad**，即使其最大频谱峰仍是72 bpm。**读出正确单一频率不能代替相位保真。**

全64项减均值后再求 DC 和为零是代数结果；这里展示它对这一类已知微动信号的影响，不能据此推断所有非零 Doppler/range/多散射体中的 HR 信息也都消失。该受控模型没有真实背景、噪声或胸壁散射，未证明它是现有 FTU 大误差的唯一主因。

![相位与幅度保真](figures/synthetic_phase_fidelity.png)

## Stage localization and checks

保存 ADC 观测、clutter 后观测、range FFT、Doppler DC、angle FFT 五阶段的幅度及相位轨迹（stage_phase_metrics.csv / stage_phase_traces.npz）。独立实现用直接复数求和核对 Doppler DC/boresight，最大相对 L2 差为 {validation['independent_stage_final_max_relative_l2']:.8g}，原始大幅度 FFT 单位最大绝对差 {validation['independent_stage_final_max_abs_error']:.8g}；相对误差来自 float32 求和与 FFT 的计算顺序。现有生产合成检查10项全部通过，计时与输出在 synthetic_validation.json / production_checks.txt。

![非均匀 chirp 时间上的频率响应](figures/doppler_physical_response.png)

本地 uv 主合成运行 **{gate['elapsed_sec']:.3f}秒**，独立阶段复核 **{validation['elapsed_sec']:.3f}秒**；本地没有原始 ADC，单记录来源核对按运行要求放在 Kaggle CPU，耗时 **{prov['elapsed_sec']:.3f}秒**。本轮没有训练或真实 HR 性能增益声明。

## Next gate

本轮不把关闭 chirp_mean 直接定为新的正式前端。先对同一六条冻结记录，在不训练、不用 HR 选点的条件下确定可验证的 I/Q 布局、TX/RX 与帧间相干性证据，并设计保留微动相位的 clutter 处理；随后重新通过端到端合成门控，才能进入同校准区域/物理 Doppler 的六记录真实对照。只有参考对应收益跨记录一致，才扩展54条并重导出、重训固定模型。原 `frame/20+0s` 同步假设仍未变为测得的同步结果。

重跑：先使用冻结 sources 与 protocol（不要重新运行 prepare 改写协议），`uv run --locked --no-sync python audit_ftu_adc_phase.py --code experiment_runs/ftu_adc_phase --out experiment_runs/ftu_adc_phase`；然后运行 summarize_ftu_adc_phase.py。来源任务配置和实际 runner 已归档，输出日志与逐帧结果在 provenance/。
'''
(root/'REPORT.md').write_text(report,encoding='utf-8')
plan=Path('PLAN.md');text=plan.read_text(encoding='utf-8');marker='## 2026-10-18 ADC→RDA 相位保真门控'
if marker not in text:plan.write_text(text+'\n\n'+marker+'\n\n协议4154b82下完成FTU配置合成2×2及单条原始记录全1200帧历史导出复现。chirp_mean消除帧内恒相位目标；连续微动下完整FFT DC归零，legacy残差相位RMSE2.225rad。关闭均值消除保留已知相位，但尚无真实HR收益证据。门控未通过，暂停六记录HR评分/54条扩展/重训，继续I/Q布局、相干性与保相位clutter审计。详见 [报告](experiment_runs/ftu_adc_phase/REPORT.md)。\n',encoding='utf-8')
for script in [Path(__file__),Path('fetch_ftu_adc_provenance.py')]:shutil.copyfile(script,root/'sources'/script.name)
stage=Path('tmp/ftu_adc_phase_upload');shutil.copyfile(stage/'kernel-metadata.json',root/'kernel-metadata.json')
shutil.copyfile(stage/'run.py',root/'sources'/'kaggle_runner.py')
(root/'final_validation.json').write_text(json.dumps(dict(protocol_pre_results_commit='4154b82',protocol_unchanged=True,
    frozen_source_hashes_verified=len(source_manifest),original_rda_hash_verified=True,synthetic_stop_obeyed=True,
    real_hr_ablation=False,training=False),indent=2))
(root/'artifact_manifest.json').write_text(json.dumps({p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'},indent=2))
print(report[:1700])
