# FTU 新时间轴与 EDACM-only 实验

完成 PLAN.md 的本轮四项任务；未训练模型。同步仍是显式假设，不能称为精确同步验证。

## 数据与实现

- 正式构建器使用 20 Hz、起始偏移 0 秒；保留参考时间戳，范围外标签为 NaN。
- 440 条记录、3,520 个窗口；窗口 256 帧，步长 128 帧。冻结历史 E2a 受试者划分。
- 每个窗口导出 x_time=(1,256)、x_freq=(1,129)，来自同一 EDACM 信号，无 VMD、无通道复制。
- TCN / Transformer / CycleFormer 的单通道工厂前向检查通过；训练入口按输入推断通道数。
- 固定偏移 0、±1、±2 秒的共同有效集合包含 3,520 个窗口，门槛为 80% 标签覆盖。

## 测试集信号诊断（704 个窗口；不是神经网络 MAE）

| 方法 | MAE (bpm) |
|---|---:|
| train_mean | 4.3315 |
| train_median | 5.9851 |
| fft | 20.6762 |
| stft | 21.5531 |

前三谱峰在参考 HR ±5 bpm 内的命中率：48.15%；循环错位四个窗口后为 47.02%。
差距很小，尚不足以支持可靠动态跟踪；循环错位对近恒定 HR 是弱对照，不能单独证明没有心跳信息。
FFT 使用窗口内选点与 EDACM；STFT 使用连续记录选点与 EDACM，两者并非仅变换算法不同。
每条记录仅八个重叠窗口，相关系数需谨慎解释；未把窗口当成独立受试者来作显著性推断。
全部记录内相关系数中位数：FFT=-0.0231，STFT=-0.0544（仅统计相关系数有定义的记录）。

## 校验与产物

独立使用历史归档的原始 Polar CSV 复算全部标签，最大误差 0.00000305 bpm。
跨环境频域特征复算最大绝对误差 1.4901161e-06；4 个窗口超过 1e-6，全部在 float32 校验容差 1e-5 内。原环境校验使用 1e-6。
Kaggle 计算用时 275.07 秒；本地独立校验 2.97 秒。

- full_artifacts.zip：完整新版窗口、manifest、配置、统计与十名受试者各一条预先固定记录的时频图。
- label_alignment.csv / offset_baselines.csv：固定偏移、共同窗口集合及对应误差。
- signal_baselines.csv / record_metrics.csv / baseline_summary.csv：逐窗口、逐记录、逐受试者结果。
- protocol.json / validation.json / independent_validation.json / source_manifest.json：协议及证据。
- 解压后训练入口使用 training_exports_ftu_edacm_v2；加载器支持迁移后的窗口路径。

## 判断

数据导出与单通道接口已具备后续最小监督学习的工程条件，但本轮频谱诊断没有提供可靠动态跟踪证据。
暂不扩大到 CycleFormer、VMD 或跨域实验；后续若做 TCN，应明确定位为检验可学习性的探索实验。
旧的 proposed / HR-AdaVMD 数值问题仍未修复；新版 EDACM-only 完全绕过该路径。

[Kaggle CPU 实验 v2](https://www.kaggle.com/code/goldfish9901/ftu-timebase-edacm-v2)
