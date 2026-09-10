# Radar Vital Signs

本项目面向毫米波雷达生命体征估计，主线任务是跨数据集心率回归。当前重点方法为 CycleFormer 周期 token 建模：

```text
RDA 统一特征 -> EDACM 目标相位表征 -> HR-AdaVMD 心率自适应分解
-> 时频特征 -> CycleFormer 心率回归 -> Pseudo-label Adaptation 跨域适配
```

## 目录结构

```text
src/data/loaders/        原始数据集加载器
src/data/build_training_dataset.py  训练窗口构建入口
src/features/            主流程特征模块与基线表征
src/models/              CycleFormer、TCN、Transformer、PatchTST、TimesNet 和模型工厂
src/training/            训练、评估、源无关域适配入口
src/training/common/     checkpoint、指标等公共训练工具
docs/                    数据集、方法和实验说明
```

本地生成物不进入版本控制：`Dataset/`、`exports/`、`training_exports/`、`model_outputs/`、`tmp/`。`docker_assets/` 当前保留用于本地 Docker 构建。

## 主流程方法

主方法拆为独立模块，便于查看和修改：

- `src/features/edacm.py`：从 complex RDA 窗口中选择稳定目标 bin，提取并融合 EDACM 相位。
- `src/features/hr_adavmd.py`：对融合相位执行 HR-AdaVMD，包含生理频带初始化、心率模态评分和自适应加权。
- `src/features/representations.py`：组装模型输入，默认表征为 `proposed`，同时保留其他基线表征。
- `src/models/cycleformer.py`：周期 token 驱动的 Transformer 主模型。
- `src/training/evaluate_signal_baselines.py`：FFT/STFT 传统信号处理 baseline。
- `src/training/adapt_source_free.py`：使用目标域无标签数据执行 pseudo-label adaptation。

## 对比实验方法

当前对比方法固定为：

- FFT
- STFT
- TCN
- Transformer
- PatchTST
- TimesNet
- Source Only
- Pseudo-label Adaptation

HeartTimeMixer 和旧表征代码保留以兼容历史实验，但新版方法和实验文档以 CycleFormer 为主线。

## 常用命令

创建环境：

```bash
conda env create -f environment.yml
conda activate radarnet
```

统一导出数据：

```bash
python src/data/loaders/export_all_datasets.py
```

构建训练窗口：

```bash
python src/data/build_training_dataset.py \
  --exports-dir exports \
  --output-dir training_exports \
  --target heart_rate \
  --window-size 256 \
  --stride 128 \
  --normalize window_zscore
```

训练源域模型：

```bash
python src/training/train_model.py \
  --model cycleformer \
  --datasets FTU \
  --export-dir training_exports \
  --output-dir model_outputs/cycleformer_ftu_source \
  --epochs 80 \
  --batch-size 32
```

直接跨域评估：

```bash
python src/training/evaluate_model.py \
  --model-dir model_outputs/cycleformer_ftu_source \
  --export-dir training_exports \
  --target-datasets PhysDrive \
  --split test
```

源无关域适配：

### 包A：数据统一与质量分析

要做的实验：

1. 三数据集读取一致性检查（shape、dtype、时间轴、标签字段）。
2. 域差异量化（特征分布图、MMD/KL、信号强度统计）。
3. 数据可用性基线（缺失率、异常样本率、可训练样本数）。

交付：`reports/A_data_audit.md`、统一数据字典、可复现分析脚本。

### 包B：基线模型建立

要做的实验：

1. 传统信号处理基线（频谱峰值法/滤波链路）。
2. 深度学习基线（1D-CNN、LSTM、CNN-LSTM）。
3. 源域内性能上界与直接迁移下界。

交付：`reports/B_baselines.md`、基线训练配置、基线权重。

### 包C：跨域方法对比

要做的实验：

1. 经典域适应基线：DANN、MMD、CORAL（作为主流对照组）。
2. Source-Free 路线：Source-Free plain、Source-Free + WPL。
3. 本项目升级版：Source-Free + WPL + 时序修正（回归任务替代Reverse-kNN）。

交付：`reports/C_domain_adaptation.md`、方法对比表、最优组合配置。

### 包D：鲁棒性与稳定性验证

要做的实验：

1. 极值心率区间测试（低心率/高心率）。
2. 干扰场景测试（运动、说话、遮挡、多人）。
3. 长时漂移测试（分段统计、漂移曲线、恢复能力）。

交付：`reports/D_robustness.md`、失败样本库、误差归因报告。

### 包E：消融与论文材料

要做的实验：

1. 消融：移除增强、移除域损失、替换骨干网络。
2. 统计显著性检验（配对检验/置信区间）。
3. 论文图表与核心结论固化。

交付：`reports/E_ablation_and_paper.md`、论文图表源文件。

## 4. 实验矩阵（可实施）

说明：每行是一个可直接执行的实验单元，按 `ExpID` 建配置文件和结果目录。  
本版已合并重复项，突出“经典DA基线 -> Source-Free升级 -> 鲁棒性/消融”主线。

| ExpID | 训练域 | 测试域 | 方法 | 主要输入 | 主要输出 | 核心指标 | 通过标准 | 实验目的（备注） |
|---|---|---|---|---|---|---|---|---|
| E01 | 4TU | 4TU | 传统信号基线 | 原始ADC + 参考心率 | HR/BR估计 | MAE, RMSE | MAE <= 6 bpm | 建立信号处理上界与可解释参考 |
| E02 | 4TU | 4TU | 深度基线(CNN-LSTM) | 统一特征张量 | 预测序列 | MAE, r | 不劣于E01 | 建立可迁移的源域监督模型 |
| E03 | 4TU | PhysDrive + BGT60 | 直接迁移(无适配) | E02模型 | 跨域预测 | MAE, 降幅率 | 记录下界 | 统一作为跨域下界（合并原E03/E04） |
| E04 | 4TU | PhysDrive + BGT60 | 经典DA基线组 | DANN / MMD / CORAL | 跨域预测 | MAE, r | 至少1种优于E03 | 合并原E05/E06/E07，保留主流对照 |
| E05 | 4TU(源) | PhysDrive + BGT60(目标) | Source-Free plain | 源模型 + 目标无标签 | 跨域预测 | MAE, r | 接近E04最优 | 验证“无源数据适配”可行性 |
| E06 | 4TU(源) | PhysDrive + BGT60(目标) | Source-Free + WPL | E05 + 置信度加权伪标签 | 跨域预测 | MAE, r | 优于E05 | 验证WPL抑制伪标签噪声有效 |
| E07 | 4TU(源) | PhysDrive + BGT60(目标) | Source-Free + WPL + 时序修正 | E06 + Temporal Correction | 跨域预测 | MAE, r, 漂移量 | 优于E06或更稳 | 对应WPL-SFUDA思想的本项目版本 |
| E08 | 最优模型 | 反向迁移(BGT60->4TU) | 方向一致性验证 | 最优配置 | 跨域预测 | MAE, r | 不出现明显反向失效 | 排除单向迁移偶然性（原E09简化） |
| E09 | 最优模型 | 极端/干扰/长时子集 | 鲁棒性与稳定性 | 场景子集 + 长时序列 | 分场景误差 + 漂移曲线 | 场景波动率, 漂移斜率 | 波动率<=25%，漂移可控 | 合并原E10/E11，统一鲁棒性评估 |
| E10 | 最优模型 | 跨域测试集 | 消融与统计检验 | 去除WPL/时序修正/前端模块 | Delta MAE, CI | 关键模块贡献显著 | 合并原E12并补统计显著性 |

## 5. 评估指标（精简且可落地）

### 5.1 主指标

- MAE（bpm）
- RMSE（bpm）
- Pearson r

### 5.2 跨域指标

- 跨域性能下降率：`(MAE_target - MAE_source) / MAE_source`
- 相对改进率：`(MAE_baseline - MAE_method) / MAE_baseline`

### 5.3 鲁棒性指标

- 场景波动率：不同干扰场景 MAE 的变异系数。
- 长时漂移：按时间窗口统计 MAE 斜率。

### 5.4 工程指标

- 单样本推理时延（ms）
- 显存占用（MB）
- 可复现实验数（成功复现实验/总实验）

建议项目目标：最优跨域方法相对“直接迁移”基线，MAE 改进 >= 30%。

## 6. 方法新颖性与参考文献

### 6.1 经典基线（必须保留）

- `DANN / MMD / CORAL` 仍是跨域任务中的主流强基线，适合本项目作为第一阶段可复现对照。
- 新颖性有限：更适合做“可靠起点”，不建议作为论文唯一创新点。

经典论文：

- DANN (ICML 2015 / JMLR 2016): https://proceedings.mlr.press/v37/ganin15 / https://jmlr.org/papers/v17/15-239.html
- DAN (MMD, ICML 2015): https://proceedings.mlr.press/v37/long15
- Deep CORAL (ECCV-W 2016): https://arxiv.org/abs/1607.01719

### 6.2 建议重点重写的升级方向（面向本项目）

- `Source-Free UDA`（更贴近真实部署）：
  - Radar HAR 代表案例：WPL-SFUDA (Pattern Recognition, 2026): https://www.sciencedirect.com/science/article/pii/S0031320325005266
  - 价值：目标域无标签，且不需要访问源域原始数据，符合隐私约束和工程落地需求。
- `对比学习 + 域对齐`（通常比纯 DANN 更稳）：
  - mmWave gait 代表案例：GaitSADA (2023): https://arxiv.org/abs/2301.13384
  - 价值：先学习更有判别性的表征，再进行分布对齐，在低标注场景更容易获得稳定提升。
- `领域相关迁移学习`（跨模态/跨设备）：
  - 与生理监测更接近：PSG -> FMCW radar 迁移 (2026): https://www.mdpi.com/2306-5354/13/3/283
  - 领域直系早期工作：IR-UWB + ECG 的 SADA (2018): https://www.sciencedirect.com/science/article/pii/S1746809418301927
  - 价值：能直接回答“跨设备、跨传感模态”下生命体征标签如何迁移的问题。
- `鲁棒前端（物理先验）+ 轻量适配`：
  - Pi-ViMo (mmWave vital signs, 2023): https://arxiv.org/abs/2303.13816
  - 价值：先做生理信号提纯，再做域适配，通常比端到端硬对齐更稳，更适合噪声和场景扰动明显的数据。

### 6.4 与本项目的适配建议

- 当前规划中的 `DANN/MMD/CORAL` 保留为基线层。
- 若要提升论文创新性，建议升级到：`基线DA + Source-Free/TTA + 雷达物理先验前端`。


## 7. 文档边界

- 保留：`docs/DATASETS.md`
- 规划、实验矩阵、指标定义统一维护在本文件

## 8. 文档同步约定

- 凡是涉及加载器输出字段、雷达轴语义、采样率/时间轴、导出格式的代码变更，需同步更新：
  - `docs/DATASETS.md`（数据事实与接口语义）
  - `README.md`（方案级摘要与影响范围）
- 提交前至少检查一次“代码实现 vs 文档描述”一致性，避免后续实验配置偏差。


## 9. 容器启动

使用 `compose.yml` 启动（参见 `AGENTS.md` 获取本机环境说明）：

```bash
podman compose up -d
podman exec -it radar_dev bash
```

- 镜像地址：`crpi-ojnb84j7hma95ay2.cn-shanghai.personal.cr.aliyuncs.com/hsun97282/radar:latest`

## 文档

- `docs/DATASETS.md`：三数据集格式、统一导出字段和注意事项。
- `docs/METHOD_PIPELINE.md`：主方法流程和代码模块对应关系。
- `docs/EXPERIMENTS.md`：实验矩阵、baseline、消融和指标。
