# 实验组织与执行方案

## 目标原则

实验围绕新版主线展开：

1. **主模型**：CycleFormer，验证周期 token 是否比普通时间序列建模更适配雷达生命体征。
2. **代表性 baseline**：只保留 FFT、STFT、TCN、Transformer、PatchTST、TimesNet、Source Only、Pseudo-label Adaptation。
3. **跨域设置**：关注同数据集性能、跨数据集泛化、源无关目标域适配。

不再设计多源训练实验。

## 方法分组

| 组别 | 方法 | 作用 |
|---|---|---|
| 传统信号处理 | FFT | 最经典频域峰值 baseline |
| 传统信号处理 | STFT | 短时频谱峰值/轨迹 baseline |
| 常规深度模型 | TCN | 卷积时序 baseline |
| 常规深度模型 | Transformer | 普通 attention baseline |
| 先进时序模型 | PatchTST | 固定 patch token baseline |
| 先进时序模型 | TimesNet | 周期 2D 变化建模 baseline |
| 域迁移设置 | Source Only | 源域训练、目标域直接测试 |
| 域适应设置 | Pseudo-label Adaptation | 目标域无标签伪标签适配 |
| 主方法 | CycleFormer | 周期 token 驱动 Transformer |

## 实验 1：同数据集内评估

目的：验证模型基本拟合能力。

| ID | 训练域 | 测试域 | 方法 |
|---|---|---|---|
| W01 | FTU | FTU | FFT/STFT/TCN/Transformer/PatchTST/TimesNet/CycleFormer |
| W02 | PhysDrive | PhysDrive | FFT/STFT/TCN/Transformer/PatchTST/TimesNet/CycleFormer |
| W03 | BGT60TR13C | BGT60TR13C | FFT/STFT/TCN/Transformer/PatchTST/TimesNet/CycleFormer |

传统 FFT/STFT 不训练，只在对应 test split 上评估。

## 实验 2：跨数据集泛化

目的：验证不同设备、姿态、距离和预处理分布下的泛化能力。

| ID | 训练域 | 测试域 | 适配 |
|---|---|---|---|
| C01 | FTU | PhysDrive | 无 |
| C02 | FTU | BGT60TR13C | 无 |
| C03 | PhysDrive | FTU | 无 |
| C04 | PhysDrive | BGT60TR13C | 无 |
| C05 | BGT60TR13C | FTU | 无 |
| C06 | BGT60TR13C | PhysDrive | 无 |

该实验中的深度方法均属于 Source Only 设置。重点比较：

```text
TCN / Transformer / PatchTST / TimesNet / CycleFormer
```

FFT/STFT 作为不依赖源域训练的传统方法，可以在目标 test split 上单独报告。

## 实验 3：源无关域适应

目的：验证无目标域标签情况下，CycleFormer 是否能通过伪标签和时间一致性适配目标域。

设置：

```text
source train with labels
target train without labels for adaptation
target test with labels only for evaluation
```

对比：

| 方法 | 说明 |
|---|---|
| Source Only | 源域模型直接测试目标域 |
| Pseudo-label Adaptation | 使用目标域无标签伪标签适配 |
| CycleFormer + Pseudo-label Adaptation | 主模型结合伪标签和时序置信度权重 |

建议优先跑：

```text
FTU -> PhysDrive
FTU -> BGT60TR13C
PhysDrive -> FTU
```

若时间充足，再补完整六个跨域方向。

## 实验 4：CycleFormer 消融

目的：证明周期 token 设计和时频融合设计确实有效。

| 消融项 | 说明 |
|---|---|
| Full CycleFormer | 完整模型 |
| w/o frequency branch | 只用 `x_time` |
| w/o respiration periods | 只保留心跳候选周期 |
| w/o heart periods | 只保留呼吸/长周期候选，验证心跳周期 token 贡献 |
| fixed patch token | 用 PatchTST 替代周期 token |
| generic Transformer token | 用普通 Transformer 替代周期 token |
| w/o pseudo-label adaptation | 跨域时只做 Source Only |

当前代码已支持 `--time-only`、`--heart-periods` 和 `--respiration-periods`，可直接通过训练命令完成周期候选消融。

## 指标

主指标：

- MAE
- RMSE
- Pearson `r`
- within 5 bpm
- within 10 bpm

推荐附加分析：

- 按 dataset 统计。
- 按 participant/session 统计。
- 按 group_key 的连续窗口统计。
- Bland-Altman 图，放在论文分析部分。

## 命令模板

### 1. 构建训练数据

```bash
python src/data/build_training_dataset.py \
  --exports-dir exports \
  --output-dir training_exports \
  --representation proposed \
  --target heart_rate \
  --window-size 256 \
  --stride 128 \
  --normalize window_zscore \
  --overwrite
```

### 2. FFT/STFT baseline

```bash
python src/training/evaluate_signal_baselines.py \
  --method fft \
  --export-dir training_exports \
  --target-datasets PhysDrive \
  --split test \
  --output-json model_outputs/signal_baselines/fft_physdrive_test.json
```

```bash
python src/training/evaluate_signal_baselines.py \
  --method stft \
  --export-dir training_exports \
  --target-datasets PhysDrive \
  --split test \
  --output-json model_outputs/signal_baselines/stft_physdrive_test.json
```

### 3. 训练 CycleFormer

```bash
python src/training/train_model.py \
  --model cycleformer \
  --datasets FTU \
  --export-dir training_exports \
  --output-dir model_outputs/cycleformer_ftu_source \
  --epochs 80 \
  --batch-size 32 \
  --d-model 64 \
  --d-ff 128 \
  --num-layers 2 \
  --nhead 4
```

### 4. 训练深度 baseline

```bash
python src/training/train_model.py \
  --model tcn \
  --datasets FTU \
  --export-dir training_exports \
  --output-dir model_outputs/tcn_ftu_source \
  --epochs 80 \
  --batch-size 32
```

将 `--model` 替换为：

```text
transformer
patchtst
timesnet
```

即可训练对应 baseline。

### 5. Source Only 跨域评估

```bash
python src/training/evaluate_model.py \
  --model-dir model_outputs/cycleformer_ftu_source \
  --export-dir training_exports \
  --target-datasets PhysDrive \
  --split test
```

### 6. Pseudo-label Adaptation

```bash
python src/training/adapt_source_free.py \
  --source-model-dir model_outputs/cycleformer_ftu_source \
  --export-dir training_exports \
  --target-datasets PhysDrive \
  --adapt-split train \
  --eval-split test \
  --output-dir model_outputs/cycleformer_ftu_to_physdrive_pseudo \
  --epochs 20 \
  --batch-size 64 \
  --temporal-window 5
```

## 推荐执行顺序

1. 构建 `training_exports`。
2. 跑 FFT 和 STFT，得到传统 baseline。
3. 训练 FTU 源域的 CycleFormer、TCN、Transformer、PatchTST、TimesNet。
4. 对 PhysDrive 和 BGT60TR13C 做 Source Only 跨域评估。
5. 对 CycleFormer 做 Pseudo-label Adaptation。
6. 补同数据集内三数据集结果。
7. 做 CycleFormer 消融。

## 结果记录规范

实验目录命名：

```text
model_outputs/{method}_{source}_{target_or_sourceonly}_{setting}
```

示例：

```text
model_outputs/cycleformer_ftu_source
model_outputs/cycleformer_ftu_to_physdrive_pseudo
model_outputs/timesnet_ftu_source
model_outputs/signal_baselines/fft_physdrive_test.json
```

每个深度模型实验至少保留：

- `run_config.json`
- `history.json`
- `summary.json`
- `best.pt`
- `eval_*.json`

## 快速调试命令

```bash
python src/data/build_training_dataset.py \
  --exports-dir exports \
  --output-dir tmp/training_exports_debug \
  --max-windows-per-sample 20 \
  --overwrite
```

```bash
python src/training/train_model.py \
  --model cycleformer \
  --datasets FTU \
  --export-dir tmp/training_exports_debug \
  --output-dir tmp/cycleformer_debug \
  --epochs 2 \
  --limit-batches 5
```
