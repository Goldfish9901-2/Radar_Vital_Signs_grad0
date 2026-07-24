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

```bash
python src/training/adapt_source_free.py \
  --source-model-dir model_outputs/cycleformer_ftu_source \
  --export-dir training_exports \
  --target-datasets PhysDrive \
  --output-dir model_outputs/cycleformer_ftu_to_physdrive_pseudo \
  --epochs 20 \
  --batch-size 64 \
  --temporal-window 5
```

## 文档

- `docs/DATASETS.md`：三数据集格式、统一导出字段和注意事项。
- `docs/METHOD_PIPELINE.md`：主方法流程和代码模块对应关系。
- `docs/EXPERIMENTS.md`：实验矩阵、baseline、消融和指标。
