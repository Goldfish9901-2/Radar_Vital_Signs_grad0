# Repository Guidelines

## 项目结构与模块组织

本仓库用于毫米波雷达生命体征的数据准备、模型训练、评估和源无关域适应。核心代码位于 `src/`：

- `src/data/loaders/`：FTU、PhysDrive、BGT60TR13C 等数据集加载器。
- `src/data/build_training_dataset.py`：将统一导出的样本切分为训练窗口。
- `src/models/`：TCN、Transformer、HeartTimeMixer 等模型定义。
- `src/training/`：训练、评估、源无关适配入口和数据集封装。

生成物按用途分离：原始或本地数据放在 `Dataset/`，统一导出放在 `exports/`，窗口化训练样本放在 `training_exports/`，模型权重和评估结果放在 `model_outputs/`，项目文档放在 `docs/`。

## 构建、测试与开发命令

- `conda env create -f environment.yml`：创建 Python 3.12 开发环境。
- `conda activate radarnet`：激活 Conda 环境。
- `docker build -t radar-vital-signs .`：使用本地 wheels 和 CUDA 运行时构建镜像。
- `python src/data/build_training_dataset.py --exports-dir exports --output-dir training_exports --target heart_rate --window-size 256 --stride 128 --normalize window_zscore`：生成窗口化训练数据。
- `python src/training/train_model.py --model tcn --datasets FTU --export-dir training_exports --output-dir model_outputs/tcn_ftu_test --epochs 80 --batch-size 32`：训练 TCN 基线模型。
- `python src/training/evaluate_model.py --model-dir model_outputs/tcn_ftu_test --export-dir training_exports --target-datasets PhysDrive --split test`：执行跨数据集评估。

## 代码风格与命名约定

Python 代码使用 4 空格缩进。变量和字段命名应保持领域含义清晰，例如 `participant_id`、`session_id`、`heart_rate`、`respiration_rate`。模块文件使用小写加下划线，参考 `build_training_dataset.py` 和 `adapt_source_free.py`。优先拆分小函数，避免过长流程式代码。环境中包含 `ruff`，提交风格相关修改前建议运行 `ruff check src`。

## 测试指南

当前仓库没有正式测试套件。修改应使用最小可运行流程验证：加载器修改需导出或检查少量样本；训练数据构建修改可使用 `--max-windows-per-sample 20`；模型修改可先用 `--limit-batches N` 快速检查。评估模型时记录 MAE、RMSE、Pearson `r` 和生成的 JSON 路径。

## 提交与 Pull Request 规范

近期提交信息较短直接，例如 `split and train`、`test models`、`domain adaption`。每个提交应聚焦一个逻辑变更，建议使用祈使句摘要，例如 `add PhysDrive loader validation`。PR 应说明影响的数据集或模型路径，列出实际运行的命令，总结关键指标或生成物，并注明未纳入版本控制的大文件。

## 安全与配置提示

不要提交原始数据集、模型权重或生成的训练窗口，除非任务明确要求。避免在源码中写入本机绝对路径；使用 `--exports-dir`、`--output-dir`、`--model-dir` 等命令行参数传入路径。
