# Repository Guidelines

## 数据获取与处理（Kaggle）

**所有数据获取与处理均在 Kaggle 上完成，本地只负责代码准备。** 具体包括：

- 原始数据集（FTU、BGT60TR13C、PhysDrive）托管在 Kaggle Dataset 上，由 Kaggle Notebook 负责下载；
- 数据统一导出（`src/data/loaders/export_all_datasets.py`）在 Kaggle Notebook 中执行；
- 训练窗口构建（`src/data/build_training_dataset.py`）、模型训练（`train_model.py`）与评估（`evaluate_model.py`）均在 Kaggle Notebook 中运行；
- 本地 Git 仓库只存放代码、配置与文档，不提交原始数据、导出结果或模型权重。

运行方式：将本仓库代码打包（或直接用 Notebook 内 `git clone`）后推送到 Kaggle，通过 Notebook 的 Dataset 依赖挂载原始数据，并按顺序执行导出 → 切窗 → 训练 → 评估。

## Kaggle 踩坑记录

以下为在 Kaggle 上试运行本流水线（合成数据 → 导出 → 切窗 → 训练 → 评估）时实际遇到的问题与解决方案：

- **网络：kaggle 命令需走代理。** 本机直接执行 `kaggle` 命令会卡住或失败，须通过 `proxychains4 -q kaggle ...` 执行（代理配置见 `/etc/proxychains4.conf`，默认 `http 127.0.0.1 7890`）。
- **dataset 上传 zip 会被自动解压。** `kaggle datasets create` 上传 `.zip` 后，Kaggle 会解压文件树，`/kaggle/input/<slug>/` 下不再是 zip，而是原目录结构。脚本应直接读取解压后的目录，不要假设 zip 仍存在。
- **dataset 挂载路径分新旧两版。** 旧版挂载在 `/kaggle/input/<slug>/`，新版挂载在 `/kaggle/input/datasets/<owner>/<slug>/`。定位代码目录时应同时检查这两个标准候选路径（见 `kaggle/run_pipeline.py`），不要探测或硬编码其中一个。
- **kernel 元数据 `id` 必须与实际 slug 一致。** `kernel-metadata.json` 的 `id`（如 `goldfish9901/radar-vital-signs-kaggle-pipeline`）若与首次 push 由 `title` 生成的 slug 不一致，会返回 `409 Client Error: Conflict`。首次 push 后再改元数据时，用 `kaggle kernels pull` 查看实际 slug 再对齐。
- **Kaggle 预装 torch 与部分 GPU 架构不匹配。** 直接训练报 `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`。小样本试运行可回退 CPU：训练/评估子进程设置 `CUDA_VISIBLE_DEVICES=""`（见 `run_pipeline.py`），速度足够；真实大规模训练需按 Kaggle GPU 型号选择兼容的 torch 版本。
- **dataset 与 kernel 存在时序问题。** 刚 push 的 dataset 可能未及时挂载到新 kernel，表现为 `/kaggle/input/<slug>` 不存在。等待 dataset 状态变为可用后再 push kernel；失败后重跑前先确认 dataset 已创建成功（`kaggle datasets list --mine`）。
- **`kaggle kernels delete` / `datasets delete` 需交互确认。** 非交互环境（CI/脚本）需加 `-y`，否则因 `input()` 读不到终端而报 `EOFError`。

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
