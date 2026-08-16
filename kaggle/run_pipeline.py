"""Radar Vital Signs Kaggle 试运行流水线（script kernel）。

流程：
  1. 解压代码 dataset 到 /kaggle/working
  2. 安装运行依赖（neurokit2 等）
  3. 合成 PhysDrive 兼容小样本数据
  4. 统一导出（export_all_datasets.py --physdrive）
  5. 构建训练窗口（build_training_dataset.py；可选 RDA 前端池：env RDA_CONFIG_JSON）
  6. 训练 TCN 基线（train_model.py）
  7. 评估（evaluate_model.py）

RDA 前端池（src/radar）：在 5 构建窗口前可选插入可替换 RDA 算法池，用于
Phase 2 单行 backbone 筛选。设环境变量 RDA_CONFIG_JSON 为一个 RDAConfig 的
JSON 字符串即可启用；留空则保持当前基线行为。

所有数据获取与处理均在 Kaggle 完成，本地只负责代码准备。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    subprocess.run(cmd, cwd=cwd, check=True, env=full_env)


def step(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def main() -> None:
    step("1/8 定位代码目录")
    # Kaggle 自动解压 dataset 中的 zip，代码位于 <dataset 挂载点>/Radar_Vital_Signs_grad0/
    # 兼容新旧挂载路径：旧版 /kaggle/input/<slug>/，新版 /kaggle/input/datasets/<owner>/<slug>/
    candidates = [
        Path("/kaggle/input/radar-vital-signs-code/Radar_Vital_Signs_grad0"),
        Path("/kaggle/input/datasets/goldfish9901/radar-vital-signs-code/Radar_Vital_Signs_grad0"),
    ]
    code_root = next((c for c in candidates if (c / "src").exists()), None)
    if code_root is None:
        raise FileNotFoundError("未找到代码目录 src/，请确认 dataset goldfish9901/radar-vital-signs-code 已挂载")
    os.chdir(code_root)
    sys.path.insert(0, str(code_root))
    print("code root:", code_root)

    step("2/8 安装依赖")
    run([sys.executable, "-m", "pip", "install", "-q", "neurokit2", "h5py", "pyyaml"])

    step("3/8 合成 PhysDrive 兼容数据")
    synth = code_root / "kaggle" / "synthesize_physdrive.py"
    data_root = WORK / "Dataset"
    run([sys.executable, str(synth), "--root", str(data_root)])

    step("4/8 统一导出")
    exports_dir = WORK / "exports"
    run(
        [
            sys.executable,
            "-m",
            "src.data.loaders.export_all_datasets",
            "--dataset-root", str(data_root),
            "--output-dir", str(exports_dir),
            "--physdrive",
            "--max-samples", "8",
            "--max-samples-per-session", "2",
        ]
    )

    step("5/8 构建训练窗口")
    train_exports_dir = WORK / "training_exports"
    build_cmd = [
        sys.executable,
        "-m",
        "src.data.build_training_dataset",
        "--exports-dir", str(exports_dir),
        "--output-dir", str(train_exports_dir),
        "--target", "heart_rate",
        "--window-size", "256",
        "--stride", "128",
        "--normalize", "window_zscore",
        "--max-windows-per-sample", "10",
    ]
    # Phase 2 RDA front-end hook: set RDA_CONFIG_JSON to a JSON string selecting
    # one method per stage (e.g. '{"clutter":"mti","localization":"hr_band",
    # "range_selection":"hr_band","beamforming":"mvdr"}'). Empty/unset => baseline.
    rda_config = os.environ.get("RDA_CONFIG_JSON")
    if rda_config:
        build_cmd += ["--rda-config", rda_config]
        print(f"[rda] using front-end pool config: {rda_config}", flush=True)
    run(build_cmd)

    step("6/8 训练 TCN 基线")
    model_out = WORK / "model_outputs" / "tcn_smoke"
    # Kaggle 预装 torch 与部分 GPU 架构不匹配（CUDA error: no kernel image），小样本试运行回退 CPU
    cpu_env = {"CUDA_VISIBLE_DEVICES": ""}
    run(
        [
            sys.executable,
            "-m",
            "src.training.train_model",
            "--model", "tcn",
            "--datasets", "PhysDrive",
            "--export-dir", str(train_exports_dir),
            "--output-dir", str(model_out),
            "--epochs", "10",
            "--batch-size", "16",
            "--num-workers", "0",
            "--patience", "5",
        ],
        env=cpu_env,
    )

    step("7/8 评估")
    run(
        [
            sys.executable,
            "-m",
            "src.training.evaluate_model",
            "--model-dir", str(model_out),
            "--export-dir", str(train_exports_dir),
            "--target-datasets", "PhysDrive",
            "--split", "test",
            "--num-workers", "0",
        ],
        env=cpu_env,
    )

    step("8/8 打印产物清单")
    for p in sorted(WORK.rglob("*.json")):
        print(p)
    for p in sorted(WORK.rglob("best.pt")):
        print(p)
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
