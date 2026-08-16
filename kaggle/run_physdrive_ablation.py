"""Kaggle kernel: PhysDrive 内部跨域消融（路段 A 平坦 → 路段 C 颠簸拥挤）。

协议（用于证明 source-free 跨域链路有效）：
    ① source-only   A 训练 TCN 直接评估 C-test（无任何适配）
    ② +SFDA         A 模型在 C-train 上伪标签自训练（时间平滑 + 置信度加权）后评估 C-test
    ③ oracle        C 训练 TCN 评估 C-test（域内上限）
    ②-no-temp       消融：关闭时间平滑（--temporal-window 1）
    ②-no-conf       消融：关闭置信度加权（--min-weight 1.0）

流程：
  1. 定位代码 dataset + 安装依赖（neurokit2/h5py/pyyaml；torch 预装）
  2. 定位官方 PhysDrive 挂载（兼容新旧路径）
  3. 导出：--session-prefix A -> exports_src；--session-prefix C -> exports_tgt
  4. 切窗：-> training_exports_src / training_exports_tgt（balanced_grouped 按 session 分组）
  5. 训练源域模型（A train，CPU）-> model_outputs/tcn_src_A
  6. ① source-only 评估（A 模型 on C-test）
  7. ② SFDA（C train 伪标签 + 时间平滑 + 置信度加权）-> model_outputs/tcn_sfda_C
  8. ③ oracle 训练（C train）-> model_outputs/tcn_oracle_C
  9. 组件消融：②-no-temp、②-no-conf
  10. 汇总 ablation_matrix.{csv,md,json}（MAE/RMSE/Pearson r）

所有步骤带 skip-if-exists 保护，可重入。Kaggle GPU 上 torch 与部分卡不匹配，
统一 CUDA_VISIBLE_DEVICES="" 回退 CPU。

dataset_sources 需包含：
  - goldfish9901/radar-vital-signs-code（代码）
  - xiaoyang274/physdrive（官方数据）
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

WORK = Path("/kaggle/working")

SOURCE_PREFIX = "A"  # Flat & Unobstructed
TARGET_PREFIX = "C"  # Bumpy & Congested
MODEL = "tcn"
TRAIN_EPOCHS = int(os.environ.get("ABLATION_TRAIN_EPOCHS", "30"))
ADAPT_EPOCHS = int(os.environ.get("ABLATION_ADAPT_EPOCHS", "20"))
BATCH_SIZE = int(os.environ.get("ABLATION_BATCH_SIZE", "64"))
WINDOW_SIZE = int(os.environ.get("ABLATION_WINDOW_SIZE", "256"))
STRIDE = int(os.environ.get("ABLATION_STRIDE", "128"))

CPU_ENV = {"CUDA_VISIBLE_DEVICES": ""}


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    subprocess.run(cmd, cwd=cwd, check=True, env=full_env)


def step(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def find_code_root() -> Path:
    candidates = [
        Path("/kaggle/input/radar-vital-signs-code/Radar_Vital_Signs_grad0"),
        Path("/kaggle/input/datasets/goldfish9901/radar-vital-signs-code/Radar_Vital_Signs_grad0"),
    ]
    code_root = next((c for c in candidates if (c / "src").exists()), None)
    if code_root is None:
        raise FileNotFoundError("未找到代码目录 src/，请确认 dataset goldfish9901/radar-vital-signs-code 已挂载")
    return code_root


def find_physdrive_root() -> Path:
    candidates = [
        Path("/kaggle/input/physdrive"),
        Path("/kaggle/input/datasets/xiaoyang274/physdrive"),
        Path("/kaggle/input/datasets/physdrive/physdrive"),
    ]
    for cand in candidates:
        if cand.exists():
            print(f"PhysDrive dataset root: {cand}", flush=True)
            return cand
    raise FileNotFoundError("未找到官方 PhysDrive dataset 挂载，请确认 dataset_sources 包含 xiaoyang274/physdrive")


def prepare_export_root(physdrive_root: Path, work: Path) -> Path:
    """返回 export_all_datasets 可用的 --dataset-root（期望 <root>/PhysDrive/mmWave/<session>）。

    真实 Kaggle 挂载是 <root>/mmWave/mmWave/<session>，而 PhysDriveDataLoader 固定使用
    <dataset_root>/PhysDrive/mmWave。这里在工作目录建符号链接凑出官方布局：
        <work>/physdrive_org/PhysDrive -> <mmWave 的父目录>
    使 <work>/physdrive_org/PhysDrive/mmWave 指向真实 session 目录。
    """
    layouts = [
        ("official", physdrive_root / "PhysDrive" / "mmWave"),
        ("kaggle-mirror", physdrive_root / "mmWave" / "mmWave"),
        ("flat", physdrive_root / "mmWave"),
    ]
    mmwave = next((p for tag, p in layouts if p.is_dir()), None)
    if mmwave is None:
        raise FileNotFoundError(f"未找到 mmWave 目录（候选: {[p for _, p in layouts]}）")
    if mmwave == physdrive_root / "PhysDrive" / "mmWave":
        print(f"官方布局，直接使用 dataset root: {physdrive_root}", flush=True)
        return physdrive_root
    org = work / "physdrive_org"
    link = org / "PhysDrive"
    if not link.exists():
        org.mkdir(parents=True, exist_ok=True)
        link.symlink_to(mmwave.parent, target_is_directory=True)
    print(f"mmWave at {mmwave} -> 符号链接 {link}（父目录 {mmwave.parent}）", flush=True)
    return org


def eval_checkpoint_on_export(
    checkpoint_dir: Path,
    export_dir: Path,
    split: str,
    out_path: Path,
) -> Dict[str, Any]:
    """用某 checkpoint（含其 label_stats）评估某 export 的指定 split。"""
    import torch
    from torch.utils.data import DataLoader

    from src.models.factory import create_model
    from src.training.common.checkpoints import load_run_config, resolve_checkpoint
    from src.training.common.metrics import aggregate, append_prediction_rows
    from src.training.datasets import LabelStats, RadarWindowDataset

    run_config = load_run_config(checkpoint_dir)
    ckpt_path = resolve_checkpoint(checkpoint_dir, None)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = run_config.get("model")
    if not model_name:
        raise ValueError(f"checkpoint {ckpt_path} 缺少 run_config.json 的 model 字段")
    label_stats = LabelStats(**ckpt["label_stats"])
    model = create_model(model_name, ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    dataset = RadarWindowDataset(export_dir, split, label_stats=label_stats, datasets={"PhysDrive"})
    loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=0)
    rows: list[Dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            x_time = batch["x_time"].to(device, non_blocking=True)
            x_freq = batch["x_freq"].to(device, non_blocking=True)
            pred = model(x_time, x_freq)
            append_prediction_rows(rows, batch, pred, label_stats)

    res: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "export_dir": str(export_dir),
        "split": split,
        "n_windows": len(dataset),
        "overall": aggregate(rows)["overall"],
        "by_group": aggregate(rows, "group_key"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res["overall"], indent=2), flush=True)
    return res


def overall_from_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    """兼容两种指标结构：
    - aggregate() 风格: {"overall": {...}}
    - train_model run_epoch 风格: {"mae_bpm":..,"rmse_bpm":..,"detail_rows":[...]}
    """
    if "overall" in m:
        return dict(m["overall"])
    detail = m.get("detail_rows")
    if detail:
        from src.training.common.metrics import aggregate
        return dict(aggregate(detail)["overall"])
    return {}


def build_matrix(evals_dir: Path, model_outputs: Path) -> Dict[str, Any]:
    """从各步骤产物汇总消融矩阵。"""
    import math

    def _pull(path: Path, ov: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "n": ov.get("count"),
            "mae_bpm": round(float(ov["mae_bpm"]), 3) if ov.get("mae_bpm") is not None and not math.isnan(float(ov["mae_bpm"])) else None,
            "rmse_bpm": round(float(ov["rmse_bpm"]), 3) if ov.get("rmse_bpm") is not None and not math.isnan(float(ov["rmse_bpm"])) else None,
            "pearson_r": round(float(ov["pearson_r"]), 4) if ov.get("pearson_r") is not None and not math.isnan(float(ov["pearson_r"])) else None,
            "source_path": str(path),
        }

    rows: list[Dict[str, Any]] = []

    # ① source-only
    so = evals_dir / "eval_cross_source_only.json"
    if so.exists():
        ov = json.loads(so.read_text(encoding="utf-8"))["overall"]
        rows.append({"setting": "1_source_only", "name": "① source-only (A→C, no adapt)", **_pull(so, ov)})

    # ② SFDA：adapt_source_free.evaluate() 返回的 before/after_adaptation
    # 本身就是 aggregate()["overall"] 结构（含 mae_bpm/rmse_bpm/pearson_r）
    sfda_summary = model_outputs / "tcn_sfda_C" / "summary.json"
    if sfda_summary.exists():
        s = json.loads(sfda_summary.read_text(encoding="utf-8"))
        before = s.get("before_adaptation") or {}
        after = s.get("after_adaptation") or {}
        rows.append({"setting": "2_sfda_before", "name": "② (SFDA before)", **_pull(sfda_summary, before)})
        rows.append({"setting": "2_sfda_after", "name": "② +SFDA full", **_pull(sfda_summary, after)})

    # ③ oracle
    oracle_eval = model_outputs / "tcn_oracle_C" / "eval_test.json"
    if oracle_eval.exists():
        m = json.loads(oracle_eval.read_text(encoding="utf-8"))
        rows.append({"setting": "3_oracle", "name": "③ in-domain oracle (C)", **_pull(oracle_eval, overall_from_metrics(m))})

    # 组件消融
    for tag, name in (
        ("tcn_sfda_C_notemp", "② -temporal-smoothing"),
        ("tcn_sfda_C_noconf", "② -confidence-weighting"),
    ):
        summary = model_outputs / tag / "summary.json"
        if summary.exists():
            s = json.loads(summary.read_text(encoding="utf-8"))
            after = s.get("after_adaptation") or {}
            rows.append({"setting": tag, "name": name, **_pull(summary, after)})

    out_dir = evals_dir
    csv_path = out_dir / "ablation_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        cols = ["setting", "name", "n", "mae_bpm", "rmse_bpm", "pearson_r", "source_path"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    md = [
        "# PhysDrive 跨域消融（A 平坦 → C 颠簸拥挤）\n",
        "",
        "MAE/RMSE in BPM；Pearson r 越高越好。② 应显著优于 ①（MAE 下降 ≥15% 为强证据），且 ② 接近 ③。\n",
        "",
        "| setting | n | MAE | RMSE | Pearson r |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['name']} | {r.get('n','')} | {r.get('mae_bpm','')} | "
            f"{r.get('rmse_bpm','')} | {r.get('pearson_r','')} |"
        )
    (out_dir / "ablation_matrix.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "ablation_matrix.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {csv_path} / ablation_matrix.md / ablation_matrix.json")
    return {"rows": rows}


def main() -> None:
    step("1/10 定位代码目录")
    code_root = find_code_root()
    os.chdir(code_root)
    sys.path.insert(0, str(code_root))
    print("code root:", code_root)

    step("2/10 安装依赖")
    run([sys.executable, "-m", "pip", "install", "-q", "neurokit2", "h5py", "pyyaml"])

    step("3/10 定位官方 PhysDrive 数据")
    physdrive_root = find_physdrive_root()
    export_root = prepare_export_root(physdrive_root, WORK)

    exports_src = WORK / "exports_src"
    exports_tgt = WORK / "exports_tgt"
    train_exports_src = WORK / "training_exports_src"
    train_exports_tgt = WORK / "training_exports_tgt"
    model_outputs = WORK / "model_outputs"
    evals_dir = WORK / "evals"

    step(f"4/10 导出源域（--session-prefix {SOURCE_PREFIX}）与目标域（--session-prefix {TARGET_PREFIX}）")
    if (exports_src / "PhysDrive" / "manifest.csv").exists():
        print(f"[skip] 已存在 {exports_src}", flush=True)
    else:
        run([
            sys.executable, "-m", "src.data.loaders.export_all_datasets",
            "--dataset-root", str(export_root),
            "--output-dir", str(exports_src),
            "--physdrive", "--session-prefix", SOURCE_PREFIX,
        ])
    if (exports_tgt / "PhysDrive" / "manifest.csv").exists():
        print(f"[skip] 已存在 {exports_tgt}", flush=True)
    else:
        run([
            sys.executable, "-m", "src.data.loaders.export_all_datasets",
            "--dataset-root", str(export_root),
            "--output-dir", str(exports_tgt),
            "--physdrive", "--session-prefix", TARGET_PREFIX,
        ])

    step("5/10 分别构建训练窗口")
    if (train_exports_src / "manifest.csv").exists():
        print(f"[skip] 已存在 {train_exports_src}", flush=True)
    else:
        run([
            sys.executable, "-m", "src.data.build_training_dataset",
            "--exports-dir", str(exports_src),
            "--output-dir", str(train_exports_src),
            "--datasets", "PhysDrive",
            "--target", "heart_rate",
            "--window-size", str(WINDOW_SIZE),
            "--stride", str(STRIDE),
            "--normalize", "window_zscore",
        ])
    if (train_exports_tgt / "manifest.csv").exists():
        print(f"[skip] 已存在 {train_exports_tgt}", flush=True)
    else:
        run([
            sys.executable, "-m", "src.data.build_training_dataset",
            "--exports-dir", str(exports_tgt),
            "--output-dir", str(train_exports_tgt),
            "--datasets", "PhysDrive",
            "--target", "heart_rate",
            "--window-size", str(WINDOW_SIZE),
            "--stride", str(STRIDE),
            "--normalize", "window_zscore",
        ])

    # 打印两个域的窗口划分摘要
    for label, d in (("SRC(A)", train_exports_src), ("TGT(C)", train_exports_tgt)):
        summary_path = d / "summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            print(f"\n[{label}] windows={s.get('windows_written')} by_split={s.get('by_split')}", flush=True)
            print(f"[{label}] groups={json.dumps(s.get('groups_by_dataset_split'), ensure_ascii=False)}", flush=True)

    step("6/10 训练源域模型（A train, CPU）")
    src_model_dir = model_outputs / "tcn_src_A"
    if (src_model_dir / "best.pt").exists():
        print(f"[skip] 已存在 {src_model_dir / 'best.pt'}", flush=True)
    else:
        run([
            sys.executable, "-m", "src.training.train_model",
            "--model", MODEL,
            "--datasets", "PhysDrive",
            "--export-dir", str(train_exports_src),
            "--output-dir", str(src_model_dir),
            "--epochs", str(TRAIN_EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--num-workers", "0",
            "--patience", "10",
        ], env=CPU_ENV)

    step("7/10 ① source-only：A 模型直接评估 C-test")
    eval_checkpoint_on_export(
        checkpoint_dir=src_model_dir,
        export_dir=train_exports_tgt,
        split="test",
        out_path=evals_dir / "eval_cross_source_only.json",
    )

    step("8/10 ② SFDA：C-train 伪标签 + 时间平滑 + 置信度加权")
    sfda_model_dir = model_outputs / "tcn_sfda_C"
    if (sfda_model_dir / "best.pt").exists():
        print(f"[skip] 已存在 {sfda_model_dir / 'best.pt'}", flush=True)
    else:
        run([
            sys.executable, "-m", "src.training.adapt_source_free",
            "--source-model-dir", str(src_model_dir),
            "--model", MODEL,
            "--export-dir", str(train_exports_tgt),
            "--target-datasets", "PhysDrive",
            "--adapt-split", "train",
            "--eval-split", "test",
            "--output-dir", str(sfda_model_dir),
            "--epochs", str(ADAPT_EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--num-workers", "0",
        ], env=CPU_ENV)

    step("9/10 ③ oracle：C-train 训练 + 组件消融")
    oracle_model_dir = model_outputs / "tcn_oracle_C"
    if (oracle_model_dir / "best.pt").exists():
        print(f"[skip] 已存在 {oracle_model_dir / 'best.pt'}", flush=True)
    else:
        run([
            sys.executable, "-m", "src.training.train_model",
            "--model", MODEL,
            "--datasets", "PhysDrive",
            "--export-dir", str(train_exports_tgt),
            "--output-dir", str(oracle_model_dir),
            "--epochs", str(TRAIN_EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--num-workers", "0",
            "--patience", "10",
        ], env=CPU_ENV)

    for tag, extra in (
        ("tcn_sfda_C_notemp", ["--temporal-window", "1"]),
        ("tcn_sfda_C_noconf", ["--min-weight", "1.0"]),
    ):
        d = model_outputs / tag
        if (d / "best.pt").exists():
            print(f"[skip] 已存在 {d / 'best.pt'}", flush=True)
            continue
        run([
            sys.executable, "-m", "src.training.adapt_source_free",
            "--source-model-dir", str(src_model_dir),
            "--model", MODEL,
            "--export-dir", str(train_exports_tgt),
            "--target-datasets", "PhysDrive",
            "--adapt-split", "train",
            "--eval-split", "test",
            "--output-dir", str(d),
            "--epochs", str(ADAPT_EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--num-workers", "0",
            *extra,
        ], env=CPU_ENV)

    step("10/10 汇总消融矩阵")
    build_matrix(evals_dir, model_outputs)
    print("\nAblation finished.")


if __name__ == "__main__":
    main()
