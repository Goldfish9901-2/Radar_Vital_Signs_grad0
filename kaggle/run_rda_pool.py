"""Kaggle kernel: RDA front-end algorithm pool — Phase 0 + Phase 1 screening.

Runs the replaceable RDA front-end pool (src.radar) end-to-end on Kaggle WITHOUT
any GPU training:

  - Phase 0 (CPU): physical sanity diagnostics for each candidate RDA config
    (hr_band energy, spectral SNR, phase variance/continuity) via
    scripts/run_rda_diagnostic.py.
  - Phase 1 (CPU): representation validator (experiment_runs/validate_representation.py)
    on the best candidate(s), giving HR correlation + linear-probe R^2 BEFORE any
    GPU work — directly reusing the already-built validator infrastructure.

This is the *cheap gate* from docs/RDA_ALGORITHMS.md: it tells you which RDA
candidates are worth a backbone run, so you do not burn GPU on a 5xN factorial.

dataset_sources must include:
  - goldfish9901/radar-vital-signs-code  (this repo)
  - xiaoyang274/physdrive               (official data, for real PhysDrive cubes)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

WORK = Path("/kaggle/working")

# Default first-round pool (see docs/RDA_ALGORITHMS.md section "First-round pool").
DEFAULT_CONFIGS: List[str] = [
    # baseline: current behaviour (no RDA re-processing)
    json.dumps({"clutter": "none", "localization": "energy", "range_selection": "global_energy", "beamforming": "fft"}),
    # clutter variants
    json.dumps({"clutter": "mti", "localization": "energy", "range_selection": "global_energy", "beamforming": "fft"}),
    json.dumps({"clutter": "pca", "localization": "energy", "range_selection": "global_energy", "beamforming": "fft"}),
    # HR-band localization + HR-band range (the key PhysDrive fix)
    json.dumps({"clutter": "mti", "localization": "hr_band", "range_selection": "hr_band", "beamforming": "fft"}),
    # angular sophistication
    json.dumps({"clutter": "mti", "localization": "hr_band", "range_selection": "hr_band", "beamforming": "mvdr"}),
]

CPU_ENV = {"CUDA_VISIBLE_DEVICES": ""}


def run(cmd: List[str], cwd: Path | None = None, env: dict | None = None) -> None:
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


def resolve_export_root(physdrive_root: Path, work: Path) -> Path:
    """Mimic run_physdrive_ablation.prepare_export_root: locate mmWave and link."""
    layouts = [
        ("official", physdrive_root / "PhysDrive" / "mmWave"),
        ("kaggle-mirror", physdrive_root / "mmWave" / "mmWave"),
        ("flat", physdrive_root / "mmWave"),
    ]
    mmwave = next((p for _, p in layouts if p.is_dir()), None)
    if mmwave is None:
        raise FileNotFoundError(f"未找到 mmWave 目录（候选: {[p for _, p in layouts]}）")
    if mmwave == physdrive_root / "PhysDrive" / "mmWave":
        return physdrive_root
    org = work / "physdrive_org"
    link = org / "PhysDrive"
    if not link.exists():
        org.mkdir(parents=True, exist_ok=True)
        link.symlink_to(mmwave.parent, target_is_directory=True)
    return org


def main() -> None:
    step("1/6 定位代码目录")
    code_root = find_code_root()
    os.chdir(code_root)
    sys.path.insert(0, str(code_root))
    print("code root:", code_root)

    step("2/6 安装依赖")
    run([sys.executable, "-m", "pip", "install", "-q", "neurokit2", "h5py", "matplotlib", "pyyaml"])

    step("3/6 定位官方 PhysDrive 数据并导出 RDA cubes")
    physdrive_root = find_physdrive_root()
    export_root = resolve_export_root(physdrive_root, WORK)
    exports_dir = WORK / "exports"
    if (exports_dir / "PhysDrive" / "manifest.csv").exists():
        print(f"[skip] 已存在 {exports_dir}", flush=True)
    else:
        run([
            sys.executable, "-m", "src.data.loaders.export_all_datasets",
            "--dataset-root", str(export_root),
            "--output-dir", str(exports_dir),
            "--physdrive", "--max-samples", "20", "--max-samples-per-session", "3",
        ], env=CPU_ENV)

    step("4/6 Phase 0 — 物理诊断（CPU，无训练）")
    diag_dir = WORK / "rda_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    cfg_env = os.environ.get("RDA_CONFIGS")
    configs = json.loads(cfg_env) if cfg_env else DEFAULT_CONFIGS
    run([
        sys.executable, str(code_root / "scripts" / "run_rda_diagnostic.py"),
        "--raw-dir", str(exports_dir),
        "--datasets", "PhysDrive",
        "--configs", *configs,
        "--max-windows", "30",
        "--out", str(diag_dir / "rda_diagnostic.csv"),
    ], env=CPU_ENV)
    print(f"[written] {diag_dir / 'rda_diagnostic.csv'}")

    step("5/6 Phase 1 — representation validator（CPU，无训练）")
    # Validate the baseline plus the HR-band / MVDR candidates (the most promising).
    validate_configs = [
        configs[0],   # baseline
        configs[3],   # hr_band loc + range
        configs[4],   # + mvdr
    ]
    for i, cfg in enumerate(validate_configs):
        tag = f"cfg{i}"
        run([
            sys.executable, str(code_root / "experiment_runs" / "validate_representation.py"),
            "--representation", "proposed",
            "--datasets", "PhysDrive",
            "--source", "raw",
            "--raw-dir", str(exports_dir),
            "--rda-config", cfg,
            "--max-windows", "60",
            "--sample-rate", "20.0",
            "--out", str(diag_dir / f"rep_report_{tag}.html"),
        ], env=CPU_ENV)

    step("6/6 汇总")
    for p in sorted(diag_dir.glob("*")):
        print(p)
    print("\nRDA pool Phase 0/1 screening finished. Candidates with high HR-correlation "
          "advance to a single-backbone Phase 2 run (see run_pipeline.py --rda-config).")


if __name__ == "__main__":
    main()
