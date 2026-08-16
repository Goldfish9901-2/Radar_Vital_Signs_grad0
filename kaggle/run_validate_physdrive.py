"""Kaggle kernel 入口：挂载官方 PhysDrive dataset（xiaoyang274/physdrive）并运行数据验证。

流程：
  1. 解压代码 dataset 到 /kaggle/working
  2. 安装运行依赖（neurokit2、h5py、matplotlib 等）
  3. 定位官方 PhysDrive 挂载路径（兼容新旧路径）
  4. 运行 kaggle/validate_physdrive.py 生成验证报告
  5. 打印 summary.json 关键结论

dataset_sources 需包含：
  - goldfish9901/radar-vital-signs-code（代码）
  - xiaoyang274/physdrive（官方数据）
"""
from __future__ import annotations

import json
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
            # 内部结构可能是 <root>/PhysDrive/mmWave 或 <root>/mmWave
            for probe in (cand / "PhysDrive" / "mmWave", cand / "mmWave"):
                if probe.exists():
                    print(f"PhysDrive root: {cand} (mmWave at {probe})", flush=True)
                    return cand
            return cand
    raise FileNotFoundError("未找到官方 PhysDrive dataset 挂载，请确认 dataset_sources 包含 xiaoyang274/physdrive")


def main() -> None:
    step("1/5 定位代码目录")
    code_root = find_code_root()
    os.chdir(code_root)
    sys.path.insert(0, str(code_root))
    print("code root:", code_root)

    step("2/5 安装依赖")
    run([sys.executable, "-m", "pip", "install", "-q", "neurokit2", "h5py", "matplotlib", "pyyaml"])

    step("3/5 定位官方 PhysDrive 数据")
    physdrive_root = find_physdrive_root()

    step("4/5 运行数据验证")
    out_dir = WORK / "physdrive_validation"
    run(
        [
            sys.executable,
            str(code_root / "kaggle" / "validate_physdrive.py"),
            "--root", str(physdrive_root),
            "--output-dir", str(out_dir),
        ]
    )

    step("5/5 打印验证结论")
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        print(f"\nsummary: {summary_path}", flush=True)
        print(f"examples: {out_dir / 'examples'}", flush=True)
    print("\nValidation finished.")


if __name__ == "__main__":
    main()
