"""Clutter x Range 交叉分析: 8-cell 矩阵 + interaction effect。

数据来源: 单轮 Kaggle kernel (clutter-x-range-cross-ablation), 4 clutter x 2 selector。
重叠 cell 与两轮独立 kernel 数值一致, 证明 8 cell 同轮可比。

交互项 (2x2 factorial, baseline = (current, global_center)):
  Δ(C, R) = MAE_{C,R} - MAE_{C,gc} - MAE_{current,R} + MAE_{current,gc}
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent / "cross"
CLUTTER = ("current", "mti", "highpass", "pca")
SELS = ("peak", "global_center")
DS = ("FTU", "BGT60TR13C", "PhysDrive")

tables = {}
for ds in DS:
    with open(BASE / f"summary_{ds}.json") as fh:
        s = json.load(fh)
    mae = {c: {sel: s[c][sel]["mae_bpm"] for sel in SELS} for c in CLUTTER}
    base = mae["current"]["global_center"]
    print(f"\n===== {ds} (n={s['n']}, fs={s['fs']}Hz) =====")
    hdr = "clutter".ljust(10) + "".join(f"{sel:>16}" for sel in SELS)
    print(hdr)
    for c in CLUTTER:
        row = "".join(f"{mae[c][sel]:>16.3f}" for sel in SELS)
        print(c.ljust(10) + row)
    print("\ninteraction Δ (vs current/global_center baseline):")
    for c in CLUTTER:
        if c == "current":
            continue
        for sel in SELS:
            if sel == "global_center":
                continue
            d = mae[c][sel] - mae[c]["global_center"] - mae["current"][sel] + base
            print(f"  Δ({c}, {sel}) = {d:+.3f} BPM")
    tables[ds] = mae

# 汇总: 各数据集最优 cell
print("\n===== 每数据集最优 cell =====")
for ds in DS:
    mae = tables[ds]
    best = min(
        ((c, sel, mae[c][sel]) for c in CLUTTER for sel in SELS),
        key=lambda t: t[2],
    )
    print(f"  {ds}: ({best[0]}, {best[1]}) = {best[2]:.3f} BPM")
