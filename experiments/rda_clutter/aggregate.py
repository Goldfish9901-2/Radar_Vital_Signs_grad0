"""汇总 rda_clutter 结果 -> 单个 markdown 表格 + JSON。

用法: .venv/bin/python experiments/rda_clutter/aggregate.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "rda_clutter"

DATASET_LABELS = {"FTU": "FTU", "BGT60TR13C": "BGT60", "PhysDrive": "PhysDrive"}
METHOD_ORDER = ["current", "mti", "highpass", "pca"]


def load(ds: str):
    p = OUT / ds / f"summary_{ds}.json"
    return json.load(open(p)) if p.exists() else None


def main():
    data = {ds: load(ds) for ds in ["FTU", "BGT60TR13C", "PhysDrive"]}
    data = {k: v for k, v in data.items() if v is not None}

    # 主表
    rows = []
    for m in METHOD_ORDER:
        cells = []
        for ds, d in data.items():
            if m not in d:
                cells.extend(["--", "--", "--"])
                continue
            v = d[m]
            w = v.get("mae_whole_bpm")
            s = v.get("mae_seg_bpm")
            cells.append(f"{w:g}" if w is not None else "--")
            cells.append(f"{s:g}" if s is not None else "--")
            cells.append(f"{v.get('sel_range_bin_mean', '--')}")
        rows.append((m, cells))

    md = ["| method | FTU whole | FTU seg | FTU sel_r | BGT60 whole | BGT60 seg | BGT60 sel_r | PD whole | PD seg | PD sel_r |",
          "|--------|----------:|--------:|----------:|------------:|---------:|------------:|---------:|-------:|---------:|"]
    for m, cells in rows:
        md.append(f"| {m} | " + " | ".join(cells) + " |")

    # 分层表
    md.append("\n### 分层 (whole-window MAE)")
    md.append("| method | dataset | strata | n | whole | seg |")
    md.append("|--------|---------|--------|---:|------:|----:|")
    for ds, d in data.items():
        for m in METHOD_ORDER:
            if m not in d:
                continue
            for s, v in d[m]["by_strata"].items():
                seg = v.get("mae_seg_bpm")
                md.append(f"| {m} | {DATASET_LABELS[ds]} | {s} | {v['n']} | {v['mae_whole_bpm']:g} | {seg if seg is not None else '--':g} |")

    md_text = "\n".join(md) + "\n"
    (OUT / "summary_table.md").write_text(md_text)
    print(md_text)

    # 机器可读
    compact = {}
    for ds, d in data.items():
        compact[ds] = {m: {k: d[m][k] for k in ("n", "mae_whole_bpm", "mae_seg_bpm")}
                       for m in METHOD_ORDER if m in d}
    (OUT / "summary_compact.json").write_text(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
