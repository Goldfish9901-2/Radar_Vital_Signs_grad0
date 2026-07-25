"""Shared metric aggregation for heart-rate experiments.

All experiment types report the same core regression metrics so proposed-method
runs and baseline runs can be compared directly: MAE, tolerance hit rates, label
mean, prediction mean, and grouped summaries.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable

import torch

from src.training.datasets import LabelStats


def denormalize(value: torch.Tensor, stats: LabelStats) -> torch.Tensor:
    """Convert normalized model output back to BPM."""
    return value * stats.std + stats.mean


def mae_bpm(pred_norm: torch.Tensor, y_norm: torch.Tensor, stats: LabelStats) -> torch.Tensor:
    """Compute mean absolute error in BPM from normalized tensors."""
    return torch.mean(torch.abs(denormalize(pred_norm, stats) - denormalize(y_norm, stats)))


def rmse_bpm(pred_norm: torch.Tensor, y_norm: torch.Tensor, stats: LabelStats) -> torch.Tensor:
    """Compute root-mean-square error in BPM from normalized tensors."""
    err = denormalize(pred_norm, stats) - denormalize(y_norm, stats)
    return torch.sqrt(torch.mean(err.pow(2)))


def participant_id(dataset: str, group_key: str, sample_tag: str) -> str:
    """Resolve a stable participant/session identifier across datasets."""
    if "/participant/" in group_key:
        return group_key.rsplit("/", 1)[-1]
    if "/session/" in group_key:
        return group_key.rsplit("/", 1)[-1]
    if dataset in {"FTU", "BGT60TR13C"}:
        match = re.match(r"p0?(\d+)", sample_tag)
        if match:
            return match.group(1)
    if dataset == "PhysDrive":
        return sample_tag.split("_", 1)[0]
    return group_key or sample_tag or "unknown"


def append_prediction_rows(
    rows: list[Dict[str, Any]],
    batch: Dict[str, Any],
    pred_norm: torch.Tensor,
    stats: LabelStats,
    include_participant: bool = True,
) -> None:
    """Append per-window prediction rows used by all aggregators."""
    pred_bpm = denormalize(pred_norm.detach().cpu(), stats)
    y_bpm = batch["y_bpm"].detach().cpu()
    for idx in range(int(y_bpm.numel())):
        dataset = batch["dataset"][idx]
        group_key = batch["group_key"][idx]
        sample_tag = batch.get("sample_tag", [""] * int(y_bpm.numel()))[idx]
        label = float(y_bpm[idx])
        pred = float(pred_bpm[idx])
        row = {
            "dataset": dataset,
            "group_key": group_key,
            "sample_tag": sample_tag,
            "label_bpm": label,
            "pred_bpm": pred,
            "abs_error_bpm": abs(pred - label),
        }
        if include_participant:
            row["participant_id"] = participant_id(dataset, group_key, sample_tag)
        rows.append(row)


def aggregate(rows: Iterable[Dict[str, Any]], key: str | None = None) -> Dict[str, Any]:
    """Aggregate prediction rows overall or by a metadata key."""
    buckets: Dict[str, list[Dict[str, Any]]] = {}
    if key is None:
        buckets["overall"] = list(rows)
    else:
        for row in rows:
            buckets.setdefault(str(row.get(key, "")), []).append(row)

    metrics: Dict[str, Any] = {}
    for name, values in buckets.items():
        errors = [float(row["abs_error_bpm"]) for row in values]
        labels = [float(row["label_bpm"]) for row in values]
        preds = [float(row["pred_bpm"]) for row in values]
        sq_errors = [(pred - label) ** 2 for pred, label in zip(preds, labels)]
        pearson_r = math.nan
        if len(values) >= 2:
            label_mean = sum(labels) / len(labels)
            pred_mean = sum(preds) / len(preds)
            numerator = sum((label - label_mean) * (pred - pred_mean) for label, pred in zip(labels, preds))
            label_var = sum((label - label_mean) ** 2 for label in labels)
            pred_var = sum((pred - pred_mean) ** 2 for pred in preds)
            denom = math.sqrt(label_var * pred_var)
            if denom > 1e-12:
                pearson_r = float(numerator / denom)
        metrics[name] = {
            "count": len(values),
            "mae_bpm": float(sum(errors) / len(errors)) if errors else math.nan,
            "rmse_bpm": float(math.sqrt(sum(sq_errors) / len(sq_errors))) if sq_errors else math.nan,
            "pearson_r": pearson_r,
            "within_3bpm_percent": float(100.0 * sum(error <= 3.0 for error in errors) / len(errors)) if errors else math.nan,
            "within_5bpm_percent": float(100.0 * sum(error <= 5.0 for error in errors) / len(errors)) if errors else math.nan,
            "within_10bpm_percent": float(100.0 * sum(error <= 10.0 for error in errors) / len(errors)) if errors else math.nan,
            "label_mean_bpm": float(sum(labels) / len(labels)) if labels else math.nan,
            "pred_mean_bpm": float(sum(preds) / len(preds)) if preds else math.nan,
        }
    return metrics


def tolerance_metrics(rows: list[Dict[str, Any]]) -> Dict[str, float]:
    """Return tolerance hit rates for a flat prediction row list."""
    if not rows:
        return {"within_3bpm_percent": math.nan, "within_5bpm_percent": math.nan}
    errors = [float(row["abs_error_bpm"]) for row in rows]
    return {
        "within_3bpm_percent": 100.0 * sum(error <= 3.0 for error in errors) / len(errors),
        "within_5bpm_percent": 100.0 * sum(error <= 5.0 for error in errors) / len(errors),
        "within_10bpm_percent": 100.0 * sum(error <= 10.0 for error in errors) / len(errors),
    }
