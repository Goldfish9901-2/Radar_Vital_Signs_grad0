"""Checkpoint and run-configuration helpers.

Training, evaluation, and source-free adaptation all read the same model
artifacts. This module prevents subtle drift in how scripts resolve
``run_config.json`` and checkpoint paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_run_config(model_dir: Path | None) -> Dict[str, Any]:
    """Load ``run_config.json`` when present; return an empty dict otherwise."""
    if model_dir is None:
        return {}
    path = model_dir / "run_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_checkpoint(model_dir: Path | None, checkpoint: Path | None) -> Path:
    """Resolve an explicit checkpoint or default to ``<model_dir>/best.pt``."""
    if checkpoint is not None:
        return checkpoint
    if model_dir is None:
        raise ValueError("Either a model directory or checkpoint path must be provided.")
    return model_dir / "best.pt"
