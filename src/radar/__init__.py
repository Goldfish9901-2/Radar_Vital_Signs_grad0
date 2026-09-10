"""RDA front-end algorithm pool.

Replaces ad-hoc ``convert_adc_to_rda_vN`` chains with a configurable, registry-
driven pipeline. A single :class:`RDAConfig` selects one method per stage;
swapping a stage is a one-line change.

Layers (see docs/RDA_ALGORITHMS.md):
    cube stages : clutter -> localization -> range_selection -> beamforming
                  each returns a complex cube of the same (F, D, A, R) shape
    cross-layer : phase  (consumes the cube, returns a 1-D phase trace)

Importing this package registers every candidate method into
``RDA_STAGE_REGISTRY``; build a config and call :func:`apply_rda_pipeline`.
"""

from __future__ import annotations

from .config import RDAConfig, StageNotFoundError
from .pipeline import (
    RDA_STAGE_REGISTRY,
    STAGE_ORDER,
    RDARCube,
    DEFAULT_FS,
    ensure_rda_4d,
    register_stage,
    list_stage_methods,
    apply_rda_pipeline,
)
from . import clutter, localization, range_selection, beamforming, phase

# Register every candidate method into the global registry on import.
clutter.register_clutter_methods()
localization.register_localization_methods()
range_selection.register_range_selection_methods()
beamforming.register_beamforming_methods()
phase.register_phase_methods()

__all__ = [
    "RDAConfig",
    "StageNotFoundError",
    "RDA_STAGE_REGISTRY",
    "STAGE_ORDER",
    "RDARCube",
    "DEFAULT_FS",
    "ensure_rda_4d",
    "register_stage",
    "list_stage_methods",
    "apply_rda_pipeline",
    "clutter",
    "localization",
    "range_selection",
    "beamforming",
    "phase",
]
