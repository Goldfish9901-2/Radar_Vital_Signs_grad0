"""RDA front-end algorithm pool for radar vital-signs.

This package implements the *replaceable RDA front-end* described in
``docs/RDA_ALGORITHMS.md``. The goal is NOT to ship one new "RDA algorithm" as a
black box, but to expose each processing stage as an independently swappable
module so several full pipelines can be composed and then judged by the
downstream representation validator + backbone benchmark.

Five replaceable stages operate on the complex RDA cube shaped
``(frames, doppler, angle, range)``:

    1. clutter        -- suppress static / stationary reflection
    2. localization   -- find the spatial region containing the subject
    3. range_selection-- pick the range bins fed downstream
    4. beamforming    -- angular / spatial filtering across RX channels
    5. phase          -- (cross-layer) micro-motion phase/representation note

Each stage is a registry of named functions. A :class:`RDAConfig` selects one
method per stage; :func:`apply_rda_pipeline` runs them in order and returns a new
complex cube of the SAME logical shape contract expected by
``src.features.representations.radar_to_feature_bundle``.

Design boundary (important):
    This layer runs *on top of* an already-constructed RDA cube (e.g. the
    publisher-processed cube from PhysDrive, or the cube built locally for FTU /
    BGT60). It does NOT re-derive RDA from raw ADC. That keeps it usable today
    while the raw-ADC path matures. Re-processing an existing cube is still
    meaningful: the publisher's static removal / localization / crop may be
    sub-optimal for vital signs (esp. in-vehicle, where PhysDrive MAE ~10.75 vs a
    ~10.94 constant baseline suggests the spatial region may be mis-localized).

CPU-only by design -- all methods here are cheap linear / filtering ops meant for
the "Phase 0 physical sanity" + "Phase 1 representation validator" gates before
any GPU work.
"""

from __future__ import annotations

from .config import RDAConfig, StageNotFoundError
from .pipeline import (
    apply_rda_pipeline,
    list_stage_methods,
    RDA_STAGE_REGISTRY,
)
from .clutter import register_clutter_methods
from .localization import register_localization_methods
from .range_selection import register_range_selection_methods
from .beamforming import register_beamforming_methods

# Register all built-in methods at import time so the registry is populated.
register_clutter_methods()
register_localization_methods()
register_range_selection_methods()
register_beamforming_methods()

__all__ = [
    "RDAConfig",
    "StageNotFoundError",
    "apply_rda_pipeline",
    "list_stage_methods",
    "RDA_STAGE_REGISTRY",
]
