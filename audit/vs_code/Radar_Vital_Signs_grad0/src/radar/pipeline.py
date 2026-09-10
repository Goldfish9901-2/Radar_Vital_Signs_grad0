"""Stage registry + ordered pipeline runner for the RDA front-end pool.

Each stage is a named function over a complex RDA cube ``(F, D, A, R)`` that
returns a new complex cube of the same shape *contract* (see notes per stage).
Stages are registered into ``RDA_STAGE_REGISTRY`` so :class:`RDAConfig` can
select them by string and so new paper-derived methods can be plugged in without
touching the runner.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

# cube shape convention used everywhere in this package:
#   axis 0 = frames (time, ~20 Hz)
#   axis 1 = doppler bins
#   axis 2 = angle / RX-spatial bins
#   axis 3 = range bins
RDARCube = np.ndarray  # complex64, shape (F, D, A, R)

# stage name -> {method_name: callable}
# Cube stages return a complex cube of the same (F,D,A,R) shape. The "phase"
# stage is cross-layer: it consumes the localized cube and returns a 1-D phase
# trace (see src/radar/phase.py and docs/RDA_ALGORITHMS.md §2 layer boundary).
RDA_STAGE_REGISTRY: Dict[str, Dict[str, Callable[..., RDARCube]]] = {
    "clutter": {},
    "localization": {},
    "range_selection": {},
    "beamforming": {},
    "phase": {},
}

# Cube stages run in this fixed order; output is still a cube.
STAGE_ORDER = ("clutter", "localization", "range_selection", "beamforming")


def register_stage(stage: str, name: str, func: Callable[..., RDARCube]) -> None:
    if stage not in RDA_STAGE_REGISTRY:
        RDA_STAGE_REGISTRY[stage] = {}
    RDA_STAGE_REGISTRY[stage][name] = func


def list_stage_methods() -> Dict[str, List[str]]:
    return {stage: sorted(methods) for stage, methods in RDA_STAGE_REGISTRY.items()}


def _resolve(stage: str, method: str) -> Callable[..., RDARCube]:
    methods = RDA_STAGE_REGISTRY.get(stage, {})
    if method not in methods:
        from .config import StageNotFoundError

        raise StageNotFoundError(stage, method, list(methods))
    return methods[method]


def apply_rda_pipeline(
    cube: np.ndarray,
    config,
    *,
    record_meta: bool = False,
    return_phase: bool = False,
) -> Any:
    """Run the selected stages in order.

    Args:
        cube: complex RDA cube ``(F, D, A, R)``.
        config: :class:`src.radar.config.RDAConfig` (or dataclass-compatible dict).
        record_meta: if True, also return a per-stage diagnostic dict.
        return_phase: if True and ``config.phase != "none"``, return the phase
            trace produced by the phase stage instead of the cube. Useful for the
            Phase 0 diagnostic / representation-probe gate without touching EDACM.

    Returns:
        The transformed cube, the phase trace, or ``(obj, meta)`` when
        ``record_meta`` is True.
    """
    if not isinstance(cube, np.ndarray):
        cube = np.asarray(cube)
    if not np.iscomplexobj(cube):
        cube = cube.astype(np.complex64)
    else:
        cube = cube.astype(np.complex64)

    if hasattr(config, "to_dict"):
        cfg = config.to_dict()
    else:
        cfg = dict(config)

    meta: Dict[str, Any] = {}
    current = cube
    for stage in STAGE_ORDER:
        method = cfg.get(stage, "none")
        if method in (None, "none", ""):
            continue
        kwargs = cfg.get(f"{stage}_kwargs", {}) or {}
        func = _resolve(stage, method)
        before = current
        current = func(current, **kwargs)
        # enforce shape contract: stages may not change (F, D, A, R)
        if current.shape != before.shape:
            raise ValueError(
                f"RDA stage '{stage}/{method}' changed cube shape "
                f"{before.shape} -> {current.shape}; stages must preserve (F,D,A,R)."
            )
        if record_meta:
            meta[stage] = {"method": method, "kwargs": kwargs}

    # Cross-layer phase stage: operates on the localized cube, returns a trace.
    phase_method = cfg.get("phase", "none")
    phase_trace: Any = None
    if phase_method not in (None, "none", ""):
        phase_kwargs = cfg.get("phase_kwargs", {}) or {}
        phase_func = _resolve("phase", phase_method)
        phase_trace = phase_func(current, **phase_kwargs)
        if record_meta:
            meta["phase"] = {"method": phase_method, "kwargs": phase_kwargs}

    if record_meta:
        return current, meta
    if return_phase:
        return phase_trace
    return current
