"""Representation contract for the radar vital-signs benchmark.

A *representation* is the model-ready output of the signal-processing front-end.
It is the single boundary between "raw radar" and "neural backbone", so it is
made a first-class, stable type rather than an ad-hoc dict.

Contract
--------
Every representation emits, at minimum, a time-domain tensor ``x_time`` and a
frequency-domain tensor ``x_freq`` with shapes ``(C, T)`` and ``(C, F)``. The
number of channels ``C`` is **representation-defined** (e.g. 7 for the proposed
EDACM+HR-AdaVMD, but a new method may emit 12); models and the training pipeline
must derive ``C`` from the data, never hard-code it.

``meta`` is a *fixed-schema* dict (not free-form) so downstream analysis scripts
never have to guess:
    sample_rate, representation, generator_version,
    time_channels, freq_channels, window_size, freq_length

New signal-processing methods should be implemented as functions that return a
:class:`RadarRepresentation` (see ``src/features/representations.py``). They only
ever need the raw radar array; the training, model, and benchmark code stays
untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

# Fixed-schema keys every ``meta`` dict is expected to carry. Documented here so
# analysis/eval code can rely on them instead of guessing per-method field names.
META_KEYS = (
    "sample_rate",
    "representation",
    "generator_version",
    "time_channels",
    "freq_channels",
    "window_size",
    "freq_length",
)

# Maps a (legacy) dict key to the dataclass attribute that backs it. This lets
# the existing pipeline keep using ``bundle["x_time"]``, ``"x_time" in bundle``,
# ``bundle.get("x_freq")`` and even ``bundle["x"] = ...`` unchanged.
_KEY_MAP = {
    "x_time": "x_time",
    "x_freq": "x_freq",
    "x": "x",
    "x_rda": "x_rda",
    "freq_hz": "freq_hz",
    "meta": "meta",
    "representation_name": "representation_name",
    "representation_version": "representation_version",
}


@dataclass
class RadarRepresentation:
    """Model-ready output of a radar signal-processing front-end.

    Attributes
    ----------
    x_time, x_freq:
        Time- and frequency-domain tensors of shape ``(C, T)`` / ``(C, F)``.
        Either may be ``None`` for representations that only emit one domain
        (those are not compatible with the dual time+freq backbones).
    representation_name:
        Stable name, e.g. ``"proposed"``, ``"edacm_only"``, ``"new_method"``.
    representation_version:
        Version string for the method; recorded by the benchmark for exact
        reproducibility.
    meta:
        Fixed-schema metadata dict (see :data:`META_KEYS`).
    x, x_rda, freq_hz:
        Auxiliary arrays kept for export compatibility.
    """

    x_time: Optional[np.ndarray] = None
    x_freq: Optional[np.ndarray] = None
    representation_name: str = "unknown"
    representation_version: str = "1.0"
    meta: Dict[str, Any] = field(default_factory=dict)
    x: Optional[np.ndarray] = None
    x_rda: Optional[np.ndarray] = None
    freq_hz: Optional[np.ndarray] = None

    # --- dict-compatible access so the existing pipeline needs no edits ---
    def __getitem__(self, key: str) -> Any:
        if key not in _KEY_MAP:
            raise KeyError(key)
        return getattr(self, _KEY_MAP[key])

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in _KEY_MAP:
            raise KeyError(f"RadarRepresentation has no field {key!r}")
        setattr(self, _KEY_MAP[key], value)

    def __contains__(self, key: str) -> bool:
        if key not in _KEY_MAP:
            return False
        return getattr(self, _KEY_MAP[key]) is not None

    def get(self, key: str, default: Any = None) -> Any:
        if key not in _KEY_MAP:
            return default
        value = getattr(self, _KEY_MAP[key])
        return value if value is not None else default


def channels_of(rep: RadarRepresentation) -> "tuple[Optional[int], Optional[int]]":
    """Return ``(time_channels, freq_channels)`` derived from the actual arrays."""
    t = int(rep.x_time.shape[0]) if rep.x_time is not None else None
    f = int(rep.x_freq.shape[0]) if rep.x_freq is not None else None
    return t, f
