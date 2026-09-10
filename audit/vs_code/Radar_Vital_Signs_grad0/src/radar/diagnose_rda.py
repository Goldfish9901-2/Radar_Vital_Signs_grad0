"""Phase 0 RDA diagnostic — physical sanity, CPU only, no NN.

Implements the Phase 0 step of the RDA front-end search plan
(see ``docs/RDA_ALGORITHMS.md`` §3-phase). For each RDA candidate we compute
cheap, physically-interpretable diagnostics from the complex RDA cube (and, when
a phase candidate is selected, from its phase trace). The goal is NOT to find
the lowest-MAE RDA algorithm but to answer *"where does our current RDA drop HR
information?"* before spending any GPU.

Metrics (per candidate, aggregated across samples):
    - range/angle/doppler energy concentration (spectral flatness complement;
      higher = more concentrated = better localization)
    - HR-band energy: fraction of total energy inside the HR band [0.8, 3.0] Hz
    - spectral SNR: in-band / out-of-band energy ratio (dB)
    - phase variance: variance of the unwrapped phase trace (lower = cleaner)
    - phase continuity: mean |delta phase| between frames (lower = smoother)

Usage:
    python -m src.radar.diagnose_rda \
        --dataset PhysDrive --root V:/ --max-samples 20 \
        --out reports/rda_diagnostic.md --json reports/rda_diagnostic.json

If the dataset is not exported yet, the PhysDrive loader is used directly so the
script runs without a prior export step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import RDAConfig
from .pipeline import apply_rda_pipeline

# HR band for drivers / adults, in Hz (heart rate, not respiration).
HR_BAND_HZ = (0.8, 3.0)
FRAME_RATE_HZ = 20.0


# --------------------------------------------------------------------------
# candidate grid: each entry is an RDAConfig we want to probe.
# --------------------------------------------------------------------------


def default_candidates() -> List[Tuple[str, RDAConfig]]:
    """The cheap, interpretable baseline pool proposed in the design note.

    Avoids 5xN full experiments; this is ~8-10 classical candidates plus the
    cross-layer phase candidates.
    """
    return [
        ("current", RDAConfig(clutter="none", localization="energy", range_selection="global_energy", beamforming="fft")),
        ("mti", RDAConfig(clutter="mti", localization="energy", range_selection="global_energy", beamforming="fft")),
        ("highpass", RDAConfig(clutter="temporal_highpass", localization="energy", range_selection="global_energy", beamforming="fft")),
        ("pca", RDAConfig(clutter="pca", localization="energy", range_selection="global_energy", beamforming="fft")),
        ("rpca", RDAConfig(clutter="rpca", localization="energy", range_selection="global_energy", beamforming="fft")),
        ("hr_range", RDAConfig(clutter="temporal_highpass", localization="hr_band", range_selection="hr_band", beamforming="fft")),
        ("mvdr", RDAConfig(clutter="temporal_highpass", localization="energy", range_selection="global_energy", beamforming="mvdr")),
        ("bartlett", RDAConfig(clutter="temporal_highpass", localization="energy", range_selection="global_energy", beamforming="bartlett")),
        ("p0_phase", RDAConfig(clutter="temporal_highpass", localization="energy", range_selection="global_energy", beamforming="fft", phase="p0_single")),
        ("p1_phase", RDAConfig(clutter="temporal_highpass", localization="energy", range_selection="global_energy", beamforming="fft", phase="p1_diff")),
        ("p2_phase", RDAConfig(clutter="temporal_highpass", localization="energy", range_selection="global_energy", beamforming="fft", phase="p2_fusion")),
    ]


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def _energy_concentration(profile: np.ndarray) -> float:
    """1 - normalized spectral flatness, in [0,1]; higher = more concentrated."""
    p = np.asarray(profile, dtype=np.float64).ravel()
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    geo = np.exp(np.mean(np.log(p + 1e-12)))
    arith = total / max(p.size, 1)
    flatness = geo / (arith + 1e-12)
    return float(np.clip(1.0 - flatness, 0.0, 1.0))


def _hr_band_energy(signal_1d: np.ndarray) -> Tuple[float, float]:
    """Return (hr_band_fraction, spectral_snr_db) from a 1-D real signal."""
    x = np.asarray(signal_1d, dtype=np.float64)
    if x.size < 8:
        return float("nan"), float("nan")
    x = x - np.nanmean(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / FRAME_RATE_HZ)
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec = spec / (spec.sum() + 1e-12)
    band = (freqs >= HR_BAND_HZ[0]) & (freqs <= HR_BAND_HZ[1])
    in_band = spec[band].sum()
    out_band = max(spec[~band].sum(), 1e-12)
    frac = float(in_band)
    snr = 10.0 * np.log10(in_band / out_band + 1e-12)
    return frac, snr


def _phase_stats(trace: np.ndarray) -> Tuple[float, float]:
    """Return (phase_variance, phase_continuity) for a 1-D phase trace."""
    t = np.asarray(trace, dtype=np.float64)
    t = t[np.isfinite(t)]
    if t.size < 4:
        return float("nan"), float("nan")
    diff = np.diff(t)
    variance = float(np.nanvar(t))
    continuity = float(np.nanmean(np.abs(diff)))
    return variance, continuity


def diagnose_cube(cube: np.ndarray) -> Dict[str, float]:
    """Compute profile-level diagnostics for a complex RDA cube (F,D,A,R)."""
    cube = np.asarray(cube)
    mag = np.abs(cube) ** 2
    # range profile: collapse doppler, angle
    range_prof = mag.sum(axis=(0, 1, 2))
    angle_prof = mag.sum(axis=(0, 1, 3))
    doppler_prof = mag.sum(axis=(0, 2, 3))
    # dominant-axis 1-D signal for HR-band: collapse doppler+angle, keep range+frames
    axis1 = mag.sum(axis=(1, 2))  # (F, R)
    dominant_r = int(np.argmax(axis1.sum(axis=0)))
    signal = axis1[:, dominant_r].astype(np.float64)

    frac, snr = _hr_band_energy(signal)
    return {
        "range_concentration": _energy_concentration(range_prof),
        "angle_concentration": _energy_concentration(angle_prof),
        "doppler_concentration": _energy_concentration(doppler_prof),
        "hr_band_energy": frac,
        "spectral_snr_db": snr,
    }


def diagnose_phase(trace: np.ndarray) -> Dict[str, float]:
    var, cont = _phase_stats(trace)
    return {"phase_variance": var, "phase_continuity": cont}


# --------------------------------------------------------------------------
# data loading (PhysDrive direct; npz export if present)
# --------------------------------------------------------------------------


def iter_cubes(root: Path, dataset: str, max_samples: int):
    """Yield (cube (complex F,D,A,R), sample_id) for up to max_samples cubes."""
    if dataset != "PhysDrive":
        raise NotImplementedError(
            f"diagnose_rda currently supports PhysDrive directly; "
            f"{dataset} requires a prior export via export_all_datasets.py."
        )
    from ..data.loaders.physdrive_loader import PhysDriveDataLoader

    loader = PhysDriveDataLoader(str(root), frame_rate_hz=FRAME_RATE_HZ)
    count = 0
    for sample_id, mask in loader.iter_samples():
        if mask is None:
            continue
        cube = loader.load_radar_cube(sample_id)
        if cube is None or cube.size == 0:
            continue
        if not np.iscomplexobj(cube):
            cube = cube.astype(np.complex64)
        yield cube.astype(np.complex64), sample_id
        count += 1
        if count >= max_samples:
            break


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def run(
    root: Path,
    dataset: str,
    max_samples: int,
    candidates: List[Tuple[str, RDAConfig]],
) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, List[float]]] = {name: {} for name, _ in candidates}
    for cube, sid in iter_cubes(root, dataset, max_samples):
        for name, cfg in candidates:
            cfg.validate()
            try:
                if cfg.phase != "none":
                    # Diagnostic needs the *raw* phase (radians), not the z-scored
                    # trace used downstream, so variance/continuity are meaningful.
                    diag_cfg = RDAConfig(**{**cfg.to_dict(), "phase_kwargs": {"normalize": False}})
                    trace = apply_rda_pipeline(cube, diag_cfg, return_phase=True)
                    cube_diag = diagnose_cube(cube)  # phase candidates still improve cube first
                    ph_diag = diagnose_phase(trace)
                    merged = {**cube_diag, **ph_diag}
                else:
                    out = apply_rda_pipeline(cube, cfg)
                    merged = diagnose_cube(out)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] candidate '{name}' failed on {sid}: {exc}")
                continue
            for k, v in merged.items():
                agg[name].setdefault(k, []).append(v)

    # aggregate: mean over samples (nan-aware)
    result: Dict[str, Dict[str, float]] = {}
    for name, d in agg.items():
        result[name] = {k: float(np.nanmean(v)) if v else float("nan") for k, v in d.items()}
    return result


METRIC_ORDER = [
    "range_concentration",
    "angle_concentration",
    "doppler_concentration",
    "hr_band_energy",
    "spectral_snr_db",
    "phase_variance",
    "phase_continuity",
]


def to_markdown(result: Dict[str, Dict[str, float]]) -> str:
    lines = ["# RDA Diagnostic Report (Phase 0, CPU-only)", ""]
    lines.append("| candidate | " + " | ".join(METRIC_ORDER) + " |")
    lines.append("|" + "--|" * (len(METRIC_ORDER) + 1))
    for name, metrics in result.items():
        cells = [f"{metrics.get(m, float('nan')):.3f}" for m in METRIC_ORDER]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 RDA diagnostic (CPU only).")
    parser.add_argument("--root", default="V:/", help="dataset root (PhysDrive).")
    parser.add_argument("--dataset", default="PhysDrive", choices=["PhysDrive"])
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--out", default="reports/rda_diagnostic.md", help="markdown report path.")
    parser.add_argument("--json", default="reports/rda_diagnostic.json", help="json report path.")
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these candidate names.")
    args = parser.parse_args()

    candidates = default_candidates()
    if args.only:
        candidates = [(n, c) for n, c in candidates if n in set(args.only)]

    result = run(Path(args.root), args.dataset, args.max_samples, candidates)

    out_md = to_markdown(result)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_md)

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2))

    print(out_md)
    print(f"\nWrote {out_path} and {json_path}")


if __name__ == "__main__":
    main()
