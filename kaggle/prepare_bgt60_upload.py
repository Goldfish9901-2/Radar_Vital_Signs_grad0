"""Build a small private Kaggle code dataset and GPU kernel upload folder."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PHASE = "--phase" in sys.argv[1:]
DIAGNOSTICS = "--diagnostics" in sys.argv[1:] or PHASE
STAGING = ROOT / "tmp" / ("bgt60_diagnostic_upload" if DIAGNOSTICS else "bgt60_frontend_upload")
DATASET = STAGING / "dataset"
KERNEL = STAGING / "kernel"
DATASET.mkdir(parents=True, exist_ok=True)
KERNEL.mkdir(parents=True, exist_ok=True)
files = list((ROOT / "src").rglob("*.py")) + [
    ROOT / "requirements.txt", ROOT / "validate_adc_rda_synthetic.py",
    ROOT / "docs/ADC_RDA_PHYSICAL_VALIDATION.md",
    ROOT / "kaggle/run_bgt60_frontend.py",
    ROOT / "validate_bgt60_diagnostic.py", ROOT / "kaggle/run_bgt60_diagnostics.py",
    ROOT / "docs/BGT60_CONFIGURATION_SEARCH.md",
    ROOT / "kaggle/run_bgt60_phase.py",
    ROOT / "validate_bgt60_phase.py",
]
calibration = ROOT / "kaggle/bgt60_calibration.json"
if calibration.is_file():
    files.append(calibration)
archive_path = DATASET / "radar_vital_signs_code.zip"
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, "Radar_Vital_Signs_grad0/" + path.relative_to(ROOT).as_posix())
metadata = {
    "id": "goldfish9901/bgt60-frontend-code",
    "title": "BGT60 Frontend Ablation Code",
    "isPrivate": True, "licenses": [{"name": "other"}],
}
(DATASET / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
RUNNER = "run_bgt60_phase.py" if PHASE else ("run_bgt60_diagnostics.py" if DIAGNOSTICS else "run_bgt60_frontend.py")
shutil.copy2(ROOT / "kaggle" / RUNNER, KERNEL)
kernel = {
    "id": "goldfish9901/bgt60-direct-phase-audit" if PHASE else ("goldfish9901/bgt60-adc-roi-phase-diagnostics" if DIAGNOSTICS else "goldfish9901/bgt60-cycleformer-frontend-ablation"),
    "title": "BGT60 Direct Phase Audit" if PHASE else ("BGT60 ADC ROI Phase Diagnostics" if DIAGNOSTICS else "BGT60 CycleFormer Frontend Ablation"),
    "code_file": RUNNER, "language": "python", "kernel_type": "script",
    "is_private": True, "enable_gpu": not DIAGNOSTICS, "enable_internet": True,
    "dataset_sources": ["goldfish9901/bgt60-frontend-code", "goldfish9901/bgt60tr13c-vital-signs"],
    "competition_sources": [], "kernel_sources": [], "model_sources": [],
}
(KERNEL / "kernel-metadata.json").write_text(json.dumps(kernel, indent=2), encoding="utf-8")
receipt = {"files": len(files), "zip_bytes": archive_path.stat().st_size,
           "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
           "dataset": metadata["id"], "kernel": kernel["id"]}
(STAGING / "bundle_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2))
