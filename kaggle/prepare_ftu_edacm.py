"""Package the FTU v2 CPU export; no model training."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
paths = ["src/__init__.py", "src/data/__init__.py", "src/data/build_training_dataset.py",
         "src/data/ftu_timebase.py", "src/data/experiment_ftu_edacm.py",
         "src/training/__init__.py", "src/training/datasets.py"]
for directory in ["src/features", "src/models"]:
    paths += [p.relative_to(root).as_posix() for p in (root / directory).glob("*.py")]
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
    for rel in paths:
        archive.writestr(rel, (root / rel).read_bytes())
stage = root / "tmp/ftu_edacm_upload"
stage.mkdir(parents=True, exist_ok=True)
runner = '''import base64, io, os, sys, zipfile
from pathlib import Path
code = Path('/kaggle/working/code'); code.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z: z.extractall(code)
sys.path.insert(0,str(code)); os.chdir(code)
from src.data.experiment_ftu_edacm import run
run(Path('/kaggle/input'), Path('/kaggle/working/ftu_timebase_edacm_v2'))
import shutil
shutil.make_archive('/kaggle/working/ftu_timebase_edacm_v2', 'zip',
                    '/kaggle/working/ftu_timebase_edacm_v2')
'''.replace("BUNDLE", repr(base64.b64encode(buffer.getvalue()).decode()))
(stage / "run.py").write_text(runner, encoding="utf-8")
meta = dict(id="goldfish9901/ftu-timebase-edacm-v2", title="FTU Timebase EDACM v2",
    code_file="run.py", language="python", kernel_type="script", is_private=True,
    enable_gpu=False, enable_internet=False, dataset_sources=["goldfish9901/ftu-rda-exports"],
    kernel_sources=["goldfish9901/nn-bench-e2a-hr-adavmd-proposed-ftu"],
    competition_sources=[], model_sources=[])
(stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
(stage / "source_manifest.json").write_text(json.dumps({r: hashlib.sha256((root/r).read_bytes()).hexdigest()
    for r in paths}, indent=2), encoding="utf-8")
print(stage)
