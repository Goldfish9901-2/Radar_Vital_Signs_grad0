"""Package a read-only CPU audit of existing E2a windows and raw FTU sidecars."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
paths = ['src/__init__.py', 'src/data/__init__.py', 'src/data/audit_ftu_p0.py']
paths += [str(p.relative_to(root)).replace('\\', '/') for p in (root/'src/features').glob('*.py')]
buf = io.BytesIO()
with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
    for rel in paths: z.writestr(rel, (root/rel).read_bytes())
stage = root/'tmp/ftu_p0_audit_upload'; stage.mkdir(parents=True, exist_ok=True)
runner = '''import base64, io, sys, zipfile
from pathlib import Path
code = Path('/kaggle/working/audit_code'); code.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z: z.extractall(code)
sys.path.insert(0, str(code))
from src.data.audit_ftu_p0 import kaggle_audit
out=Path('/kaggle/working/ftu_p0'); out.mkdir(exist_ok=True)
kaggle_audit(out)
'''.replace('BUNDLE', repr(base64.b64encode(buf.getvalue()).decode()))
(stage/'run_p0.py').write_text(runner, encoding='utf-8')
meta=dict(id='goldfish9901/ftu-p0-mode-time-audit', title='FTU P0 Mode Time Audit',
    code_file='run_p0.py', language='python', kernel_type='script', is_private=True,
    enable_gpu=False, enable_internet=False, competition_sources=[], model_sources=[],
    dataset_sources=[f'goldfish9901/ftu-part{i}' for i in range(1, 6)],
    kernel_sources=['goldfish9901/nn-bench-e2a-hr-adavmd-proposed-ftu'])
(stage/'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
(stage/'source_manifest.json').write_text(json.dumps({p:hashlib.sha256((root/p).read_bytes()).hexdigest()
    for p in paths}, indent=2), encoding='utf-8')
print(stage)
