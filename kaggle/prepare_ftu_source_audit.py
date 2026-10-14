"""Build an inference-only CPU notebook using the checkpoint's archived model code."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT/'experiment_runs/ftu_source_audit_2026-10-13/e2a_evidence/source_notebook.py'
if not archive.exists():
    archive = ROOT/'tmp/ftu_audit_discovery/e2a/nn-bench-e2a-hr-adavmd-proposed-ftu.py'
tree = ast.parse(archive.read_text(encoding='utf-8'))
files = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id == 'FILES' for t in node.targets))
selected = {k:v for k,v in files.items() if k.startswith('src/models/') or k in
            ('src/__init__.py','src/training/__init__.py','src/training/datasets.py')}
for name in ('audit_predictions','audit_feature_controls'):
    rel = 'src/training/'+name+'.py'
    selected[rel] = (ROOT/rel).read_text(encoding='utf-8')
buf = io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for name, text in selected.items(): z.writestr(name,text)
bundle = base64.b64encode(buf.getvalue()).decode('ascii')
stage = ROOT/'tmp/ftu_source_audit_upload'; stage.mkdir(parents=True,exist_ok=True)
runner = '''import base64, io, os, sys, zipfile
from pathlib import Path
code = Path('/kaggle/working/audit_code'); code.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z: z.extractall(code)
sys.path.insert(0,str(code))
roots = list(Path('/kaggle/input').rglob('preds_cycleformer_FTU_test.csv'))
if len(roots) != 1: raise RuntimeError(f'Ambiguous/missing E2a artifact: {roots}')
root = roots[0].parent
from src.training.audit_feature_controls import run
run(root/'model_outputs/cycleformer', root/'training_exports',
    Path('/kaggle/working/ftu_audit_controls'), roots[0])
'''.replace('BUNDLE',repr(bundle))
(stage/'run_ftu_audit.py').write_text(runner,encoding='utf-8')
metadata = dict(id='goldfish9901/ftu-source-signal-use-audit',title='FTU Source Signal Use Audit',
    code_file='run_ftu_audit.py', language='python',kernel_type='script',is_private=True,
    enable_gpu=False,enable_internet=False,dataset_sources=[],competition_sources=[],model_sources=[],
    kernel_sources=['goldfish9901/nn-bench-e2a-hr-adavmd-proposed-ftu'])
(stage/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
(stage/'source_manifest.json').write_text(json.dumps({k:hashlib.sha256(v.encode()).hexdigest()
    for k,v in selected.items()},indent=2),encoding='utf-8')
print(stage)
