"""Bundle archived E2a feature implementations for three continuous-record diagnostics."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
evidence = root/'experiment_runs/ftu_source_audit_2026-10-13/e2a_evidence'
tree = ast.parse((evidence/'source_notebook.py').read_text(encoding='utf-8'))
files = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id == 'FILES' for t in n.targets))
selected = {k:v for k,v in files.items() if k.startswith('src/features/') or k in
            ('src/__init__.py','src/data/__init__.py','src/data/build_training_dataset.py',
             'src/training/__init__.py')}
for rel in ('src/data/audit_ftu_continuous.py','src/training/audit_predictions.py'):
    selected[rel] = (root/rel).read_text(encoding='utf-8')
cases = json.loads((root/'experiment_runs/ftu_source_audit_2026-10-13/e2a_cycleformer/summary.json').read_text())
selected['src/data/ftu_cases.json'] = json.dumps(cases['examples_for_continuous_signal_audit'])
buf = io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for name,body in selected.items(): z.writestr(name,body)
bundle = base64.b64encode(buf.getvalue()).decode()
stage = root/'tmp/ftu_continuous_audit_upload'; stage.mkdir(parents=True,exist_ok=True)
runner = '''import base64,io,sys,zipfile
from pathlib import Path
code=Path('/kaggle/working/audit_code');code.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:z.extractall(code)
sys.path.insert(0,str(code))
from src.data.audit_ftu_continuous import main
main()
'''.replace('BUNDLE',repr(bundle))
(stage/'run_continuous.py').write_text(runner,encoding='utf-8')
meta=dict(id='goldfish9901/ftu-continuous-stage-audit',title='FTU Continuous Stage Audit',
    code_file='run_continuous.py',language='python',kernel_type='script',is_private=True,
    enable_gpu=False,enable_internet=False,dataset_sources=['goldfish9901/ftu-rda-exports'],
    kernel_sources=['goldfish9901/nn-bench-e2a-hr-adavmd-proposed-ftu'],competition_sources=[],model_sources=[])
(stage/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
(stage/'source_manifest.json').write_text(json.dumps({k:hashlib.sha256(v.encode()).hexdigest()
    for k,v in selected.items()},indent=2),encoding='utf-8')
print(stage)
