"""CPU-only 54-record target-selection diagnostic bundle."""
import base64,hashlib,io,json,subprocess,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
paths=['src/__init__.py','src/data/__init__.py','src/data/ftu_timebase.py','src/data/audit_ftu_targets.py']
paths += [p.relative_to(root).as_posix() for p in (root/'src/features').glob('*.py')]
files={p:(root/p).read_bytes() for p in paths}
for name in ['selected_records.csv','protocol.json']:files[name]=(root/'experiment_runs/ftu_target_audit'/name).read_bytes()
manifest=dict(files={p:hashlib.sha256(b).hexdigest() for p,b in files.items()},
    base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    input_archive_sha256=hashlib.sha256((root/'experiment_runs/ftu_timebase_edacm_v2/full_artifacts.zip').read_bytes()).hexdigest())
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for p,b in files.items():z.writestr(p,b)
    z.writestr('source_manifest.json',json.dumps(manifest,indent=2))
runner="""import base64,io,os,sys,zipfile
from pathlib import Path
code=Path('/kaggle/working/code');code.mkdir()
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:z.extractall(code)
sys.path.insert(0,str(code));os.chdir(code)
from src.data.audit_ftu_targets import run
run(Path('/kaggle/input'),Path('/kaggle/working/ftu_target_audit'),code)
""".replace('BUNDLE',repr(base64.b64encode(buf.getvalue()).decode()))
stage=root/'tmp/ftu_target_audit_upload';stage.mkdir(parents=True,exist_ok=True)
(stage/'run.py').write_text(runner,encoding='utf-8')
(stage/'kernel-metadata.json').write_text(json.dumps(dict(id='goldfish9901/ftu-target-selection-audit',title='FTU Target Selection Audit',
    code_file='run.py',language='python',kernel_type='script',is_private=True,enable_gpu=False,enable_internet=False,
    dataset_sources=['goldfish9901/ftu-rda-exports'],kernel_sources=['goldfish9901/ftu-timebase-edacm-v2'],competition_sources=[],model_sources=[]),indent=2))
print(stage)
