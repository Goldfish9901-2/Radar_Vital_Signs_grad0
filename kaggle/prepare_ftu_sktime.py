"""Private CPU-only standard VMD experiment, using a temporary sktime package."""
import base64,io,json,zipfile,hashlib
from pathlib import Path

root=Path(__file__).resolve().parents[1]
paths=['src/__init__.py','src/data/__init__.py','src/data/experiment_ftu_sktime.py',
       'src/data/audit_ftu_p0.py']
paths += [p.relative_to(root).as_posix() for p in (root/'src/features').glob('*.py')]
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for rel in paths:z.writestr(rel,(root/rel).read_bytes())
stage=root/'tmp/ftu_sktime_upload';stage.mkdir(parents=True,exist_ok=True)
runner='''import subprocess,sys,base64,io,zipfile
from pathlib import Path
subprocess.run([sys.executable,'-m','pip','install','--quiet','sktime==1.2.0'],check=True)
code=Path('/kaggle/working/audit_code');code.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:z.extractall(code)
sys.path.insert(0,str(code))
from src.data.experiment_ftu_sktime import kaggle
out=Path('/kaggle/working/ftu_sktime');out.mkdir(exist_ok=True)
kaggle(out)
'''.replace('BUNDLE',repr(base64.b64encode(buf.getvalue()).decode()))
(stage/'run.py').write_text(runner,encoding='utf-8')
meta=dict(id='goldfish9901/ftu-sktime-vmd-baseline',title='FTU sktime VMD Baseline',
    code_file='run.py',language='python',kernel_type='script',is_private=True,
    enable_gpu=False,enable_internet=True,dataset_sources=['goldfish9901/ftu-rda-exports'],
    kernel_sources=['goldfish9901/nn-bench-e2a-hr-adavmd-proposed-ftu'],competition_sources=[],model_sources=[])
(stage/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
(stage/'source_manifest.json').write_text(json.dumps({r:hashlib.sha256((root/r).read_bytes()).hexdigest()
    for r in paths},indent=2),encoding='utf-8')
print(stage)
