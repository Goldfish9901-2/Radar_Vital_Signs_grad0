"""Bundle frozen FTU A-D diagnostics; all model fitting stays on Kaggle."""
import base64,hashlib,io,json,subprocess,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
paths=['src/__init__.py','src/training/__init__.py','src/training/ftu_position_protocol.py',
       'src/training/ftu_tcn_protocol.py','src/training/run_ftu_position.py']
for directory in ['src/models','src/features']:
    paths += [p.relative_to(root).as_posix() for p in (root/directory).glob('*.py')]
source=dict(files={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
            base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for p in paths:z.writestr(p,(root/p).read_bytes())
    z.writestr('source_manifest.json',json.dumps(source,indent=2))
data=hashlib.sha256((root/'experiment_runs/ftu_timebase_edacm_v2/full_artifacts.zip').read_bytes()).hexdigest()
checkpoint=hashlib.sha256((root/'experiment_runs/ftu_tcn_p1_seed42/model/best.pt').read_bytes()).hexdigest()
runner='''import base64,io,os,sys,zipfile
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
code=Path('/kaggle/working/code');code.mkdir()
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:z.extractall(code)
sys.path.insert(0,str(code));os.chdir(code)
from src.training.run_ftu_position import run
run(DATAHASH,CHECKPOINTHASH)
'''.replace('BUNDLE',repr(base64.b64encode(buf.getvalue()).decode())).replace('DATAHASH',repr(data)).replace('CHECKPOINTHASH',repr(checkpoint))
stage=root/'tmp/ftu_position_upload';stage.mkdir(parents=True,exist_ok=True)
(stage/'run.py').write_text(runner,encoding='utf-8')
(stage/'source_manifest.json').write_text(json.dumps(source,indent=2))
meta=dict(id='goldfish9901/ftu-position-abcd-diagnostics',title='FTU Position ABCD Diagnostics',
    code_file='run.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,
    enable_internet=False,dataset_sources=[],kernel_sources=['goldfish9901/ftu-timebase-edacm-v2',
    'goldfish9901/ftu-edacm-tcn-p1-seed42'],competition_sources=[],model_sources=[])
(stage/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
print(stage)
