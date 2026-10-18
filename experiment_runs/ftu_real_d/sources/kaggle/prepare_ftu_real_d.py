"""Bundle fixed real FTU D experiment and original capture split."""
import base64,hashlib,io,json,subprocess,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
paths=['src/__init__.py','src/training/__init__.py','src/training/ftu_position_protocol.py',
       'src/training/ftu_spectrum_factorial.py','src/training/run_ftu_real_d.py']
files={p:(root/p).read_bytes() for p in paths}
files['record_split.csv']=(root/'experiment_runs/ftu_record_learnability/record_split.csv').read_bytes()
manifest=dict(files={p:hashlib.sha256(b).hexdigest() for p,b in files.items()},
              base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for p,b in files.items():z.writestr(p,b)
    z.writestr('source_manifest.json',json.dumps(manifest,indent=2))
datahash=hashlib.sha256((root/'experiment_runs/ftu_timebase_edacm_v2/full_artifacts.zip').read_bytes()).hexdigest()
runner="""import base64,io,os,sys,zipfile
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
code=Path('/kaggle/working/code');code.mkdir()
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:z.extractall(code)
sys.path.insert(0,str(code));os.chdir(code)
from src.training.run_ftu_real_d import run
run(DATAHASH)
""".replace('BUNDLE',repr(base64.b64encode(buf.getvalue()).decode())).replace('DATAHASH',repr(datahash))
stage=root/'tmp/ftu_real_d_upload';stage.mkdir(parents=True,exist_ok=True)
(stage/'run.py').write_text(runner,encoding='utf-8')
(stage/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
(stage/'kernel-metadata.json').write_text(json.dumps(dict(id='goldfish9901/ftu-real-d-controls',
    title='FTU Real D Controls',code_file='run.py',language='python',kernel_type='script',is_private=True,
    enable_gpu=True,enable_internet=False,dataset_sources=[],kernel_sources=['goldfish9901/ftu-timebase-edacm-v2'],competition_sources=[],model_sources=[]),indent=2))
print(stage)
