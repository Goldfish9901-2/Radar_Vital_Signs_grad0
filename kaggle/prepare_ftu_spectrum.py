"""Bundle the frozen synthetic factorial, without external datasets."""
import base64,hashlib,io,json,subprocess,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
paths=['src/__init__.py','src/training/__init__.py','src/training/ftu_spectrum_factorial.py']
paths += [p.relative_to(root).as_posix() for p in (root/'src/features').glob('*.py')]
manifest=dict(files={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
              base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
reference=(root/'experiment_runs/ftu_record_learnability/synthetic_features.npz').read_bytes()
manifest['files']['reference_features.npz']=hashlib.sha256(reference).hexdigest()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for p in paths:z.writestr(p,(root/p).read_bytes())
    z.writestr('reference_features.npz',reference)
    z.writestr('source_manifest.json',json.dumps(manifest,indent=2))
runner="""import base64,io,os,sys,zipfile
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
code=Path('/kaggle/working/code');code.mkdir()
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:z.extractall(code)
sys.path.insert(0,str(code));os.chdir(code)
from src.training.ftu_spectrum_factorial import run
run()
""".replace('BUNDLE',repr(base64.b64encode(buf.getvalue()).decode()))
stage=root/'tmp/ftu_spectrum_upload';stage.mkdir(parents=True,exist_ok=True)
(stage/'run.py').write_text(runner,encoding='utf-8')
(stage/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
(stage/'kernel-metadata.json').write_text(json.dumps(dict(id='goldfish9901/ftu-spectrum-factorial',
    title='FTU Spectrum Factorial',code_file='run.py',language='python',kernel_type='script',is_private=True,
    enable_gpu=True,enable_internet=False,dataset_sources=[],kernel_sources=[],competition_sources=[],model_sources=[]),indent=2))
print(stage)
