"""Package the finite radar-only audit for a private CPU run."""
from pathlib import Path
import hashlib
import json

root=Path('experiment_runs/ellipse_applicability')
protocol=json.loads((root/'audit_protocol.json').read_text())
bundle={name:Path(name).read_text(encoding='utf-8') for name in protocol['source_hashes']}
for name,digest in protocol['source_hashes'].items(): assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
bundle['audit_protocol.json']=(root/'audit_protocol.json').read_text()
bundle['protocol.json']=Path('experiment_runs/synchronized_real_pilot/protocol.json').read_text()
stage=Path('tmp/ellipse_applicability_upload');stage.mkdir(exist_ok=True)
launch='from pathlib import Path\nimport json,subprocess,sys\n'
launch+='payload=json.loads('+repr(json.dumps(bundle))+')\n'
launch+="root=Path('/kaggle/working/ellipse_audit_code');root.mkdir(exist_ok=True)\n"
launch+="for name,text in payload.items():(root/name).write_text(text,encoding='utf-8')\n"
launch+="subprocess.run([sys.executable,str(root/'run_ellipse_applicability.py')],check=True)\n"
(stage/'launch.py').write_text(launch,encoding='utf-8')
(stage/'kernel-metadata.json').write_text(json.dumps({'id':'goldfish9901/fixed-ellipse-applicability-audit','title':'Fixed Ellipse Applicability Audit',
    'code_file':'launch.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_internet':True,
    'dataset_sources':[],'kernel_sources':[],'competition_sources':[]},indent=2))
print(stage)
