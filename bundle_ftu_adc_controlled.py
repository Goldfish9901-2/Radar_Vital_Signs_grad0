"""Bundle frozen candidates that have passed the local synthetic gate."""
import base64,hashlib,io,json,zipfile,shutil
from pathlib import Path
root=Path(__file__).resolve().parent;code=root/'experiment_runs/ftu_adc_controlled'
gate=json.loads((code/'synthetic_gate.json').read_text());assert gate['candidates_pass']
paths=[p for p in (code/'sources').glob('*.py')]
paths+=[code/n for n in ['protocol.json','selected_records.csv','expected_rda_hashes.json','synthetic_gate.json','source_manifest.json']]
paths+=[code/n for n in ['recovery_amendment.json','official_recovery/p03_official_member.json','official_recovery/p03_member_header.bin']]
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for p in paths:z.writestr(p.relative_to(code).as_posix(),p.read_bytes())
runner='''import base64,io,os,sys,zipfile
from pathlib import Path
code=Path('/kaggle/working/code');code.mkdir()
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:z.extractall(code)
sys.path.insert(0,str(code/'sources'))
from audit_ftu_adc_controlled import real
from recover_ftu_p03 import recover
supplement=recover(code,Path('/kaggle/working/p03_recovered'))
real(Path('/kaggle/input'),code,Path('/kaggle/working/ftu_adc_controlled'),supplement_root=supplement)
'''.replace('BUNDLE',repr(base64.b64encode(buf.getvalue()).decode()))
stage=root/'tmp/ftu_adc_controlled_upload';stage.mkdir(exist_ok=True)
(stage/'run.py').write_text(runner)
meta=dict(id='goldfish9901/ftu-adc-controlled-six',title='FTU ADC Controlled Six',code_file='run.py',language='python',kernel_type='script',
    is_private=True,enable_gpu=False,enable_internet=True,dataset_sources=['goldfish9901/ftu-rda-exports']+[f'goldfish9901/ftu-part{i}' for i in range(1,6)],
    kernel_sources=[],competition_sources=[],model_sources=[])
(stage/'kernel-metadata.json').write_text(json.dumps(meta,indent=2));shutil.copyfile(stage/'kernel-metadata.json',code/'kernel-metadata.json')
(code/'runner_sha256.json').write_text(json.dumps(dict(sha256=hashlib.sha256((stage/'run.py').read_bytes()).hexdigest()),indent=2))
print(stage)
