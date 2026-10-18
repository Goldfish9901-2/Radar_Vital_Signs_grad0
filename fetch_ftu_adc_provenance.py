"""Download only the provenance ZIP/log from the fixed Kaggle CPU run."""
import hashlib,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,r'C:\Users\goldfish\AppData\Roaming\uv\tools\kaggle\Lib\site-packages')
from kaggle import api
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
import requests

root=Path('experiment_runs/ftu_adc_phase');out=root/'provenance';out.mkdir(exist_ok=True)
with api.build_kaggle_client() as client:
    req=ApiListKernelSessionOutputRequest();req.user_name='goldfish9901';req.kernel_slug='ftu-adc-phase-provenance'
    api._set_paging(req,200,None);result=client.kernels.kernels_api_client.list_kernel_session_output(req)
if result.log:(root/'kaggle_run.txt').write_text(result.log,encoding='utf-8')
files=[f for f in result.files or [] if f.file_name=='000_ftu_adc_provenance.zip']
if not files:
    print('Final provenance ZIP not ready; log saved if available.');sys.exit(0)
dest=Path('tmp/ftu_adc_provenance.zip')
with requests.get(files[0].url,stream=True,timeout=120) as response:
    response.raise_for_status()
    with dest.open('wb') as stream:
        for block in response.iter_content(1024*1024):stream.write(block)
with zipfile.ZipFile(dest) as archive:
    for name in archive.namelist():
        path=Path(name);assert not path.is_absolute() and '..' not in path.parts
        if name.endswith('/'):continue
        target=out/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(name))
print((out/'provenance.json').read_text())
