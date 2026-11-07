"""Package only frozen code and protocols for one Kaggle CPU run."""
from pathlib import Path
import hashlib
import json

ROOT = Path('experiment_runs/peak_subject_validation')
sources = json.loads((ROOT/'source_hashes.json').read_text())
bundle = {}
for name, expected in sources.items():
    content = Path(name).read_bytes()
    assert hashlib.sha256(content).hexdigest() == expected, name
    bundle[Path(name).name] = content.decode('utf-8')
# protocol.json is already bundled byte-for-byte above. Reading it again in
# text mode would normalize CRLF and break the frozen input digest on Linux.
bundle['validation_protocol.json'] = (ROOT/'protocol.json').read_text(encoding='utf-8')
stage = Path('tmp/peak_subject_validation_upload')
stage.mkdir(exist_ok=True)
script = 'from pathlib import Path\nimport json,subprocess,sys,hashlib\n'
script += 'payload=json.loads('+repr(json.dumps(bundle))+')\n'
script += "root=Path('/kaggle/working/frozen_validation_code'); root.mkdir(exist_ok=True)\n"
script += "for name,content in payload.items(): (root/name).write_text(content,encoding='utf-8')\n"
script += 'expected='+repr({Path(name).name: expected for name, expected in sources.items()})+'\n'
script += "for name,digest in expected.items(): assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name\n"
script += "subprocess.run([sys.executable,'-m','pip','install','--quiet','neurokit2==0.2.12'],check=True)\n"
script += "subprocess.run([sys.executable,str(root/'run_peak_subject_validation.py')],check=True)\n"
(stage/'launch.py').write_text(script, encoding='utf-8')
(stage/'kernel-metadata.json').write_text(json.dumps(dict(id='goldfish9901/frozen-peak-subject-validation',
    title='Frozen Peak Subject Validation', code_file='launch.py', language='python', kernel_type='script',
    is_private=True, enable_gpu=False, enable_internet=True, dataset_sources=[], competition_sources=[], kernel_sources=[]), indent=2))
print(stage)
