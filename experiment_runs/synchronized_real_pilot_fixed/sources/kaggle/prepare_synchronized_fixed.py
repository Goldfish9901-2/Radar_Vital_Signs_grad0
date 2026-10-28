"""Package the frozen fix, preserving all original acquisition/scoring code."""
from pathlib import Path
import hashlib
import json

root = Path('experiment_runs/synchronized_real_pilot_fixed')
repair = json.loads((root/'repair_protocol.json').read_text())
bundle = {name: Path(name).read_text(encoding='utf-8') for name in repair['source_hashes']}
for name, digest in repair['source_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
bundle['protocol.json'] = Path('experiment_runs/synchronized_real_pilot/protocol.json').read_text()
bundle['repair_protocol.json'] = (root/'repair_protocol.json').read_text()
stage = Path('tmp/sync_fixed_upload'); stage.mkdir(exist_ok=True)
script = 'from pathlib import Path\nimport json,hashlib,subprocess,sys\n'
script += 'payload=json.loads('+repr(json.dumps(bundle))+')\n'
script += "root=Path('/kaggle/working/frozen_fixed_code'); root.mkdir(exist_ok=True)\n"
script += "for name,text in payload.items(): (root/name).write_text(text,encoding='utf-8')\n"
script += "repair=json.loads((root/'repair_protocol.json').read_text())\n"
script += "for name,digest in repair['source_hashes'].items(): assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest\n"
script += "subprocess.run([sys.executable,'-m','pip','install','--quiet','neurokit2==0.2.12'],check=True)\n"
script += "subprocess.run([sys.executable,str(root/'run_synchronized_real_pilot_fixed.py')],check=True)\n"
(stage/'launch.py').write_text(script, encoding='utf-8')
(stage/'kernel-metadata.json').write_text(json.dumps({'id': 'goldfish9901/synchronized-phase-calibration-fixed',
    'title': 'Synchronized Phase Calibration Fixed', 'code_file': 'launch.py', 'language': 'python', 'kernel_type': 'script',
    'is_private': True, 'enable_gpu': False, 'enable_internet': True, 'dataset_sources': [], 'competition_sources': [], 'kernel_sources': []}, indent=2))
print(stage)
