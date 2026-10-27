"""Create a private CPU runner with frozen source/protocol hashes."""
from pathlib import Path
import hashlib
import json
import shutil

stage = Path('tmp/sync_real_upload')
stage.mkdir(parents=True, exist_ok=True)
for source in [Path('run_synchronized_real_pilot.py'), Path('experiment_runs/synchronized_real_pilot/protocol.json')]:
    shutil.copyfile(source, stage/source.name)
protocol = json.loads((stage/'protocol.json').read_text())
assert hashlib.sha256((stage/'run_synchronized_real_pilot.py').read_bytes()).hexdigest() == protocol['runner_sha256']
(stage/'launch.py').write_text('''from pathlib import Path
import hashlib, json, subprocess, sys
source = Path(__file__).parent
protocol = json.loads((source/'protocol.json').read_text())
runner = source/'run_synchronized_real_pilot.py'
assert hashlib.sha256(runner.read_bytes()).hexdigest() == protocol['runner_sha256']
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'neurokit2==0.2.12'], check=True)
subprocess.run([sys.executable, str(runner)], check=True)
''', encoding='utf-8')
metadata = {'id': 'goldfish9901/synchronized-real-phase-pilot', 'title': 'Synchronized Real Phase Pilot',
            'code_file': 'launch.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
            'enable_gpu': False, 'enable_internet': True, 'dataset_sources': [], 'competition_sources': [],
            'kernel_sources': []}
# Kaggle scripts do not upload sibling files. Embed both immutable inputs in launcher.
bundle = {p.name: p.read_text(encoding='utf-8') for p in [stage/'protocol.json', stage/'run_synchronized_real_pilot.py']}
prefix = 'from pathlib import Path\nimport json\n'
prefix += 'payload = json.loads('+repr(json.dumps(bundle))+')\n'
prefix += "source = Path('/kaggle/working/frozen_pilot_code'); source.mkdir(exist_ok=True)\n"
prefix += "for name, value in payload.items(): (source/name).write_text(value, encoding='utf-8')\n"
prefix += "import runpy\nrunpy.run_path(str(source/'launch_inner.py'), run_name='__main__')\n"
inner = (stage/'launch.py').read_text()
bundle['launch_inner.py'] = inner
prefix = prefix.replace('payload = json.loads('+repr(json.dumps({k: v for k, v in bundle.items() if k != 'launch_inner.py'}))+')',
                        'payload = json.loads('+repr(json.dumps(bundle))+')')
(stage/'launch.py').write_text(prefix, encoding='utf-8')
(stage/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
print(stage)
