"""Archive exact submitted source and verify publication bytes."""
import ast
import base64
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

root=Path('experiment_runs/ftu_record_learnability')
tree=ast.parse(Path('tmp/ftu_record_upload/run.py').read_text(encoding='utf-8'))
payload=next(n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
             and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
    hashes=json.loads(z.read('source_manifest.json'))['files']
    for name,digest in hashes.items():
        data=z.read(name);assert hashlib.sha256(data).hexdigest()==digest
        path=root/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
for name in ['prepare_ftu_record_protocol.py','summarize_ftu_record.py','archive_ftu_record.py','kaggle/prepare_ftu_record.py']:
    path=root/'sources'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(Path(name).read_bytes())
(root/'kernel-metadata.json').write_bytes(Path('tmp/ftu_record_upload/kernel-metadata.json').read_bytes())
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
          for p in root.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
(root/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('archived files',len(manifest))
