"""Archive submitted sources and final evidence hashes."""
import ast,base64,hashlib,io,json,zipfile
from pathlib import Path
root=Path('experiment_runs/ftu_objectives')
tree=ast.parse(Path('tmp/ftu_objectives_upload/run.py').read_text(encoding='utf-8'))
payload=next(n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
             and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
    manifest=json.loads(z.read('source_manifest.json'))
    for name,h in manifest['files'].items():
        data=z.read(name);assert hashlib.sha256(data).hexdigest()==h
        p=root/'sources'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
for name in ['validate_ftu_objectives.py','summarize_ftu_objectives.py','archive_ftu_objectives.py','kaggle/prepare_ftu_objectives.py']:
    p=root/'sources'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(name).read_bytes())
(root/'kernel-metadata.json').write_bytes(Path('tmp/ftu_objectives_upload/kernel-metadata.json').read_bytes())
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
          for p in root.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
(root/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Archived',len(manifest),'files')
