"""Read public 4TU file metadata only; never download the radar archive."""
import json
import time
import urllib.request
from pathlib import Path

ROOT=Path('experiment_runs/ftu_reference_support_v2')
url='https://data.4tu.nl/v2/articles/48acba04-96bc-4131-b52f-9e18458ad92b/files'
begin=time.perf_counter()
try:
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 reference-support audit'})
    with urllib.request.urlopen(req,timeout=25) as r:
        body=r.read(2_000_000);status=r.status
    files=json.loads(body);(ROOT/'official_files.json').write_bytes(body)
    result=dict(url=url,status=status,files=[dict(name=x['name'],size=x['size'],uuid=x['uuid']) for x in files],
        elapsed_seconds=time.perf_counter()-begin,
        limitation='Top-level file list only; RAR internals not exhaustively searched. No claim that longer HR or synchronization data are unpublished.')
except Exception as e:
    result=dict(url=url,error=str(e),elapsed_seconds=time.perf_counter()-begin)
(ROOT/'official_release_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
