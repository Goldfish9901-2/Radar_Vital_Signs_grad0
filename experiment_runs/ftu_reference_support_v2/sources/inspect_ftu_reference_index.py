"""Offline inspection of the previously cached official RAR QuickOpen headers.

QuickOpen is a header cache; absence here is not proof of absence in the RAR.
"""
import ast
import hashlib
import json
import struct
import time
import zlib
from pathlib import Path

ROOT=Path('experiment_runs/ftu_reference_support_v2')
OLD=Path('experiment_runs/ftu_adc_controlled/official_recovery')
begin=time.perf_counter()
# Parse only pure varint/header functions; never execute the old network indexer.
decoder=ROOT/'rar_header_decoder.py'
source=(decoder if decoder.exists() else Path('tmp/index_ftu_rar.py')).read_text()
tree=ast.parse(source)
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['vint','header']]
env=dict(struct=struct,zlib=zlib)
exec(compile(ast.Module(body=functions,type_ignores=[]),'<archived RAR header decoder>','exec'),env)
(ROOT/'rar_header_decoder.py').write_text('\n\n'.join(ast.unparse(n) for n in functions))
b=(OLD/'quickopen.bin').read_bytes();info=json.loads((OLD/'quickopen_header.json').read_text())
p=info['header']['header_size'];end=p+info['header']['data_size'];rows=[]
while p<end:
    size,q=env['vint'](b,p+4);finish=q+size
    assert finish<=end and zlib.crc32(b[p+4:finish])&0xffffffff==struct.unpack_from('<I',b,p)[0]
    flags,q=env['vint'](b,q);offset,q=env['vint'](b,q);hsize,q=env['vint'](b,q)
    h=env['header'](b[q:q+hsize]);h['offset']=info['offset']-offset;rows.append(h);p=finish
assert p==end
(ROOT/'cached_official_headers.json').write_text(json.dumps(rows,indent=2))
matches=[r for r in rows if any(s in r.get('name','').lower() for s in ['ref','polar','ecg','sync','timestamp','rr_interval','ibi'])]
result=dict(elapsed_seconds=time.perf_counter()-begin,cached_headers=len(rows),
    quickopen_sha256=hashlib.sha256(b).hexdigest(),matching_names=[r.get('name') for r in matches],
    limitation='Entire locally cached QuickOpen block parsed with CRC checks; cache need not contain every RAR member. No payload read; no claim of absent unpublished/full references.')
(ROOT/'cached_index_inspection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
