"""Bounded metadata-only acquisition probe; never reads radar/ECG signal arrays."""
from pathlib import Path
import hashlib
import json
import time
import urllib.request
import io
import zipfile


class RemoteArchive(io.RawIOBase):
    """Read ZIP metadata with bounded HTTP ranges; refuses whole-archive replies."""
    def __init__(self, url, size):
        self.url, self.size, self.pos, self.bytes_read = url, size, 0, 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.size)
        return self.pos

    def read(self, length=-1):
        length = self.size-self.pos if length < 0 else min(length, self.size-self.pos)
        if length == 0: return b''
        if length > 5_000_000 or self.bytes_read + length > 16_000_000:
            raise ValueError('ZIP metadata network budget exceeded')
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{self.pos+length-1}'})
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status != 206:
                raise ValueError('Range unsupported; whole archive download deliberately refused')
            data = response.read(length+1)
        if len(data) != length: raise ValueError('Incorrect byte range length')
        self.pos += length
        self.bytes_read += length
        return data


def main():
    started = time.perf_counter()
    root = Path('/kaggle/working/sync_readiness')
    root.mkdir(parents=True, exist_ok=True)
    results = []
    urls = {
        'clinical_article': 'https://api.figshare.com/v2/articles/12186516',
        'clinical_code_tree': 'https://gitlab.com/api/v4/projects/sven_schellenberger%2Fscidata_phase1/repository/tree?recursive=true&per_page=100',
    }
    for name, url in urls.items():
        entry = {'name': name, 'url': url}
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Radar-reference-readiness/1.0'})
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read(5_000_001)
            if len(content) > 5_000_000:
                raise ValueError('Metadata response exceeded fixed 5 MB bound')
            parsed = json.loads(content)
            (root / (name + '.json')).write_bytes(content)
            entry.update(ok=True, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            if name == 'clinical_article':
                files = [{k: f.get(k) for k in ['id', 'name', 'size', 'download_url', 'computed_md5']} for f in parsed.get('files', [])]
                (root / 'file_inventory.json').write_text(json.dumps(files, indent=2))
                print('OFFICIAL FILE INVENTORY', json.dumps(files), flush=True)
                for item in files:
                    if item['name'] == 'additional_data.xlsx':
                        with urllib.request.urlopen(item['download_url'], timeout=30) as response:
                            data = response.read(1_000_001)
                        if len(data) != item['size']: raise ValueError('Auxiliary metadata size mismatch')
                        (root/item['name']).write_bytes(data)
                    elif item['name'].startswith('datasets_subject_01'):
                        try:
                            remote = RemoteArchive(item['download_url'], item['size'])
                            with zipfile.ZipFile(remote) as archive:
                                inventory = [{'name': x.filename, 'size': x.file_size, 'compressed_size': x.compress_size,
                                              'crc32': x.CRC} for x in archive.infolist()]
                            (root/'archive_members.json').write_text(json.dumps(inventory, indent=2))
                            print('ZIP METADATA BYTES', remote.bytes_read, flush=True)
                        except Exception as exc:
                            (root/'archive_metadata_error.json').write_text(json.dumps({'error': repr(exc)}))
                    
                for source in ['README.md', 'TechnicalValidation.m', 'plotData.m', 'utils/getInterventions.m', 'utils/rpeak.m']:
                    code_url = 'https://gitlab.com/api/v4/projects/sven_schellenberger%2Fscidata_phase1/repository/files/'+source.replace('/', '%2F')+'/raw?ref=master'
                    try:
                        with urllib.request.urlopen(code_url, timeout=30) as response: data = response.read(1_000_001)
                        if len(data) > 1_000_000: raise ValueError('Source too large')
                        (root/source.replace('/', '__')).write_bytes(data)
                    except Exception as exc: print('SOURCE ERROR', source, repr(exc), flush=True)
        except Exception as exc:
            entry.update(ok=False, error=repr(exc))
        results.append(entry)
        print(json.dumps(entry), flush=True)
    (root / 'execution.json').write_text(json.dumps({
        'elapsed_seconds': time.perf_counter() - started,
        'metadata_only': True, 'radar_or_ecg_arrays_read': False,
        'results': results,
    }, indent=2))


if __name__ == '__main__':
    main()
