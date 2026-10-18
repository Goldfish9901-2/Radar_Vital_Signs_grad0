"""Recover exactly one non-solid RAR5 member by official HTTP Range + libarchive."""
import ctypes,ctypes.util,hashlib,json,struct,time,urllib.request,zlib
from pathlib import Path
def recover(code,work):
    start=time.perf_counter();info=json.loads((code/'official_recovery/p03_official_member.json').read_text())
    assert not info['solid'] and info['unpacked_size']==614400000
    header=(code/'official_recovery/p03_member_header.bin').read_bytes();assert len(header)==info['header_size']
    def block(body):
        assert len(body)<128
        data=bytes([len(body)])+body
        return struct.pack('<I',zlib.crc32(data)&0xffffffff)+data
    archive=work/'p03_member.rar';work.mkdir(exist_ok=True,parents=True)
    first=info['data_offset'];last=first+info['data_size']-1
    req=urllib.request.Request(info['download_url'],headers={'Range':f'bytes={first}-{last}','User-Agent':'Mozilla/5.0 research diagnostic','Accept-Encoding':'identity'})
    with urllib.request.urlopen(req,timeout=120) as r,archive.open('wb') as f:
        assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {first}-{last}/')
        f.write(b'Rar!\x1a\x07\x01\x00'+block(b'\x01\x00\x00')+header);remaining=info['data_size'];h=hashlib.sha256()
        while remaining:
            data=r.read(min(4*1024*1024,remaining));assert data;f.write(data);h.update(data);remaining-=len(data)
        f.write(block(b'\x05\x04\x00'))
    lib=ctypes.CDLL(ctypes.util.find_library('archive') or 'libarchive.so.13')
    lib.archive_read_new.restype=ctypes.c_void_p
    for name in ['archive_read_support_filter_all','archive_read_support_format_all','archive_read_free']:
        getattr(lib,name).argtypes=[ctypes.c_void_p]
    lib.archive_read_open_filename.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_size_t]
    lib.archive_read_next_header.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p)]
    lib.archive_read_data.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];lib.archive_read_data.restype=ctypes.c_ssize_t
    lib.archive_entry_pathname.argtypes=[ctypes.c_void_p];lib.archive_entry_pathname.restype=ctypes.c_char_p
    lib.archive_error_string.argtypes=[ctypes.c_void_p];lib.archive_error_string.restype=ctypes.c_char_p
    obj=lib.archive_read_new();lib.archive_read_support_filter_all(obj);lib.archive_read_support_format_all(obj)
    assert lib.archive_read_open_filename(obj,str(archive).encode(),1024*1024)==0
    entry=ctypes.c_void_p();status=lib.archive_read_next_header(obj,ctypes.byref(entry));assert status==0,lib.archive_error_string(obj)
    assert lib.archive_entry_pathname(entry).decode()==info['name']
    target=work/'P3_1_Distance_Scenario/1. Distance Scenario/40 cm/3/data_Raw_0.bin';target.parent.mkdir(parents=True,exist_ok=True)
    buffer=ctypes.create_string_buffer(4*1024*1024);rawh=hashlib.sha256();total=0
    with target.open('wb') as f:
        while True:
            n=lib.archive_read_data(obj,buffer,len(buffer));assert n>=0,lib.archive_error_string(obj)
            if not n:break
            data=buffer.raw[:n];f.write(data);rawh.update(data);total+=n
    lib.archive_read_free(obj);assert total==info['unpacked_size']
    result=dict(official_member=info['name'],member_offset=info['offset'],compressed_bytes=info['data_size'],unpacked_bytes=total,
        compressed_payload_sha256=h.hexdigest(),raw_sha256=rawh.hexdigest(),elapsed_sec=time.perf_counter()-start,
        non_solid=True,libarchive_crc_checked=True,selection_unchanged=True)
    (work/'recovery.json').write_text(json.dumps(result,indent=2));print('Official p03 restored',json.dumps(result),flush=True)
    return work
