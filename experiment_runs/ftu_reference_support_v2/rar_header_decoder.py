def vint(b, p):
    v = 0
    shift = 0
    while True:
        x = b[p]
        p += 1
        v |= (x & 127) << shift
        if not x & 128:
            return (v, p)
        shift += 7
        assert shift < 70

def header(b, p=0):
    hs, q = vint(b, p + 4)
    end = q + hs
    typ, q = vint(b, q)
    flags, q = vint(b, q)
    extra = 0
    data = 0
    if flags & 1:
        extra, q = vint(b, q)
    if flags & 2:
        data, q = vint(b, q)
    assert zlib.crc32(b[p + 4:end]) & 4294967295 == struct.unpack_from('<I', b, p)[0]
    result = dict(type=typ, flags=flags, header_size=end - p, extra_size=extra, data_size=data)
    if typ == 1:
        af, q = vint(b, q)
        result['archive_flags'] = af
        if extra:
            result['extra_hex'] = b[end - extra:end].hex()
    if typ in [2, 3]:
        ff, q = vint(b, q)
        unp, q = vint(b, q)
        attrs, q = vint(b, q)
        if ff & 2:
            q += 4
        if ff & 4:
            q += 4
        compression, q = vint(b, q)
        host, q = vint(b, q)
        nl, q = vint(b, q)
        name = b[q:q + nl].decode('utf8')
        result.update(file_flags=ff, unpacked_size=unp, name=name, compression=compression, solid=bool(compression & 64), method=compression >> 7 & 7)
    return result