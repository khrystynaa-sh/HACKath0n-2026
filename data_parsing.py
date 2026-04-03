import struct

FORMAT_CHARS = {
    'b': ('b', 1), 'B': ('B', 1),
    'h': ('h', 2), 'H': ('H', 2),
    'i': ('i', 4), 'I': ('I', 4),
    'f': ('f', 4), 'd': ('d', 8),
    'Q': ('Q', 8), 'q': ('q', 8),
    'n': ('4s', 4), 'N': ('16s', 16), 'Z': ('64s', 64),
    'c': ('h', 2),   # int16 * 0.01
    'C': ('H', 2),   # uint16 * 0.01
    'e': ('i', 4),   # int32 * 0.01
    'E': ('I', 4),   # uint32 * 0.01
    'L': ('i', 4),   # int32 (lat/lng * 1e-7)
    'M': ('B', 1),   # flight mode
    'a': ('32h', 64), # int16[32] array
}

def read_file(filename: str):
    with open(filename, "rb") as file:
        raw = file.read()

    # parse FMT messages
    formats = {}
    i = 0
    while i < len(raw) - 3:
        if raw[i] == 0xA3 and raw[i + 1] == 0x95 and raw[i + 2] == 0x80:
            fmt_type = raw[i + 3]
            fmt_len = raw[i + 4]
            fmt_name = raw[i + 5:i + 9].decode('ascii', errors='replace').rstrip('\x00')
            fmt_format = raw[i + 9:i + 25].decode('ascii', errors='replace').rstrip('\x00')
            fmt_columns = raw[i + 25:i + 89].decode('ascii', errors='replace').rstrip('\x00')
            formats[fmt_type] = {
                'name': fmt_name,
                'length': fmt_len,
                'format': fmt_format,
                'columns': fmt_columns.split(','),
            }
            i += 89
        else:
            i += 1

    # parse all messages
    decoded = {}
    i = 0
    while i < len(raw) - 2:
        if raw[i] == 0xA3 and raw[i + 1] == 0x95:
            msg_type = raw[i + 2]
            if msg_type in formats:
                fmt = formats[msg_type]
                msg_len = fmt['length']
                if i + msg_len > len(raw):
                    break
                payload = raw[i + 3:i + msg_len]
                try:
                    values = _unpack_payload(fmt['format'], payload)
                    row = dict(zip(fmt['columns'], values))
                    decoded.setdefault(fmt['name'], []).append(row)
                except Exception:
                    pass
                i += msg_len
            else:
                i += 1
        else:
            i += 1

    return decoded


def _unpack_payload(fmt_str: str, payload: bytes):
    values = []
    offset = 0
    for ch in fmt_str:
        if ch not in FORMAT_CHARS:
            break
        struct_fmt, size = FORMAT_CHARS[ch]
        if offset + size > len(payload):
            break
        raw = struct.unpack_from('<' + struct_fmt, payload, offset)
        if ch in ('n', 'N', 'Z'):
            values.append(raw[0].decode('ascii', errors='replace').rstrip('\x00'))
        elif ch == 'a':
            values.append(list(raw))
        elif ch in ('c', 'C', 'e', 'E'):
            values.append(raw[0] * 0.01)
        elif ch == 'L':
            values.append(raw[0] * 1e-7)
        else:
            values.append(raw[0])
        offset += size
    return values

# data = read_file("00000001.BIN")
# print(data[0])
# print(data.keys())

# print(data["IMU"])

# for point in data["GPS"][:5]:
#     print(point["Lat"], point["Lng"], point["Alt"])
