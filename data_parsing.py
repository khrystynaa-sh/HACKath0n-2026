"""
Telemetry Parser for ArduPilot binary log files (.BIN).

Extends the base binary reader with:
  - Extraction of GPS and IMU sensor messages
  - Automatic sampling frequency detection
  - Unit resolution from UNIT / MULT / FMTU metadata tables
  - Structured pandas DataFrames ready for analysis
"""

import struct
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# ── ArduPilot binary format character table ──────────────────────────────────

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


# ── Low-level binary reading ─────────────────────────────────────────────────

def _unpack_payload(fmt_str: str, payload: bytes) -> list:
    """Unpack a binary payload according to an ArduPilot format string."""
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


def read_bin_file(filename: str) -> Dict[str, list]:
    """
    Read an ArduPilot .BIN log and return all decoded messages grouped by type.

    Returns
    -------
    dict  :  { "GPS": [row, ...], "IMU": [row, ...], ... }
             Each row is a dict mapping column names to values.
    """
    with open(filename, "rb") as f:
        raw = f.read()

    # ── Pass 1: collect FMT (type 0x80 = 128) definitions ──
    formats: Dict[int, dict] = {}
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

    # ── Pass 2: decode every message ──
    decoded: Dict[str, list] = {}
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


# ── Unit resolution ──────────────────────────────────────────────────────────

# ArduPilot encodes unit IDs as single ASCII characters.  The UNIT table maps
# char-id → label (e.g.  109 → "m",  110 → "m/s").  The MULT table maps
# char-id → numeric multiplier.  FMTU links each message type to its per-column
# unit-id and mult-id strings.

def _build_unit_tables(decoded: Dict[str, list]) -> Tuple[dict, dict]:
    """Build  unit_id→label  and  mult_id→multiplier  lookup dicts."""
    units: Dict[int, str] = {}
    mults: Dict[int, float] = {}

    for row in decoded.get('UNIT', []):
        units[row['Id']] = row['Label']
    for row in decoded.get('MULT', []):
        mults[row['Id']] = row['Mult']

    return units, mults


def _build_fmtu_index(decoded: Dict[str, list]) -> Dict[int, dict]:
    """Map  message-type-id → { 'UnitIds': str, 'MultIds': str }."""
    idx: Dict[int, dict] = {}
    for row in decoded.get('FMTU', []):
        idx[row['FmtType']] = row
    return idx


def resolve_units(decoded: Dict[str, list], msg_name: str) -> Dict[str, str]:
    """
    Return { column_name: unit_label } for a given message type.

    Example
    -------
    >>> resolve_units(decoded, "GPS")
    {'TimeUS': 'us', 'Lat': 'deglatitude', 'Lng': 'deglongitude',
     'Alt': 'm', 'Spd': 'm/s', ...}
    """
    units_tbl, _ = _build_unit_tables(decoded)
    fmtu_idx = _build_fmtu_index(decoded)

    # Find the numeric type-id for msg_name from the FMT rows
    fmt_type_id = None
    for row in decoded.get('FMT', []):
        if row.get('Name') == msg_name:
            fmt_type_id = row.get('Type')
            break
    if fmt_type_id is None or fmt_type_id not in fmtu_idx:
        return {}

    fmtu = fmtu_idx[fmt_type_id]
    unit_ids_str = fmtu.get('UnitIds', '')

    # The FMT columns list (without the implicit first 'Type' column that
    # isn't stored in FMTU) corresponds 1-to-1 with FMTU unit-id chars.
    # First column is always TimeUS with unit 's' after scaling.
    fmt_entry = None
    for row in decoded.get('FMT', []):
        if row.get('Name') == msg_name:
            fmt_entry = row
            break
    if fmt_entry is None:
        return {}

    columns = fmt_entry['Columns'].split(',') if isinstance(fmt_entry['Columns'], str) else fmt_entry['Columns']

    result: Dict[str, str] = {}
    for col_idx, col_name in enumerate(columns):
        if col_idx < len(unit_ids_str):
            uid = ord(unit_ids_str[col_idx])
            label = units_tbl.get(uid, '')
            if label and label != 'UNKNOWN':
                result[col_name] = label
            elif unit_ids_str[col_idx] == '#':
                result[col_name] = 'instance'
    return result


# ── Sampling frequency detection ─────────────────────────────────────────────

def detect_sampling_rate(messages: List[dict], instance: Optional[int] = None) -> dict:
    """
    Compute sampling frequency from TimeUS timestamps.

    Parameters
    ----------
    messages  : list of row-dicts (must contain 'TimeUS')
    instance  : if the message has an 'I' (instance) column, filter to this
                instance first (e.g.  IMU instance 0 vs 1).

    Returns
    -------
    dict with keys:
        mean_dt_us   – mean time-step in microseconds
        std_dt_us    – std-dev of time-step
        freq_hz      – sampling frequency in Hz
        n_samples    – number of samples used
    """
    rows = messages
    if instance is not None and rows and 'I' in rows[0]:
        rows = [r for r in rows if r.get('I') == instance]

    if len(rows) < 2:
        return {'mean_dt_us': None, 'std_dt_us': None, 'freq_hz': None, 'n_samples': len(rows)}

    times = np.array([r['TimeUS'] for r in rows], dtype=np.float64)
    diffs = np.diff(times)
    diffs = diffs[diffs > 0]  # remove zero-diffs (same-timestamp twin instances)

    mean_dt = float(np.mean(diffs))
    std_dt = float(np.std(diffs))
    freq = 1e6 / mean_dt if mean_dt > 0 else 0.0

    return {
        'mean_dt_us': round(mean_dt, 2),
        'std_dt_us': round(std_dt, 2),
        'freq_hz': round(freq, 2),
        'n_samples': len(rows),
    }


# ── DataFrame builders ───────────────────────────────────────────────────────

def extract_gps(decoded: Dict[str, list], instance: int = 0) -> pd.DataFrame:
    """
    Extract GPS messages into a DataFrame with a proper datetime-like index.

    Columns kept: TimeUS, Status, NSats, HDop, Lat, Lng, Alt, Spd, GCrs, VZ, Yaw
    An extra column `time_s` gives seconds since log start.
    """
    rows = decoded.get('GPS', [])
    if not rows:
        return pd.DataFrame()

    rows = [r for r in rows if r.get('I', 0) == instance]
    df = pd.DataFrame(rows)

    cols_keep = ['TimeUS', 'Status', 'NSats', 'HDop', 'Lat', 'Lng',
                 'Alt', 'Spd', 'GCrs', 'VZ', 'Yaw']
    cols_keep = [c for c in cols_keep if c in df.columns]
    df = df[cols_keep].copy()

    df['time_s'] = (df['TimeUS'] - df['TimeUS'].iloc[0]) / 1e6
    return df.reset_index(drop=True)


def extract_imu(decoded: Dict[str, list], instance: int = 0) -> pd.DataFrame:
    """
    Extract IMU messages into a DataFrame.

    Columns kept: TimeUS, GyrX/Y/Z (rad/s), AccX/Y/Z (m/s²), T (°C)
    An extra column `time_s` gives seconds since log start.
    """
    rows = decoded.get('IMU', [])
    if not rows:
        return pd.DataFrame()

    rows = [r for r in rows if r.get('I', 0) == instance]
    df = pd.DataFrame(rows)

    cols_keep = ['TimeUS', 'GyrX', 'GyrY', 'GyrZ',
                 'AccX', 'AccY', 'AccZ', 'T', 'GHz', 'AHz']
    cols_keep = [c for c in cols_keep if c in df.columns]
    df = df[cols_keep].copy()

    df['time_s'] = (df['TimeUS'] - df['TimeUS'].iloc[0]) / 1e6
    return df.reset_index(drop=True)


def extract_baro(decoded: Dict[str, list], instance: int = 0) -> pd.DataFrame:
    """Extract barometer messages into a DataFrame."""
    rows = decoded.get('BARO', [])
    if not rows:
        return pd.DataFrame()
    rows = [r for r in rows if r.get('I', 0) == instance]
    df = pd.DataFrame(rows)
    cols_keep = ['TimeUS', 'Alt', 'Press', 'Temp', 'CRt']
    cols_keep = [c for c in cols_keep if c in df.columns]
    df = df[cols_keep].copy()
    df['time_s'] = (df['TimeUS'] - df['TimeUS'].iloc[0]) / 1e6
    return df.reset_index(drop=True)


def extract_att(decoded: Dict[str, list]) -> pd.DataFrame:
    """Extract attitude messages into a DataFrame."""
    rows = decoded.get('ATT', [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['time_s'] = (df['TimeUS'] - df['TimeUS'].iloc[0]) / 1e6
    return df.reset_index(drop=True)


def to_dataframe(decoded: Dict[str, list], msg_name: str) -> pd.DataFrame:
    """
    Generic converter: turn any message type into a pandas DataFrame.

    Parameters
    ----------
    decoded   : output of read_bin_file()
    msg_name  : e.g. "GPS", "IMU", "BARO", "ATT", "VIBE", ...

    Returns
    -------
    pd.DataFrame with a `time_s` column (seconds since first message of that type).
    """
    rows = decoded.get(msg_name, [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if 'TimeUS' in df.columns:
        df['time_s'] = (df['TimeUS'] - df['TimeUS'].iloc[0]) / 1e6
    return df.reset_index(drop=True)


# ── High-level convenience function ──────────────────────────────────────────

def parse_telemetry(filename: str) -> dict:
    """
    One-call entry point: read a .BIN file and return structured data.

    Returns
    -------
    dict with keys:
        raw          – full decoded dict (all message types)
        gps          – pd.DataFrame of GPS data
        imu          – pd.DataFrame of IMU data
        baro         – pd.DataFrame of barometer data
        att          – pd.DataFrame of attitude data
        sampling     – dict of detected sampling rates per sensor
        units        – dict of { msg_type: { col: unit_label } }
        msg_types    – list of all message type names found
    """
    decoded = read_bin_file(filename)

    gps_df = extract_gps(decoded)
    imu_df = extract_imu(decoded)
    baro_df = extract_baro(decoded)
    att_df = extract_att(decoded)

    # Sampling rates
    sampling = {}
    for name in ['GPS', 'IMU', 'BARO', 'ATT', 'VIBE']:
        if name in decoded and decoded[name]:
            sampling[name] = detect_sampling_rate(decoded[name], instance=0)

    # Units
    units = {}
    for name in decoded:
        u = resolve_units(decoded, name)
        if u:
            units[name] = u

    return {
        'raw': decoded,
        'gps': gps_df,
        'imu': imu_df,
        'baro': baro_df,
        'att': att_df,
        'sampling': sampling,
        'units': units,
        'msg_types': list(decoded.keys()),
    }


# ── CLI demo ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Parse ArduPilot .BIN telemetry logs')
    parser.add_argument('filename')
    args = parser.parse_args()

    fname = args.filename
    print(f"Parsing {fname} ...")

    result = parse_telemetry(fname)

    print(f"\n{'='*60}")
    print(f"Message types found ({len(result['msg_types'])}): {result['msg_types']}")

    print(f"\n{'='*60}")
    print("SAMPLING RATES:")
    for sensor, info in result['sampling'].items():
        print(f"  {sensor:6s}  →  {info['freq_hz']:8.2f} Hz   "
              f"(dt={info['mean_dt_us']:.0f} ± {info['std_dt_us']:.0f} μs, "
              f"n={info['n_samples']})")

    print(f"\n{'='*60}")
    print("UNITS:")
    for msg, cols in result['units'].items():
        print(f"  {msg}: {cols}")

    print(f"\n{'='*60}")
    print(f"GPS DataFrame: {result['gps'].shape}")
    if not result['gps'].empty:
        print(result['gps'].head())

    print(f"\nIMU DataFrame: {result['imu'].shape}")
    if not result['imu'].empty:
        print(result['imu'].head())

    print(f"\nBARO DataFrame: {result['baro'].shape}")
    if not result['baro'].empty:
        print(result['baro'].head())
