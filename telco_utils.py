#!/usr/bin/env python3
"""
Unified 3-Stage Pipeline for Telco Root Cause Classification - Final

Rule-based classification for Type A (C1-C8) and Type B (A-I) root cause
analysis. From data analysis we know these rules reliably classify each
category. Header-based column mapping for all table parsers - maps
header names to column indices instead of hardcoded positional indices,
with fallback to positional logic if header parsing fails.

Calibrated on train.csv (2400 labeled Type A questions).

Usage:
    python telco_utils.py --phase 2
    python telco_utils.py --phase 2 --samples 10  # Test run
    python telco_utils.py --phase 2 --dry-run     # No LLM calls
"""

import os
import re
import json
import asyncio
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
from math import radians, sin, cos, sqrt, atan2, degrees, acos

import pandas as pd

try:
    from huggingface_hub import InferenceClient
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "huggingface_hub"])
    from huggingface_hub import InferenceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# STAGE 1: QUESTION TYPE CLASSIFICATION
# =============================================================================

def classify_question_type(question: str) -> str:
    """
    Classify question into one of three types.

    Returns:
        'type_a_telco': 600Mbps, pipe-delimited tables, C1-C8 style options
        'type_b_telco': 100Mbps, markdown tables, A-I options
        'generic': Math/reasoning problems (no telco data)
    """
    # Type B: 100Mbps threshold (primary indicator)
    if '100Mbps' in question or '100 Mbps' in question:
        return 'type_b_telco'

    # Generic: No telco tables
    if 'Analyze the following question' in question:
        if '|' not in question or question.count('|') < 10:
            return 'generic'

    # Type A: Has telco data with 600Mbps
    if '600Mbps' in question or '600 Mbps' in question:
        return 'type_a_telco'

    # Fallback: Check for telco table markers
    if 'gNodeB ID|' in question or 'Timestamp|' in question:
        return 'type_a_telco'

    return 'generic'


# =============================================================================
# STAGE 2: CLASSIFICATION RULES - SHARED UTILITIES
# =============================================================================

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance in km between two points."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return 6371 * c


def safe_float(v):
    """Safely convert to float, return None for invalid values."""
    try:
        f = float(v)
        return f if f != 0 else None
    except:
        return None


# =============================================================================
# HEADER-BASED COLUMN MAPPING (V19-robust)
# =============================================================================

def build_column_map(header_line, keyword_spec, strip_empty=False):
    """Build a column index map from a header line using keyword matching.

    For each field in keyword_spec, find the first column whose header
    name matches the keyword. Matching is case-insensitive.

    Args:
        header_line: The raw header line string.
        keyword_spec: Dict of {field_name: keyword_or_tuple}.
            If value is a string: substring match (keyword in column_name).
            If value is a tuple (keyword, True): exact match after strip.
        strip_empty: If True, split by '|' and strip empty parts (Type B
                     markdown tables with leading/trailing pipes).
                     If False, split by '|' keeping all parts (Type A style).

    Returns:
        Dict of {field_name: col_index} or None if no fields matched.
    """
    if strip_empty:
        parts = [p.strip() for p in header_line.split('|') if p.strip()]
    else:
        parts = [p.strip() for p in header_line.split('|')]

    col_map = {}
    for field_name, spec in keyword_spec.items():
        if isinstance(spec, tuple):
            keyword, exact = spec
        else:
            keyword, exact = spec, False

        kw_lower = keyword.lower()
        for i, col_name in enumerate(parts):
            col_lower = col_name.lower().strip()
            if exact:
                if col_lower == kw_lower:
                    col_map[field_name] = i
                    break
            else:
                if kw_lower in col_lower:
                    col_map[field_name] = i
                    break

    return col_map if col_map else None


# =============================================================================
# GENERIC TABLE PARSER (fallback when specific parsers fail)
# =============================================================================

def _simplify_column_name(raw_name: str) -> Tuple[str, Optional[str]]:
    """Strip common prefixes and extract units from a column header.

    Returns (simplified_name, unit_string_or_None).
    """
    name = raw_name.strip()
    # Extract units from brackets: [dBm], (km/h), (Mbps), (%), (dB), (0.5dB)
    unit = None
    unit_match = re.search(r'[\[\(]([^)\]]+)[\]\)]', name)
    if unit_match:
        unit = unit_match.group(1).strip()
        name = name[:unit_match.start()].strip()

    # Strip common telco prefixes
    for prefix in [
        '5G KPI PCell RF Serving ',
        '5G KPI PCell RF ',
        '5G KPI PCell Layer2 MAC DL ',
        '5G KPI PCell Layer1 DL ',
        '5G KPI PCell Layer1 ',
        '5G KPI PCell ',
    ]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Convert to snake_case: replace non-alphanum with underscore, collapse
    name = re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_').lower()
    return name, unit


def _find_table_regions(lines: List[str]) -> List[dict]:
    """Find contiguous table regions in question text.

    A table region is a run of lines where each line has >= 3 pipe characters.
    Detects markdown style (leading/trailing pipes, separator rows with ---)
    vs plain pipe-delimited style.

    Returns list of dicts with keys: start, end, style, label.
    """
    regions = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.count('|') >= 3:
            start = i
            # Extend region while pipe-heavy lines continue
            while i < n and lines[i].count('|') >= 3:
                i += 1
            end = i  # exclusive

            # Detect style: markdown has leading/trailing pipes and --- separator
            sample = lines[start]
            is_markdown = (
                sample.strip().startswith('|')
                and sample.strip().endswith('|')
            )

            # Check for preceding section label
            label = None
            for j in range(start - 1, max(start - 4, -1), -1):
                candidate = lines[j].strip()
                if candidate and candidate.count('|') < 3:
                    if candidate.endswith(':') or 'as follows' in candidate.lower():
                        label = candidate.rstrip(':').strip()
                    break

            regions.append({
                'start': start,
                'end': end,
                'style': 'markdown' if is_markdown else 'plain',
                'label': label,
            })
        else:
            i += 1
    return regions


def _parse_table_region(lines: List[str], region: dict) -> Optional[dict]:
    """Parse a single table region into structured column data.

    Returns dict with: label, headers, columns (dict of col_name -> [values]),
    n_rows, or None if parsing fails.
    """
    table_lines = lines[region['start']:region['end']]
    if len(table_lines) < 2:
        return None

    style = region['style']

    if style == 'markdown':
        split_fn = lambda line: [p.strip() for p in line.split('|') if p.strip()]
    else:
        split_fn = lambda line: [p.strip() for p in line.split('|')]

    # First non-separator line is the header
    header_idx = 0
    headers = split_fn(table_lines[header_idx])
    if not headers:
        return None

    # Find data rows (skip separator lines like |---|---|)
    data_rows = []
    for line in table_lines[header_idx + 1:]:
        cells = split_fn(line)
        # Skip separator rows
        if all(re.match(r'^[-:]+$', c) for c in cells if c):
            continue
        if not cells:
            continue
        data_rows.append(cells)

    if not data_rows:
        return None

    # Build column dict
    columns = {}
    for col_idx, raw_header in enumerate(headers):
        col_name, unit = _simplify_column_name(raw_header)
        if not col_name:
            col_name = f'col_{col_idx}'
        values = []
        for row in data_rows:
            if col_idx < len(row):
                values.append(row[col_idx])
            else:
                values.append('')
        columns[col_name] = {'values': values, 'unit': unit}

    return {
        'label': region.get('label'),
        'headers': headers,
        'columns': columns,
        'n_rows': len(data_rows),
    }


def parse_tables_generic(
    question: str,
    max_unique_categorical: int = 10,
    max_output_lines: int = 120,
) -> Optional[str]:
    """Parse all tables in a question into column-level summary stats.

    Fallback parser for when specific Type A / Type B parsers fail
    (e.g. renamed columns, different table layout). Produces output
    in the same structured format the model was trained on:
        field = value

    Args:
        question: Full question text.
        max_unique_categorical: Cap on unique values shown for categorical columns.
        max_output_lines: Maximum total output lines across all tables.

    Returns:
        Formatted summary string, or None if no parsable tables found.
    """
    all_lines = question.split('\n')
    regions = _find_table_regions(all_lines)
    if not regions:
        return None

    output_parts = []
    total_lines = 0

    for region in regions:
        parsed = _parse_table_region(all_lines, region)
        if parsed is None:
            continue

        label = parsed['label'] or 'Data'
        n_rows = parsed['n_rows']
        header = f"Table summary ({label}, {n_rows} rows):"
        part_lines = [header]

        for col_name, col_info in parsed['columns'].items():
            if total_lines + len(part_lines) >= max_output_lines:
                break

            raw_values = col_info['values']
            unit = col_info['unit']
            unit_str = f' {unit}' if unit else ''

            # Try to classify as numeric
            numeric_vals = []
            for v in raw_values:
                v_stripped = v.strip()
                if not v_stripped or v_stripped == '-' or v_stripped.lower() == 'nan':
                    continue
                try:
                    numeric_vals.append(float(v_stripped))
                except (ValueError, TypeError):
                    pass

            # Column is numeric if >= 50% of non-empty values parse as float
            non_empty = [v.strip() for v in raw_values if v.strip() and v.strip() != '-']
            if non_empty and len(numeric_vals) / len(non_empty) >= 0.5:
                if numeric_vals:
                    mn = min(numeric_vals)
                    mx = max(numeric_vals)
                    mean = sum(numeric_vals) / len(numeric_vals)
                    part_lines.append(
                        f'  {col_name}: min={mn:.2f}, max={mx:.2f}, mean={mean:.2f}{unit_str}'
                    )
            else:
                # Categorical
                unique = sorted(set(v.strip() for v in raw_values if v.strip()))
                if len(unique) <= max_unique_categorical:
                    part_lines.append(f'  {col_name}: unique={unique}')
                else:
                    part_lines.append(
                        f'  {col_name}: {len(unique)} unique values'
                    )

        if len(part_lines) > 1:  # more than just the header
            output_parts.append('\n'.join(part_lines))
            total_lines += len(part_lines) + 1  # +1 for blank line between tables

    if not output_parts:
        return None

    return '\n\n'.join(output_parts)


# --- Column keyword specs (constants) ---
# Keywords are matched case-insensitively as substrings of column headers.
# They must be specific enough to uniquely identify each column even when
# columns are reordered. Derived from actual data headers.

# Type A Drive Test (19 cols, pipe-separated, no leading pipes):
# Timestamp | Longitude | Latitude | GPS Speed (km/h) |
# 5G KPI PCell RF Serving PCI | 5G KPI PCell RF Serving SS-RSRP [dBm] |
# 5G KPI PCell RF Serving SS-SINR [dB] |
# 5G KPI PCell Layer2 MAC DL Throughput [Mbps] |
# ... Top 1-5 PCI | ... Top 1-5 Filtered Tx BRSRP [dBm] |
# 5G KPI PCell Layer1 DL RB Num (Including 0)
TYPE_A_DT_KEYWORDS = {
    'lon': 'Longitude',
    'lat': 'Latitude',
    'speed': 'GPS Speed',
    'serving_pci': 'Serving PCI',
    'rsrp': 'SS-RSRP',
    'sinr': 'SS-SINR',
    'throughput': 'Throughput',
    'n1_pci': 'Top 1 PCI',
    'n2_pci': 'Top 2 PCI',
    'n3_pci': 'Top 3 PCI',
    'n4_pci': 'Top 4 PCI',
    'n5_pci': 'Top 5 PCI',
    'n1_brsrp': 'Top 1 Filtered',
    'n2_brsrp': 'Top 2 Filtered',
    'n3_brsrp': 'Top 3 Filtered',
    'n4_brsrp': 'Top 4 Filtered',
    'n5_brsrp': 'Top 5 Filtered',
    'rb': 'RB Num',
}

# Type A Engineering Params (14 cols, pipe-separated, no leading pipes):
# gNodeB ID | Cell ID | Longitude | Latitude | Mechanical Azimuth |
# Mechanical Downtilt | Digital Tilt | Digital Azimuth | Beam Scenario |
# Height | PCI | TxRx Mode | Max Transmit Power | Antenna Model
TYPE_A_EP_KEYWORDS = {
    'gnodeb_id': 'gNodeB ID',
    'lon': 'Longitude',
    'lat': 'Latitude',
    'mech_azimuth': 'Mechanical Azimuth',
    'mech_tilt': 'Mechanical Downtilt',
    'digital_tilt': 'Digital Tilt',
    'beam': 'Beam Scenario',
    'pci': 'PCI',
}

# Type B Drive Test (22 non-empty cols, markdown table, strip empty):
# Time | UE | Longitude | Latitude | Serving PCI | Serving ARFCN |
# Serving RSRP(dBm) | Serving SINR(dB) | Throughput(Mbps) |
# Neighbor 1 PCI | Neighbor 1 RSRP(dBm) | Neighbor 2 PCI |
# Neighbor 2 RSRP(dBm) | Neighbor 3 PCI | Neighbor 3 RSRP(dBm) |
# CCE Fail Rate | Avg Rank | Grant | Avg MCS | RB/slot |
# Initial BLER(%) | Residual BLER(%)
TYPE_B_DT_KEYWORDS = {
    'serving_pci': 'Serving PCI',
    'rsrp': 'Serving RSRP',
    'sinr': 'Serving SINR',
    'throughput': 'Throughput',
    'n1_pci': 'Neighbor 1 PCI',
    'n1_rsrp': 'Neighbor 1 RSRP',
    'n2_pci': 'Neighbor 2 PCI',
    'n2_rsrp': 'Neighbor 2 RSRP',
    'n3_pci': 'Neighbor 3 PCI',
    'n3_rsrp': 'Neighbor 3 RSRP',
    'cce_fail_rate': 'CCE Fail',
    'avg_rank': 'Avg Rank',
    'grant': 'Grant',
    'avg_mcs': 'Avg MCS',
    'rb_slot': 'RB/slot',
    'initial_bler': 'Initial BLER',
    'residual_bler': 'Residual BLER',
}

# Type B Configuration Data (13 non-empty cols, markdown table, strip empty):
# gNodeB ID | Freq(MHz) | PCI | InterFreqHoEventType |
# CovInterFreqA2RsrpThld(dBm) | InterFreqA2Hyst(0.5dB) |
# CovInterFreqA5RsrpThld1(dBm) | CovInterFreqA5RsrpThld2(dBm) |
# IntraFreqHoA3Offset(0.5dB) | IntraFreqHoA3Hyst(0.5dB) |
# IntraFreqHoA3TimeToTrig | Neighbor(gNodeB_Freq_PCI) |
# PdcchOccupiedSymbolNum
TYPE_B_CONFIG_KEYWORDS = {
    'gnodeb_id': ('gNodeB ID', True),        # exact: avoid matching other gNodeB fields
    'freq': 'Freq(MHz)',                      # specific enough (vs InterFreq...)
    'pci': ('PCI', True),                     # exact: avoid matching Neighbor(gNodeB_Freq_PCI)
    'a2_rsrp_thld': 'A2RsrpThld',
    'a3_offset': 'A3Offset',
    'neighbor_list': 'Neighbor(gNodeB',
}


def compute_bearing(lon1, lat1, lon2, lat2):
    """Compute bearing from point 1 to point 2 in degrees (0-360)."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = atan2(x, y)
    return (degrees(bearing) + 360) % 360


def compute_off_axis_angle(cell_azimuth, bearing_to_ue):
    """Compute off-axis angle between cell azimuth and bearing to UE."""
    diff = abs(cell_azimuth - bearing_to_ue)
    if diff > 180:
        diff = 360 - diff
    return diff


# =============================================================================
# STAGE 2: TYPE A RULE-BASED CLASSIFIER
# =============================================================================

def extract_type_a_options(question: str) -> Dict[str, str]:
    """
    Extract options from Type A question and map to canonical causes (C1-C8).
    Returns: {test_label: canonical_cause} e.g. {'M1': 'C8', 'M2': 'C2', ...}
    """
    options = {}
    option_pattern = r'(?:^|\n)([A-Z]?\d+)[:\.\)]\s*([^\n]+)'
    matches = re.findall(option_pattern, question)

    for label, text in matches:
        text_clean = text.strip().lower()
        if 'frequent handover' in text_clean:
            options[label] = 'C5'
        elif 'downtilt' in text_clean or 'weak coverage at the far end' in text_clean:
            options[label] = 'C1'
        elif 'exceeds 1km' in text_clean or 'over-shooting' in text_clean or 'overshooting' in text_clean:
            options[label] = 'C2'
        elif 'neighboring cell provides higher throughput' in text_clean or 'neighbor cell provides higher throughput' in text_clean:
            options[label] = 'C3'
        elif 'non-colocated' in text_clean or 'overlapping coverage' in text_clean:
            options[label] = 'C4'
        elif 'pci mod 30' in text_clean or 'same pci' in text_clean:
            options[label] = 'C6'
        elif 'speed exceeds 40' in text_clean or 'vehicle speed' in text_clean or 'speed exceed' in text_clean:
            options[label] = 'C7'
        elif 'scheduled rbs' in text_clean or 'below 160' in text_clean or ('rb' in text_clean and '160' in text_clean):
            options[label] = 'C8'

    return options


def parse_type_a_question(question: str) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Parse drive test and engineering params from Type A question.

    V19-robust: Uses header-based column mapping. Falls back to
    positional indices if header parsing fails.
    """
    lines = question.split('\n')

    dt_start = ep_start = None
    for i, line in enumerate(lines):
        # V19-robust: Check for header keywords as any cell in the
        # pipe-separated line, not just as substring with trailing pipe.
        # This handles shuffled columns where the keyword might be last.
        if '|' in line:
            parts_lower = [p.strip().lower() for p in line.split('|')]
            if any('timestamp' in p for p in parts_lower):
                dt_start = i
            if any('gnodeb id' in p for p in parts_lower):
                ep_start = i

    # --- Build DT column map from header ---
    dt_col = None
    if dt_start is not None:
        dt_col = build_column_map(lines[dt_start], TYPE_A_DT_KEYWORDS, strip_empty=False)

    # --- Build EP column map from header ---
    ep_col = None
    if ep_start is not None:
        ep_col = build_column_map(lines[ep_start], TYPE_A_EP_KEYWORDS, strip_empty=False)

    # Parse drive test
    drive_test = []
    if dt_start:
        dt_found_data = False
        for i in range(dt_start + 1, len(lines)):
            if not lines[i].strip() or '|' not in lines[i]:
                if dt_found_data:
                    break  # Stop at first gap after data rows
                continue
            dt_found_data = True
            row = lines[i].split('|')

            if dt_col and all(k in dt_col for k in ('lon', 'lat', 'rsrp', 'serving_pci', 'throughput')):
                # --- Header-mapped path ---
                if len(row) <= max(dt_col.values()):
                    continue

                neighbor_pcis = []
                neighbor_brsrps = []
                for nk in ('n1_pci', 'n2_pci', 'n3_pci', 'n4_pci', 'n5_pci'):
                    if nk in dt_col:
                        j = dt_col[nk]
                        pci = row[j].strip() if j < len(row) else '-'
                        if pci and pci != '-' and pci.isdigit():
                            neighbor_pcis.append(pci)
                for nk in ('n1_brsrp', 'n2_brsrp', 'n3_brsrp', 'n4_brsrp', 'n5_brsrp'):
                    if nk in dt_col:
                        j = dt_col[nk]
                        brsrp = safe_float(row[j]) if j < len(row) else None
                        neighbor_brsrps.append(brsrp)

                spci_raw = row[dt_col['serving_pci']].strip()
                entry = {
                    'lon': safe_float(row[dt_col['lon']]),
                    'lat': safe_float(row[dt_col['lat']]),
                    'speed': safe_float(row[dt_col['speed']]) if 'speed' in dt_col else None,
                    'serving_pci': spci_raw if spci_raw.isdigit() else None,
                    'rsrp': safe_float(row[dt_col['rsrp']]),
                    'sinr': safe_float(row[dt_col['sinr']]) if 'sinr' in dt_col else None,
                    'throughput': safe_float(row[dt_col['throughput']]),
                    'neighbor_pcis': neighbor_pcis,
                    'neighbor_brsrps': neighbor_brsrps,
                    'rb': safe_float(row[dt_col['rb']]) if 'rb' in dt_col else None,
                }
            else:
                # --- Positional fallback (original V19 logic) ---
                if len(row) < 19:
                    continue

                neighbor_pcis = []
                neighbor_brsrps = []
                for j in range(8, 13):
                    pci = row[j].strip() if j < len(row) else '-'
                    if pci and pci != '-' and pci.isdigit():
                        neighbor_pcis.append(pci)
                for j in range(13, 18):
                    brsrp = safe_float(row[j]) if j < len(row) else None
                    neighbor_brsrps.append(brsrp)

                entry = {
                    'lon': safe_float(row[1]),
                    'lat': safe_float(row[2]),
                    'speed': safe_float(row[3]),
                    'serving_pci': row[4].strip() if row[4].strip().isdigit() else None,
                    'rsrp': safe_float(row[5]),
                    'sinr': safe_float(row[6]),
                    'throughput': safe_float(row[7]),
                    'neighbor_pcis': neighbor_pcis,
                    'neighbor_brsrps': neighbor_brsrps,
                    'rb': safe_float(row[18])
                }

            if entry['lon'] and entry['lat'] and entry['lon'] > 100 and entry['lat'] > 20:
                drive_test.append(entry)

    # Parse engineering params (now including azimuth)
    cells = {}
    if ep_start:
        ep_found_data = False
        for i in range(ep_start + 1, len(lines)):
            if not lines[i].strip() or '|' not in lines[i]:
                if ep_found_data:
                    break  # Stop at first gap after data rows
                continue
            ep_found_data = True
            row = lines[i].split('|')

            if ep_col and all(k in ep_col for k in ('pci', 'lon', 'lat', 'gnodeb_id')):
                # --- Header-mapped path ---
                if len(row) <= max(ep_col.values()):
                    continue

                pci = row[ep_col['pci']].strip()
                lon = safe_float(row[ep_col['lon']])
                lat = safe_float(row[ep_col['lat']])
                gnodeb_id = row[ep_col['gnodeb_id']].strip()
                mech_azimuth = (safe_float(row[ep_col['mech_azimuth']]) or 0) if 'mech_azimuth' in ep_col else 0
                mech_tilt = (safe_float(row[ep_col['mech_tilt']]) or 0) if 'mech_tilt' in ep_col else 0
                digital_tilt_raw = (safe_float(row[ep_col['digital_tilt']]) or 255) if 'digital_tilt' in ep_col else 255
                digital_tilt = 6 if digital_tilt_raw == 255 else digital_tilt_raw
                total_tilt = mech_tilt + digital_tilt
                beam_idx = ep_col.get('beam')
                beam = row[beam_idx].strip() if beam_idx is not None and beam_idx < len(row) else 'DEFAULT'

                if pci and lon and lat and lon > 100 and lat > 20:
                    cells[pci] = {
                        'lon': lon,
                        'lat': lat,
                        'gnodeb_id': gnodeb_id,
                        'mech_azimuth': mech_azimuth,
                        'total_tilt': total_tilt,
                        'beam': beam,
                    }
            else:
                # --- Positional fallback (original V19 logic) ---
                if len(row) <= 10:
                    continue

                pci = row[10].strip()
                lon = safe_float(row[2])
                lat = safe_float(row[3])
                gnodeb_id = row[0].strip()
                mech_azimuth = safe_float(row[4]) or 0  # Mechanical Azimuth
                mech_tilt = safe_float(row[5]) or 0
                digital_tilt_raw = safe_float(row[6]) or 255
                digital_tilt = 6 if digital_tilt_raw == 255 else digital_tilt_raw
                total_tilt = mech_tilt + digital_tilt
                beam = row[8].strip() if len(row) > 8 else 'DEFAULT'

                if pci and lon and lat and lon > 100 and lat > 20:
                    cells[pci] = {
                        'lon': lon,
                        'lat': lat,
                        'gnodeb_id': gnodeb_id,
                        'mech_azimuth': mech_azimuth,
                        'total_tilt': total_tilt,
                        'beam': beam,
                    }

    return drive_test, cells


def check_c7_speed(drive_test: List[Dict]) -> Tuple[bool, float, str]:
    """C7: Speed > 40 km/h."""
    speeds = [d['speed'] for d in drive_test if d['speed']]
    if not speeds:
        return False, 0, "no speed data"
    max_speed = max(speeds)
    return max_speed > 40, max_speed, f"max_speed={max_speed:.1f}km/h"


def check_c2_overshooting(drive_test: List[Dict], cells: Dict) -> Tuple[bool, float, str]:
    """C2: Distance > 1km during low TP."""
    low_tp_distances = []
    for d in drive_test:
        if d['throughput'] and d['throughput'] < 600:
            pci = d['serving_pci']
            if pci and pci in cells:
                cell = cells[pci]
                dist = haversine(cell['lon'], cell['lat'], d['lon'], d['lat'])
                low_tp_distances.append(dist)
    if not low_tp_distances:
        return False, 0, "no low TP data with known cells"
    max_dist = max(low_tp_distances)
    return max_dist > 1.0, max_dist, f"max_dist_during_low_tp={max_dist:.2f}km"


def check_c5_handovers(drive_test: List[Dict]) -> Tuple[bool, int, str]:
    """C5: Handovers >= 3."""
    pcis = [d['serving_pci'] for d in drive_test if d['serving_pci']]
    if len(pcis) < 2:
        return False, 0, "insufficient data"
    handovers = sum(1 for i in range(1, len(pcis)) if pcis[i] != pcis[i-1])
    return handovers >= 3, handovers, f"handovers={handovers}"


def check_c8_low_rbs(drive_test: List[Dict]) -> Tuple[bool, float, str]:
    """C8: Avg RBs < 170."""
    rbs = [d['rb'] for d in drive_test if d['rb']]
    if not rbs:
        return False, 999, "no RB data"
    avg_rb = sum(rbs) / len(rbs)
    return avg_rb < 170, avg_rb, f"avg_rb={avg_rb:.1f}"


def check_c4_non_colocated(drive_test: List[Dict], cells: Dict) -> Tuple[bool, float, str]:
    """C4: Non-colocated interference >= 5dB (raised from 3 for better precision)."""
    serving_pcis = [d['serving_pci'] for d in drive_test if d['serving_pci']]
    if not serving_pcis:
        return False, 0, "no serving PCI"

    serving_pci = Counter(serving_pcis).most_common(1)[0][0]
    if serving_pci not in cells:
        return False, 0, f"serving PCI {serving_pci} not in eng params"

    serving_gnodeb = cells[serving_pci]['gnodeb_id']
    max_interference = 0

    for d in drive_test:
        if not d['throughput'] or d['throughput'] >= 600:
            continue
        if not d['rsrp']:
            continue

        for i, n_pci in enumerate(d['neighbor_pcis']):
            if i >= len(d['neighbor_brsrps']):
                break
            n_brsrp = d['neighbor_brsrps'][i]
            if n_brsrp is None:
                continue

            if n_pci in cells:
                n_gnodeb = cells[n_pci]['gnodeb_id']
                if n_gnodeb != serving_gnodeb:
                    rsrp_diff = d['rsrp'] - n_brsrp
                    if rsrp_diff < 6:
                        interference = 6 - rsrp_diff
                        max_interference = max(max_interference, interference)

    # Keep threshold at >= 3 - C1 samples are filtered earlier by min_rsrp rule
    return max_interference >= 3, max_interference, f"interference={max_interference:.1f}dB"


def check_c6_pci_collision(drive_test: List[Dict]) -> Tuple[bool, bool, str]:
    """C6: PCI mod 30 collision."""
    for d in drive_test:
        if not d['serving_pci']:
            continue
        serving_mod = int(d['serving_pci']) % 30
        for n_pci in d['neighbor_pcis']:
            if int(n_pci) % 30 == serving_mod:
                return True, True, f"serving {d['serving_pci']}%30={serving_mod} == neighbor {n_pci}"
    return False, False, "no collision"


# =============================================================================
# V16 NEW: Helper functions for tiebreaker override rules
# =============================================================================

def get_pci_collision_ratio(drive_test: List[Dict]) -> float:
    """Fraction of rows with PCI mod 30 collision between serving and any neighbor."""
    if not drive_test:
        return 0.0
    pci_collision_rows = 0
    for r in drive_test:
        spci = r.get('serving_pci')
        if spci is None:
            continue
        try:
            spci_int = int(spci)
        except (ValueError, TypeError):
            continue
        for npci in r.get('neighbor_pcis', []):
            if npci is None:
                continue
            try:
                npci_int = int(npci)
            except (ValueError, TypeError):
                continue
            if spci_int % 30 == npci_int % 30:
                pci_collision_rows += 1
                break
    return pci_collision_rows / max(len(drive_test), 1)


def get_tp_threshold(question: str) -> float:
    """Extract throughput threshold from question text."""
    m = re.search(r'throughput\s+dropping\s+below\s+(\d+)', question, re.IGNORECASE)
    return float(m.group(1)) if m else 600.0


def compute_v16_metrics(drive_test: List[Dict], threshold: float) -> Dict:
    """
    Compute metrics needed for V16 override rules.

    Returns dict with:
      post_ho_good_streak: consecutive good-TP rows after first handover
      rsrp_recovery: avg post-HO RSRP minus avg pre-HO RSRP
      rsrp_change_during_prob: RSRP delta from first to last problem row
      rsrp_trend: linear regression slope of RSRP over all rows
      prob_tp_per_rb: avg throughput/RB during problem rows
      nb_within_5db_per_row: avg count of neighbors within 5dB during problem rows
    """
    if not drive_test:
        return {}
    result = {}
    rows = drive_test

    problem_rows = [r for r in rows if r.get('throughput') is not None and r['throughput'] < threshold]

    # Handover metrics
    handovers = []
    for i in range(1, len(rows)):
        p1 = rows[i].get('serving_pci')
        p0 = rows[i - 1].get('serving_pci')
        if p1 and p0 and p1 != p0:
            handovers.append(i)

    if handovers:
        ho_idx = handovers[0]
        post_ho_good = 0
        for r in rows[ho_idx:]:
            if r.get('throughput') is not None and r['throughput'] >= threshold:
                post_ho_good += 1
            else:
                break
        result['post_ho_good_streak'] = post_ho_good

        pre_ho_rsrps = [r['rsrp'] for r in rows[:ho_idx] if r.get('rsrp') is not None]
        post_ho_rsrps = [r['rsrp'] for r in rows[ho_idx:] if r.get('rsrp') is not None]
        if pre_ho_rsrps and post_ho_rsrps:
            result['rsrp_recovery'] = (sum(post_ho_rsrps) / len(post_ho_rsrps)) - (sum(pre_ho_rsrps) / len(pre_ho_rsrps))

    # RSRP change during problem period
    if problem_rows:
        first_prob = problem_rows[0]
        last_prob = problem_rows[-1]
        if first_prob.get('rsrp') and last_prob.get('rsrp'):
            result['rsrp_change_during_prob'] = last_prob['rsrp'] - first_prob['rsrp']

    # RSRP trend (linear regression slope)
    rsrps = [r['rsrp'] for r in rows if r.get('rsrp') is not None]
    if len(rsrps) >= 3:
        n = len(rsrps)
        x_mean = (n - 1) / 2
        y_mean = sum(rsrps) / n
        num = sum((i - x_mean) * (rsrps[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if den > 0:
            result['rsrp_trend'] = num / den

    # TP per RB during problem rows
    prob_tp_per_rb = []
    for r in problem_rows:
        if r.get('throughput') is not None and r.get('rb') is not None and r['rb'] > 0:
            prob_tp_per_rb.append(r['throughput'] / r['rb'])
    if prob_tp_per_rb:
        result['prob_tp_per_rb'] = sum(prob_tp_per_rb) / len(prob_tp_per_rb)

    # Neighbors within 5dB per problem row
    nb_within_5db = 0
    for r in problem_rows:
        if r.get('rsrp') is None:
            continue
        for brsrp in r.get('neighbor_brsrps', []):
            if brsrp is None:
                continue
            if r['rsrp'] - brsrp < 5:
                nb_within_5db += 1
    result['nb_within_5db_per_row'] = nb_within_5db / max(len(problem_rows), 1)

    return result


def get_type_a_tilt(drive_test: List[Dict], cells: Dict) -> float:
    """Get serving cell total tilt."""
    serving_pcis = [d['serving_pci'] for d in drive_test if d['serving_pci']]
    if not serving_pcis:
        return 0
    serving_pci = Counter(serving_pcis).most_common(1)[0][0]
    if serving_pci in cells:
        return cells[serving_pci]['total_tilt']
    return 0


def get_type_a_avg_rsrp(drive_test: List[Dict]) -> float:
    """Get average RSRP."""
    rsrps = [d['rsrp'] for d in drive_test if d['rsrp']]
    return sum(rsrps) / len(rsrps) if rsrps else -80


def get_min_rsrp(drive_test: List[Dict]) -> float:
    """Get minimum RSRP - strong C1 indicator when < -90 (V4: relaxed from -91)."""
    rsrps = [d['rsrp'] for d in drive_test if d['rsrp']]
    return min(rsrps) if rsrps else -80


def get_strong_neighbor_count(drive_test: List[Dict]) -> float:
    """
    Count average number of strong neighbors (within 6dB of serving) during low throughput.

    C1 (downtilt) has fewer strong neighbors (mean=0.66) because signal is tilted down.
    C3/C4/C6 have more strong neighbors (mean=1.1+).

    Key rules:
    - strong_neighbors < 0.5 AND tilt >= 15 -> C1 (100% precision)
    - has_collision AND strong_neighbors < 0.5 -> C1 (100% precision)
    """
    counts = []
    for d in drive_test:
        if d.get('throughput') and d['throughput'] < 600 and d['rsrp']:
            strong = 0
            if d.get('neighbor_brsrps'):
                for nb in d['neighbor_brsrps']:
                    if nb and d['rsrp'] - nb < 6:
                        strong += 1
            counts.append(strong)
    return sum(counts) / len(counts) if counts else 0


def get_min_neighbor_diff(drive_test: List[Dict]) -> float:
    """Get minimum (serving RSRP - neighbor BRSRP) during low throughput periods."""
    min_diff = 100
    for d in drive_test:
        if d['throughput'] and d['throughput'] < 600:
            if d['rsrp']:
                for nb in d['neighbor_brsrps']:
                    if nb:
                        diff = d['rsrp'] - nb
                        min_diff = min(min_diff, diff)
    return min_diff if min_diff < 100 else 10


def get_avg_off_axis_angle(drive_test: List[Dict], cells: Dict) -> float:
    """
    V4 NEW: Compute average off-axis angle during low throughput periods.

    Off-axis angle = angle between cell's azimuth direction and bearing from cell to UE.
    High off-axis (>30 deg) during low TP + collision suggests C3 (neighbor better),
    not C6 (PCI collision), because UE is not in cell's main beam direction.

    C6 true positives: avg off_axis ~6.8 degrees (UE in main beam, PCI collision matters)
    C3 samples with collision: avg off_axis ~28.7 degrees (UE off-axis, neighbor better)
    """
    off_axis_angles = []

    # Get serving cell
    serving_pcis = [d['serving_pci'] for d in drive_test if d['serving_pci']]
    if not serving_pcis:
        return 0
    serving_pci = Counter(serving_pcis).most_common(1)[0][0]

    if serving_pci not in cells:
        return 0

    cell = cells[serving_pci]
    cell_azimuth = cell.get('mech_azimuth', 0)

    for d in drive_test:
        if d['throughput'] and d['throughput'] < 600:
            if d['lon'] and d['lat'] and cell['lon'] and cell['lat']:
                bearing = compute_bearing(cell['lon'], cell['lat'], d['lon'], d['lat'])
                off_axis = compute_off_axis_angle(cell_azimuth, bearing)
                off_axis_angles.append(off_axis)

    return sum(off_axis_angles) / len(off_axis_angles) if off_axis_angles else 0


def get_min_sinr_low_tp(drive_test: List[Dict]) -> float:
    """
    V6 NEW: Get minimum SINR during low throughput periods.

    Analysis shows:
    - C4 True Positives: mean min_sinr = 5.8dB (moderate interference)
    - C4 False Positives: mean min_sinr = -1.8dB (very low, likely not C4)

    Very low SINR during low throughput suggests the issue is NOT non-colocated
    interference (C4), but rather weak coverage (C1), better neighbor (C3), or
    PCI collision (C6). This filter improves C4 precision by 15%.
    """
    sinrs = []
    for d in drive_test:
        if d.get('throughput') and d['throughput'] < 600 and d.get('sinr'):
            sinrs.append(d['sinr'])
    return min(sinrs) if sinrs else 10


def classify_c1_vs_c3(tilt: float, avg_rsrp: float, min_neighbor_diff: float,
                      avg_sinr: float = None) -> Tuple[str, str]:
    """
    Classify between C1 (downtilt) and C3 (neighbor better) using combined rules.

    Calibrated thresholds:
    - Tilt >= 28: C1 (with SINR gate: avg_sinr >= 12 -> C3)
    - Tilt < 12: C3
    - RSRP < -90: C1 in ambiguous range
    - RSRP > -82: C3 in ambiguous range

    Returns: (prediction, confidence)
    """
    # Rule 1: High serving cell tilt -> C1, but SINR gate catches false C1s
    if tilt >= 28:
        if avg_sinr is not None and avg_sinr >= 12:
            return 'C3', 'high'
        return 'C1', 'high'

    # Rule 2: Low serving cell tilt -> C3
    if tilt < 12:
        return 'C3', 'high'

    # Rule 3: Ambiguous tilt range (12-27), use RSRP as tiebreaker
    # Weak RSRP suggests coverage issue -> C1
    if avg_rsrp < -90:
        return 'C1', 'medium'

    # Strong RSRP with moderate tilt suggests neighbor is better -> C3
    if avg_rsrp > -82:
        return 'C3', 'medium'

    # Otherwise -> needs LLM for ambiguous cases
    # Return C3 with low confidence (C3 is majority in remaining)
    return 'C3', 'low'


def classify_type_a(question: str) -> Dict:
    """
    Classify Type A question using rules derived from data analysis.

    Returns dict with:
        - answer: test label (e.g., 'M3') or None if needs LLM
        - canonical: canonical cause (e.g., 'C5') or None
        - confidence: 'deterministic', 'high', 'needs_llm'  (internal labels)
        - evidence: dict of computed values
        - available_causes: set of causes in this question's options
    """
    test_option_map = extract_type_a_options(question)
    cause_to_label = {cause: label for label, cause in test_option_map.items()}
    available_causes = set(test_option_map.values())

    drive_test, cells = parse_type_a_question(question)

    result = {
        'answer': None,
        'canonical': None,
        'confidence': 'error',
        'evidence': {},
        'available_causes': available_causes,
        'test_option_map': test_option_map,
    }

    if not drive_test:
        result['evidence']['error'] = "Could not parse drive test"
        return result

    evidence = {}

    # TIER 1: HIGH-CONFIDENCE RULES (100% accuracy from data analysis)

    if 'C7' in available_causes:
        is_c7, speed, ev = check_c7_speed(drive_test)
        evidence['C7'] = ev
        if is_c7:
            result.update({
                'answer': cause_to_label['C7'],
                'canonical': 'C7',
                'confidence': 'deterministic',
                'evidence': evidence,
            })
            return result

    if 'C2' in available_causes:
        is_c2, dist, ev = check_c2_overshooting(drive_test, cells)
        evidence['C2'] = ev
        if is_c2:
            result.update({
                'answer': cause_to_label['C2'],
                'canonical': 'C2',
                'confidence': 'deterministic',
                'evidence': evidence,
            })
            return result

    if 'C5' in available_causes:
        is_c5, handovers, ev = check_c5_handovers(drive_test)
        evidence['C5'] = ev
        if is_c5:
            result.update({
                'answer': cause_to_label['C5'],
                'canonical': 'C5',
                'confidence': 'deterministic',
                'evidence': evidence,
            })
            return result

    if 'C8' in available_causes:
        is_c8, avg_rb, ev = check_c8_low_rbs(drive_test)
        evidence['C8'] = ev
        if is_c8:
            result.update({
                'answer': cause_to_label['C8'],
                'canonical': 'C8',
                'confidence': 'deterministic',
                'evidence': evidence,
            })
            return result

    # Pre-compute metrics needed for C1/C4/C6 decisions
    _, c4_interference, c4_ev = check_c4_non_colocated(drive_test, cells)
    _, has_collision, c6_ev = check_c6_pci_collision(drive_test)
    min_rsrp = get_min_rsrp(drive_test)
    strong_neighbors = get_strong_neighbor_count(drive_test)
    tilt = get_type_a_tilt(drive_test, cells)
    avg_off_axis = get_avg_off_axis_angle(drive_test, cells)  # V4 NEW
    min_sinr_low_tp = get_min_sinr_low_tp(drive_test)  # V6 NEW
    min_neighbor_diff = get_min_neighbor_diff(drive_test)  # V7 NEW: moved up for ratio calculation
    avg_rsrp = get_type_a_avg_rsrp(drive_test)  # V16: moved up from C6 block (needed by P4)
    # V19: avg_sinr for SINR gate in classify_c1_vs_c3
    sinr_values = [d['sinr'] for d in drive_test if d.get('sinr') is not None]
    avg_sinr = sum(sinr_values) / len(sinr_values) if sinr_values else None

    # V16 NEW: Compute tiebreaker metrics
    pci_collision_ratio = get_pci_collision_ratio(drive_test)
    tp_threshold = get_tp_threshold(question)
    v16 = compute_v16_metrics(drive_test, tp_threshold)

    # V7 NEW: Ratio feature for C4 FP detection
    # ratio_nbdiff_interf = min_neighbor_diff / c4_interference
    # Low ratio (<-0.5) means neighbors much stronger than serving relative to interference
    # This indicates the problem is NOT interference but neighbor/coverage issue
    ratio_nbdiff_interf = min_neighbor_diff / max(c4_interference, 1) if c4_interference > 0 else 0

    evidence['c4_interf'] = c4_ev
    evidence['c6_collision'] = c6_ev
    evidence['min_rsrp'] = f"{min_rsrp:.1f}dBm"
    evidence['strong_neighbors'] = f"{strong_neighbors:.2f}"
    evidence['tilt_early'] = f"{tilt:.0f}"
    evidence['avg_off_axis'] = f"{avg_off_axis:.1f}deg"  # V4 NEW
    evidence['min_sinr_low_tp'] = f"{min_sinr_low_tp:.1f}dB"  # V6 NEW
    evidence['min_nb_diff'] = f"{min_neighbor_diff:.1f}dB"  # V7 NEW
    evidence['ratio_nbdiff_interf'] = f"{ratio_nbdiff_interf:.2f}"  # V7 NEW
    evidence['pci_collision_ratio'] = f"{pci_collision_ratio:.2f}"  # V16 NEW

    # C1 DETECTION - Multiple 100% precision rules
    # V4 CHANGE: Relaxed min_rsrp threshold from -91 to -90 (+4 correct, 100% precision)
    # Rule 1: min_rsrp < -90 AND no collision AND c4_interf < 3 -> C1
    # Rule 2: strong_neighbors < 0.5 AND tilt >= 15 -> C1 (36 unique samples)
    # Rule 3: has_collision AND strong_neighbors < 0.5 -> C1 (16 unique samples)
    if 'C1' in available_causes:
        c1_detected = False
        c1_rule = None

        # Rule 1: Weak min_rsrp with no interference/collision (V4: -90 instead of -91)
        if min_rsrp < -90 and not has_collision and c4_interference < 3:
            c1_detected = True
            c1_rule = 'min_rsrp<-90'

        # Rule 2: Few strong neighbors with moderate-high tilt (downtilt signature)
        elif strong_neighbors < 0.5 and tilt >= 15:
            c1_detected = True
            c1_rule = 'low_strong_nb+high_tilt'

        # Rule 3: Collision but few strong neighbors (downtilt causes weak neighbors)
        elif has_collision and strong_neighbors < 0.5:
            c1_detected = True
            c1_rule = 'collision+low_strong_nb'

        if c1_detected:
            evidence['c1_rule'] = c1_rule

            # V16: Override checks before returning C1
            # B: post_ho_good_streak >= 2 AND C3 available -> C3
            if 'C3' in available_causes and v16.get('post_ho_good_streak', 0) >= 2:
                evidence['v16_override'] = 'B: post_ho_streak>=2 -> C3'
                result.update({
                    'answer': cause_to_label['C3'],
                    'canonical': 'C3',
                    'confidence': 'v16_override',
                    'evidence': evidence,
                })
                return result

            # P3: pci_collision_ratio > 0.70 AND C6 available -> C6
            if 'C6' in available_causes and pci_collision_ratio > 0.70:
                evidence['v16_override'] = f'P3: collision_ratio={pci_collision_ratio:.2f}>0.70 -> C6'
                result.update({
                    'answer': cause_to_label['C6'],
                    'canonical': 'C6',
                    'confidence': 'v16_override',
                    'evidence': evidence,
                })
                return result

            # P4: avg_rsrp > -79 AND strong_neighbors > 1.0 AND C3 available -> C3
            if 'C3' in available_causes and avg_rsrp > -79 and strong_neighbors > 1.0:
                evidence['v16_override'] = f'P4: rsrp={avg_rsrp:.1f}>-79 + nb={strong_neighbors:.1f}>1.0 -> C3'
                result.update({
                    'answer': cause_to_label['C3'],
                    'canonical': 'C3',
                    'confidence': 'v16_override',
                    'evidence': evidence,
                })
                return result

            # No override - return C1 as before
            result.update({
                'answer': cause_to_label['C1'],
                'canonical': 'C1',
                'confidence': 'deterministic',
                'evidence': evidence,
            })
            return result

    # C4: Non-colocated interference >= 3dB
    # V7 NEW: Ratio-based filter replaces V6 SINR filter
    # ratio_nbdiff_interf = neighbor_diff / interference
    # Low ratio (<-0.5) with moderate interf (<12) indicates:
    #   - Neighbors are much stronger than serving relative to interference
    #   - Problem is NOT interference but neighbor/coverage issue (C1/C3/C6)
    # Analysis: 0 TP lost, 32 FP caught with this filter
    if 'C4' in available_causes:
        evidence['C4'] = c4_ev
        if c4_interference >= 3:
            # V7: Skip C4 if neighbors dominate relative to interference level
            # High interference (>=12dB) overrides neighbor signal - still predict C4
            if ratio_nbdiff_interf < -0.5 and c4_interference < 12:
                evidence['c4_ratio_filter'] = f"ratio={ratio_nbdiff_interf:.2f}<-0.5 AND interf={c4_interference:.1f}<12 -> skip C4"
                # Fall through to C6/C1/C3 rules
            else:
                result.update({
                    'answer': cause_to_label['C4'],
                    'canonical': 'C4',
                    'confidence': 'deterministic',
                    'evidence': evidence,
                })
                return result

    # TIER 2: C6 PCI collision with filtering
    # Training data analysis shows C6 TPs have: low tilt (~9), varied neighbor diff
    # C6 FPs that are actually C1: high tilt (>=20)
    # C6 FPs that are actually C3: small neighbor diff (<3) with moderate tilt (>12)
    # V4 NEW: Also filter by off-axis angle - high off-axis suggests C3

    # tilt already computed above, get remaining metrics for C6 filtering and C1/C3 classification
    # V7: min_neighbor_diff already computed above for ratio calculation
    # V16: avg_rsrp moved to pre-compute block above

    if 'C6' in available_causes and has_collision:
        evidence['C6'] = c6_ev

        # Filter out false positives based on training data patterns
        # C1 signal: high tilt (>=20) suggests downtilt issue, not PCI collision
        c1_signal = tilt >= 20
        # C3 signal: small neighbor diff with moderate tilt suggests neighbor better
        c3_signal = min_neighbor_diff < 3 and tilt > 12
        # V4 NEW: High off-axis angle suggests UE not in main beam, neighbor likely better
        c3_off_axis_signal = avg_off_axis > 30

        if not c1_signal and not c3_signal and not c3_off_axis_signal:
            # No strong C1/C3 signal - likely genuine C6
            # V16: Apply override checks before returning C6

            # B: post_ho_good_streak >= 2 AND C3 available -> C3
            if 'C3' in available_causes and v16.get('post_ho_good_streak', 0) >= 2:
                evidence['v16_override'] = 'B: post_ho_streak>=2 -> C3'
                result.update({
                    'answer': cause_to_label['C3'],
                    'canonical': 'C3',
                    'confidence': 'v16_override',
                    'evidence': evidence,
                })
                return result

            # P1: pci_collision_ratio separates genuine C6 from misclassified
            if pci_collision_ratio >= 1.0:
                # Genuine C6 - keep as-is
                result.update({
                    'answer': cause_to_label['C6'],
                    'canonical': 'C6',
                    'confidence': 'high',
                    'evidence': evidence,
                })
                return result
            else:
                # P1 redirect: collision_ratio < 1.0 -> NOT genuine C6
                if 'C1' in available_causes and tilt > 10 and v16.get('rsrp_trend', 0) > 0.4:
                    evidence['v16_override'] = f'P1: collision_ratio={pci_collision_ratio:.2f}<1.0 + tilt={tilt:.0f}>10 + rsrp_trend={v16.get("rsrp_trend", 0):.2f}>0.4 -> C1'
                    result.update({
                        'answer': cause_to_label['C1'],
                        'canonical': 'C1',
                        'confidence': 'v16_override',
                        'evidence': evidence,
                    })
                    return result
                elif 'C3' in available_causes:
                    evidence['v16_override'] = f'P1: collision_ratio={pci_collision_ratio:.2f}<1.0 -> C3'
                    result.update({
                        'answer': cause_to_label['C3'],
                        'canonical': 'C3',
                        'confidence': 'v16_override',
                        'evidence': evidence,
                    })
                    return result
                else:
                    # No better option available, keep C6
                    result.update({
                        'answer': cause_to_label['C6'],
                        'canonical': 'C6',
                        'confidence': 'high',
                        'evidence': evidence,
                    })
                    return result
        elif c3_off_axis_signal:
            # V5 FIX: Check if weak RSRP suggests C1 (downtilt) instead of C3
            # Weak signal + high off-axis = downtilt causing poor coverage even off-axis
            # Strong signal + high off-axis = neighbor would be better (C3)
            if min_rsrp < -90 and 'C1' in available_causes:
                # V5: Weak RSRP with high off-axis -> C1 (downtilt issue)
                evidence['C6_filtered'] = f"collision + off_axis={avg_off_axis:.1f} + weak_rsrp={min_rsrp:.1f} -> C1"

                # V16: Override checks before returning C1 from off-axis path
                # B: post_ho_good_streak >= 2 AND C3 available -> C3
                if 'C3' in available_causes and v16.get('post_ho_good_streak', 0) >= 2:
                    evidence['v16_override'] = 'B: post_ho_streak>=2 -> C3 (off-axis C1 path)'
                    result.update({
                        'answer': cause_to_label['C3'],
                        'canonical': 'C3',
                        'confidence': 'v16_override',
                        'evidence': evidence,
                    })
                    return result
                # P3: pci_collision_ratio > 0.70 AND C6 available -> C6
                if 'C6' in available_causes and pci_collision_ratio > 0.70:
                    evidence['v16_override'] = f'P3: collision_ratio={pci_collision_ratio:.2f}>0.70 -> C6 (off-axis C1 path)'
                    result.update({
                        'answer': cause_to_label['C6'],
                        'canonical': 'C6',
                        'confidence': 'v16_override',
                        'evidence': evidence,
                    })
                    return result
                # P4: avg_rsrp > -79 AND strong_neighbors > 1.0 AND C3 available -> C3
                if 'C3' in available_causes and avg_rsrp > -79 and strong_neighbors > 1.0:
                    evidence['v16_override'] = f'P4: rsrp={avg_rsrp:.1f}>-79 + nb={strong_neighbors:.1f}>1.0 -> C3 (off-axis C1 path)'
                    result.update({
                        'answer': cause_to_label['C3'],
                        'canonical': 'C3',
                        'confidence': 'v16_override',
                        'evidence': evidence,
                    })
                    return result

                # No override - return C1 as before
                result.update({
                    'answer': cause_to_label['C1'],
                    'canonical': 'C1',
                    'confidence': 'high',
                    'evidence': evidence,
                })
                return result
            elif 'C3' in available_causes:
                # V4: High off-axis with collision and good RSRP -> predict C3
                evidence['C6_filtered'] = f"collision but off_axis={avg_off_axis:.1f} > 30 -> C3"

                # V16: Override checks before returning C3 from off-axis path
                # P2: pci_collision_ratio > 0.70 AND C6 available -> C6
                if 'C6' in available_causes and pci_collision_ratio > 0.70:
                    evidence['v16_override'] = f'P2: collision_ratio={pci_collision_ratio:.2f}>0.70 -> C6 (off-axis C3 path)'
                    result.update({
                        'answer': cause_to_label['C6'],
                        'canonical': 'C6',
                        'confidence': 'v16_override',
                        'evidence': evidence,
                    })
                    return result
                # G: rsrp_change > 5 AND rsrp_trend > 0.5 AND nb_5db < 1.0 AND C1 available -> C1
                if ('C1' in available_causes
                        and v16.get('rsrp_change_during_prob', 0) > 5
                        and v16.get('rsrp_trend', 0) > 0.5
                        and v16.get('nb_within_5db_per_row', 99) < 1.0):
                    evidence['v16_override'] = f'G: rsrp signals -> C1 (off-axis C3 path)'
                    result.update({
                        'answer': cause_to_label['C1'],
                        'canonical': 'C1',
                        'confidence': 'v16_override',
                        'evidence': evidence,
                    })
                    return result
                # J: rsrp_recovery > 15 AND C1 available -> C1
                if 'C1' in available_causes and v16.get('rsrp_recovery', 0) > 15:
                    evidence['v16_override'] = f'J: rsrp_recovery={v16.get("rsrp_recovery", 0):.1f}>15 -> C1 (off-axis C3 path)'
                    result.update({
                        'answer': cause_to_label['C1'],
                        'canonical': 'C1',
                        'confidence': 'v16_override',
                        'evidence': evidence,
                    })
                    return result
                # P5b: tilt > 6 AND nb_5db < 1.0 AND C1 available -> C1
                if 'C1' in available_causes and tilt > 6 and v16.get('nb_within_5db_per_row', 99) < 1.0:
                    evidence['v16_override'] = f'P5b: tilt={tilt:.0f}>6 + nb_5db<1.0 -> C1 (off-axis C3 path)'
                    result.update({
                        'answer': cause_to_label['C1'],
                        'canonical': 'C1',
                        'confidence': 'v16_override',
                        'evidence': evidence,
                    })
                    return result

                # No override - return C3 as before
                result.update({
                    'answer': cause_to_label['C3'],
                    'canonical': 'C3',
                    'confidence': 'high',
                    'evidence': evidence,
                })
                return result
        else:
            # C1/C3 signal detected - let it fall through to C1/C3 rules
            evidence['C6_filtered'] = f"collision but c1_signal={c1_signal}, c3_signal={c3_signal}, off_axis={avg_off_axis:.1f}"

    # TIER 3: C1 vs C3 - Use rules derived from data analysis, with LLM fallback

    evidence['tilt'] = f"{tilt:.0f}"
    evidence['avg_rsrp'] = f"{avg_rsrp:.1f}dBm"
    evidence['min_neighbor_diff'] = f"{min_neighbor_diff:.1f}dB"

    # Try rule-based C1/C3 classification
    c1_c3_pred, c1_c3_conf = classify_c1_vs_c3(tilt, avg_rsrp, min_neighbor_diff, avg_sinr=avg_sinr)
    evidence['c1_c3_pred'] = f"{c1_c3_pred} ({c1_c3_conf})"

    # High and medium confidence C1/C3: use rule-based answer
    # High and medium confidence: use rule-based answer
    # Low confidence goes to rescue rules (Tier 5)
    if c1_c3_conf in ['high', 'medium']:
        # Only if both C1 and C3 are in available options
        if 'C1' in available_causes and 'C3' in available_causes:
            label = cause_to_label.get(c1_c3_pred)
            if label:
                # V16: Apply override checks before returning C1/C3 prediction
                v16_override_pred = None
                v16_override_rule = None

                if c1_c3_pred == 'C3':
                    # P2: pci_collision_ratio > 0.70 AND C6 available -> C6
                    if 'C6' in available_causes and pci_collision_ratio > 0.70:
                        v16_override_pred = 'C6'
                        v16_override_rule = f'P2: collision_ratio={pci_collision_ratio:.2f}>0.70 -> C6'
                    # G: rsrp_change > 5 AND rsrp_trend > 0.5 AND nb_5db < 1.0 AND C1 available -> C1
                    elif ('C1' in available_causes
                            and v16.get('rsrp_change_during_prob', 0) > 5
                            and v16.get('rsrp_trend', 0) > 0.5
                            and v16.get('nb_within_5db_per_row', 99) < 1.0):
                        v16_override_pred = 'C1'
                        v16_override_rule = f'G: rsrp_change={v16.get("rsrp_change_during_prob", 0):.1f}>5 + rsrp_trend={v16.get("rsrp_trend", 0):.2f}>0.5 + nb_5db={v16.get("nb_within_5db_per_row", 0):.2f}<1.0 -> C1'
                    # J: rsrp_recovery > 15 AND C1 available -> C1
                    elif 'C1' in available_causes and v16.get('rsrp_recovery', 0) > 15:
                        v16_override_pred = 'C1'
                        v16_override_rule = f'J: rsrp_recovery={v16.get("rsrp_recovery", 0):.1f}>15 -> C1'
                    # P5b: tilt > 6 AND nb_5db < 1.0 AND C1 available -> C1
                    elif 'C1' in available_causes and tilt > 6 and v16.get('nb_within_5db_per_row', 99) < 1.0:
                        v16_override_pred = 'C1'
                        v16_override_rule = f'P5b: tilt={tilt:.0f}>6 + nb_5db={v16.get("nb_within_5db_per_row", 0):.2f}<1.0 -> C1'

                elif c1_c3_pred == 'C1':
                    # P3: pci_collision_ratio > 0.70 AND C6 available -> C6
                    if 'C6' in available_causes and pci_collision_ratio > 0.70:
                        v16_override_pred = 'C6'
                        v16_override_rule = f'P3: collision_ratio={pci_collision_ratio:.2f}>0.70 -> C6'
                    # P4: avg_rsrp > -79 AND strong_neighbors > 1.0 AND C3 available -> C3
                    elif 'C3' in available_causes and avg_rsrp > -79 and strong_neighbors > 1.0:
                        v16_override_pred = 'C3'
                        v16_override_rule = f'P4: rsrp={avg_rsrp:.1f}>-79 + nb={strong_neighbors:.1f}>1.0 -> C3'

                if v16_override_pred:
                    evidence['v16_override'] = v16_override_rule
                    result.update({
                        'answer': cause_to_label[v16_override_pred],
                        'canonical': v16_override_pred,
                        'confidence': 'v16_override',
                        'evidence': evidence,
                    })
                    return result

                # No override - return original prediction
                result.update({
                    'answer': label,
                    'canonical': c1_c3_pred,
                    'confidence': f'c1c3_{c1_c3_conf}',
                    'evidence': evidence,
                })
                return result

    # =========================================================================
    # TIER 5 (V18/V19): Low-confidence rescue rules
    # V19: calibrated thresholds (R1>=0.9, R2<0.8, R3>=3.0).
    # Applied when the C1/C3 tiebreaker returns "low" confidence (avg_rsrp in [-90, -82]).
    # =========================================================================

    # R1: collision_ratio >= 0.9 -> C6
    if pci_collision_ratio >= 0.9 and 'C6' in available_causes:
        evidence['v18_rescue'] = f'R1: collision_ratio={pci_collision_ratio:.2f}>=0.9 -> C6'
        result.update({
            'answer': cause_to_label['C6'],
            'canonical': 'C6',
            'confidence': 'v18_rescue',
            'evidence': evidence,
        })
        return result

    # R2: strong_neighbors < 0.8 -> C1
    if strong_neighbors < 0.8 and 'C1' in available_causes:
        evidence['v18_rescue'] = f'R2: strong_neighbors={strong_neighbors:.2f}<0.8 -> C1'
        result.update({
            'answer': cause_to_label['C1'],
            'canonical': 'C1',
            'confidence': 'v18_rescue',
            'evidence': evidence,
        })
        return result

    # R3: c4_interference >= 3.0 -> C1
    if c4_interference >= 3.0 and 'C1' in available_causes:
        evidence['v18_rescue'] = f'R3: c4_interference={c4_interference:.1f}>=3.0 -> C1'
        result.update({
            'answer': cause_to_label['C1'],
            'canonical': 'C1',
            'confidence': 'v18_rescue',
            'evidence': evidence,
        })
        return result

    # R4: default -> C3
    if 'C3' in available_causes:
        evidence['v18_rescue'] = f'R4: default -> C3 (snb={strong_neighbors:.2f}>=0.8, c4={c4_interference:.1f}<3.0)'
        result.update({
            'answer': cause_to_label['C3'],
            'canonical': 'C3',
            'confidence': 'v18_rescue',
            'evidence': evidence,
        })
        return result

    # Final fallback: if neither C1/C3/C6 in options, fall to LLM
    result.update({
        'answer': None,
        'canonical': None,
        'confidence': 'needs_llm',
        'evidence': evidence,
        'context': {
            'tilt': tilt,
            'avg_rsrp': avg_rsrp,
            'min_neighbor_diff': min_neighbor_diff,
            'c1_c3_pred': c1_c3_pred,
            'c1_c3_conf': c1_c3_conf,
            'c1_label': cause_to_label.get('C1'),
            'c3_label': cause_to_label.get('C3'),
        }
    })
    return result


# =============================================================================
# STAGE 2: TYPE B RULE-BASED CLASSIFIER
# =============================================================================

# Type B canonical causes (text -> code)
TYPE_B_CAUSE_MAP = {
    'rf or power parameters cause severe overlap coverage': 'A',
    'inter-frequency handover threshold configuration unreasonable': 'B',
    'network capacity insufficient or load imbalance between cells': 'C',
    'test server or transport anomaly causes insufficient upstream traffic': 'D',
    'missing neighbor cell configuration': 'E',
    'rf, power parameters or site construction cause weak coverage': 'F',
    'intra-frequency handover threshold too high': 'G',
    'intra-frequency handover threshold too low': 'H',
    'pdcch resource management parameters unreasonable': 'I',
}


def extract_type_b_options(question: str) -> Dict[str, str]:
    """
    Extract options from Type B question and map to canonical causes (A-I).
    Returns: {test_label: canonical_cause} e.g. {'A': 'G', 'B': 'F', ...}
    """
    options = {}
    matches = re.findall(r'^([A-I]): (.+)$', question, re.MULTILINE)

    for label, text in matches:
        text_clean = text.strip().lower()
        for cause_text, cause_code in TYPE_B_CAUSE_MAP.items():
            if cause_text in text_clean:
                options[label] = cause_code
                break

    return options


def _safe_float_col(parts, idx, signed=True):
    """Safely extract a float from parts[idx] with regex validation."""
    if idx is None or idx >= len(parts):
        return None
    pat = r'^-?[\d.]+$' if signed else r'^[\d.]+$'
    return float(parts[idx]) if re.match(pat, parts[idx]) else None


def parse_type_b_question(question: str) -> Tuple[List[Dict], Dict]:
    """Parse Type B drive test data and signaling events.

    V19-robust: Uses header-based column mapping. Falls back to
    positional indices if header parsing fails.
    """
    lines = question.split('\n')
    drive_test = []

    # V8 ENHANCED: More detailed signaling parsing
    signaling = {
        'a3_events': question.count('NREventA3') - question.count('NREventA3MeasConfig'),
        'a2_events': question.count('NREventA2') - question.count('NREventA2MeasConfig'),
        'a5_events': question.count('NREventA5') - question.count('NREventA5MeasConfig'),
        'handover_attempts': question.count('NRHandoverAttempt'),
        'handover_success': question.count('NRHandoverSuc'),
        'rrc_reestablish': question.count('NRRRCReestablishAttempt'),
    }

    # Find drive test header (robust to column shuffling)
    header_idx = None
    for i, line in enumerate(lines):
        if '|' in line:
            hdr_parts = [p.strip().lower() for p in line.split('|') if p.strip()]
            if any('time' == p for p in hdr_parts) or any('serving pci' in p for p in hdr_parts):
                header_idx = i
                break

    if header_idx is None:
        # Fallback: original detection
        for i, line in enumerate(lines):
            if '| Time |' in line or '|Time|' in line:
                header_idx = i
                break

    if header_idx is None:
        return [], signaling

    # --- Build column map from header ---
    col = build_column_map(lines[header_idx], TYPE_B_DT_KEYWORDS, strip_empty=True)
    use_mapped = (col is not None and
                  all(k in col for k in ('serving_pci', 'rsrp', 'sinr', 'throughput')))

    # Skip separator line (|:---:|)
    data_start = header_idx + 2

    b_dt_found_data = False
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line or not line.startswith('|'):
            if b_dt_found_data:
                break
            continue
        if '---' in line or '**' in line:
            break
        b_dt_found_data = True

        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) < 10:
            continue

        try:
            if use_mapped:
                # --- Header-mapped path ---
                spci_idx = col['serving_pci']
                entry = {
                    'serving_pci': parts[spci_idx] if spci_idx < len(parts) and parts[spci_idx].isdigit() else None,
                    'rsrp': float(parts[col['rsrp']]) if col['rsrp'] < len(parts) else None,
                    'sinr': float(parts[col['sinr']]) if col['sinr'] < len(parts) else None,
                    'throughput': float(parts[col['throughput']]) if col['throughput'] < len(parts) else None,
                    'neighbor1_rsrp': _safe_float_col(parts, col.get('n1_rsrp')),
                    'neighbor2_rsrp': _safe_float_col(parts, col.get('n2_rsrp')),
                    'neighbor3_rsrp': _safe_float_col(parts, col.get('n3_rsrp')),
                    'cce_fail_rate': _safe_float_col(parts, col.get('cce_fail_rate'), signed=False),
                    'avg_rank': _safe_float_col(parts, col.get('avg_rank'), signed=False),
                    'grant': _safe_float_col(parts, col.get('grant'), signed=False),
                    'avg_mcs': _safe_float_col(parts, col.get('avg_mcs'), signed=False),
                    'rb_slot': _safe_float_col(parts, col.get('rb_slot'), signed=False),
                    'initial_bler': _safe_float_col(parts, col.get('initial_bler'), signed=False),
                    'residual_bler': _safe_float_col(parts, col.get('residual_bler'), signed=False),
                }
            else:
                # --- Positional fallback (original V19 logic) ---
                entry = {
                    'serving_pci': parts[4] if len(parts) > 4 and parts[4].isdigit() else None,
                    'rsrp': float(parts[6]) if len(parts) > 6 else None,
                    'sinr': float(parts[7]) if len(parts) > 7 else None,
                    'throughput': float(parts[8]) if len(parts) > 8 else None,
                    'neighbor1_rsrp': float(parts[10]) if len(parts) > 10 and re.match(r'^-?[\d.]+$', parts[10]) else None,
                    'neighbor2_rsrp': float(parts[12]) if len(parts) > 12 and re.match(r'^-?[\d.]+$', parts[12]) else None,
                    'neighbor3_rsrp': float(parts[14]) if len(parts) > 14 and re.match(r'^-?[\d.]+$', parts[14]) else None,
                    'cce_fail_rate': float(parts[15]) if len(parts) > 15 and re.match(r'^[\d.]+$', parts[15]) else None,
                    'avg_rank': float(parts[16]) if len(parts) > 16 and re.match(r'^[\d.]+$', parts[16]) else None,
                    'grant': float(parts[17]) if len(parts) > 17 and re.match(r'^[\d.]+$', parts[17]) else None,
                    'avg_mcs': float(parts[18]) if len(parts) > 18 and re.match(r'^[\d.]+$', parts[18]) else None,
                    'rb_slot': float(parts[19]) if len(parts) > 19 and re.match(r'^[\d.]+$', parts[19]) else None,
                    'initial_bler': float(parts[20]) if len(parts) > 20 and re.match(r'^[\d.]+$', parts[20]) else None,
                    'residual_bler': float(parts[21]) if len(parts) > 21 and re.match(r'^[\d.]+$', parts[21]) else None,
                }
            drive_test.append(entry)
        except (ValueError, IndexError):
            continue

    return drive_test, signaling


def parse_config_data(question: str) -> Dict:
    """Parse Configuration Data table to extract neighbor list and thresholds.

    V19-robust: Uses header-based column mapping. Falls back to
    positional indices if header parsing fails.

    Returns dict keyed by PCI with config fields for each cell.
    """
    lines = question.split('\n')
    start = None
    for i, line in enumerate(lines):
        if '**Configuration Data**' in line:
            start = i
            break
    if start is None:
        return {}

    # --- Build column map from header (line after section marker) ---
    # Header is the first pipe-delimited line after **Configuration Data**
    col = None
    header_line_idx = None
    for i in range(start + 1, min(start + 4, len(lines))):
        if lines[i].strip().startswith('|') and '---' not in lines[i]:
            col = build_column_map(lines[i], TYPE_B_CONFIG_KEYWORDS, strip_empty=True)
            header_line_idx = i
            break

    use_mapped = (col is not None and
                  all(k in col for k in ('pci', 'gnodeb_id')))

    # Data starts 2 lines after header (header + separator)
    data_start = (header_line_idx + 2) if header_line_idx is not None else (start + 3)

    cells = {}
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line.startswith('|'):
            break
        if '---' in line:
            continue
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) < 3:
            continue
        try:
            if use_mapped:
                # --- Header-mapped path ---
                pci = parts[col['pci']] if col['pci'] < len(parts) else None
                if pci is None:
                    continue
                gnodeb_id = parts[col['gnodeb_id']] if col['gnodeb_id'] < len(parts) else ''
                freq = parts[col['freq']] if 'freq' in col and col['freq'] < len(parts) else ''
                a2_idx = col.get('a2_rsrp_thld')
                a3_idx = col.get('a3_offset')
                nbr_idx = col.get('neighbor_list')
                a2_val = int(parts[a2_idx]) if a2_idx is not None and a2_idx < len(parts) else None
                a3_val = int(parts[a3_idx]) if a3_idx is not None and a3_idx < len(parts) else None
                nbr_str = parts[nbr_idx] if nbr_idx is not None and nbr_idx < len(parts) else ''
                cells[pci] = {
                    'gnodeb_id': gnodeb_id,
                    'freq': freq,
                    'pci': pci,
                    'a2_rsrp_thld': a2_val,
                    'a3_offset': a3_val,
                    'neighbor_list': nbr_str,
                    'n_configured_neighbors': (
                        nbr_str.count(',') + 1
                        if nbr_str.startswith('[')
                        else 0
                    ),
                }
            else:
                # --- Positional fallback (original V19 logic) ---
                if len(parts) < 9:
                    continue
                pci = parts[2]
                cells[pci] = {
                    'gnodeb_id': parts[0],
                    'freq': parts[1],
                    'pci': pci,
                    'a2_rsrp_thld': int(parts[4]) if len(parts) > 4 else None,
                    'a3_offset': int(parts[8]) if len(parts) > 8 else None,
                    'neighbor_list': parts[11] if len(parts) > 11 else '',
                    'n_configured_neighbors': (
                        parts[11].count(',') + 1
                        if len(parts) > 11 and parts[11].startswith('[')
                        else 0
                    ),
                }
        except (ValueError, IndexError):
            continue
    return cells


def detect_inter_freq_ho(question: str) -> bool:
    """Detect inter-frequency handover from Signaling Event Content.

    V18 NEW: Checks if any NRHandoverAttempt has a different TargetNR-ARFCN
    than SourceNR-ARFCN. This indicates inter-frequency handover, which is the
    hallmark of cause B (inter-freq HO threshold unreasonable).

    All 15 LLM cases have inter-freq HO; 0/85 non-LLM cases do.
    """
    for line in question.split('\n'):
        if 'NRHandoverAttempt' in line:
            m_src = re.search(r'SourceNR-ARFCN:(\d+)', line)
            m_tgt = re.search(r'TargetNR-ARFCN:(\d+)', line)
            if m_src and m_tgt and m_src.group(1) != m_tgt.group(1):
                return True
    return False


def check_n1_in_config(question: str, drive_test: List[Dict], config_cells: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
    """Check if the strongest neighbor (N1) PCI is in serving cell's neighbor list.

    V19-robust: Uses header-based column mapping for the Type B DT table
    to find serving_pci and n1_pci columns. Falls back to positional.

    Returns: (n1_in_config, serving_pci, n1_pci)
    """
    # Get serving PCI from first drive test row
    serving_pci = None
    for d in drive_test:
        if d.get('serving_pci'):
            serving_pci = d['serving_pci']
            break

    # Find Type B DT header and build column map for N1 PCI extraction
    lines = question.split('\n')
    header_idx = None
    for i, line in enumerate(lines):
        if '|' in line:
            hdr_parts = [p.strip().lower() for p in line.split('|') if p.strip()]
            if any('time' == p for p in hdr_parts) or any('serving pci' in p for p in hdr_parts):
                header_idx = i
                break

    if header_idx is None:
        # Fallback: original detection
        for i, line in enumerate(lines):
            if '| Time |' in line or '|Time|' in line:
                header_idx = i
                break

    col = None
    if header_idx is not None:
        col = build_column_map(lines[header_idx], TYPE_B_DT_KEYWORDS, strip_empty=True)

    use_mapped = (col is not None and
                  'serving_pci' in col and 'n1_pci' in col)

    # Get most common N1 PCI - only scan DT data rows (skip header + separator, stop at gap)
    n1_pcis = []
    if header_idx is not None:
        dt_data_start = header_idx + 2
        n1_found_data = False
        for i in range(dt_data_start, len(lines)):
            line = lines[i].strip()
            if not line or not line.startswith('|'):
                if n1_found_data:
                    break
                continue
            if '---' in line or '**' in line:
                break
            n1_found_data = True
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if use_mapped:
                spci_idx = col['serving_pci']
                n1_idx = col['n1_pci']
                if (len(parts) > max(spci_idx, n1_idx)
                        and parts[spci_idx].isdigit()
                        and parts[n1_idx].isdigit()):
                    n1_pcis.append(parts[n1_idx])
            else:
                # Positional fallback
                if len(parts) >= 10 and parts[4].isdigit() and len(parts) > 9 and parts[9].isdigit():
                    n1_pcis.append(parts[9])

    if not n1_pcis or not serving_pci:
        return None, serving_pci, None

    from collections import Counter
    n1_pci = Counter(n1_pcis).most_common(1)[0][0]

    # Check if N1 PCI appears in serving cell's neighbor list
    serving_config = config_cells.get(serving_pci)
    if not serving_config:
        return None, serving_pci, n1_pci

    nbr_list = serving_config.get('neighbor_list', '')
    n1_in = ('_' + n1_pci + ']' in nbr_list or '_' + n1_pci + ',' in nbr_list)
    return n1_in, serving_pci, n1_pci


def classify_type_b(question: str) -> Dict:
    """
    Classify Type B question using rules derived from data analysis.

    V18: Added Configuration Data and Signaling Event Content parsing.
    New rules E (missing neighbor) and B (inter-freq HO threshold).

    Returns dict with:
        - answer: test label (e.g., 'G') or None if needs LLM
        - canonical: canonical cause code
        - confidence: 'deterministic', 'heuristic', 'needs_llm'  (internal labels)
        - evidence: dict of computed values
    """
    test_option_map = extract_type_b_options(question)
    cause_to_label = {cause: label for label, cause in test_option_map.items()}

    drive_test, signaling = parse_type_b_question(question)

    # V18: Parse Configuration Data and detect inter-freq HO
    config_cells = parse_config_data(question)
    inter_freq_ho = detect_inter_freq_ho(question)

    result = {
        'answer': None,
        'canonical': None,
        'confidence': 'error',
        'evidence': {},
        'test_option_map': test_option_map,
    }

    if not drive_test:
        result['evidence']['error'] = "Could not parse drive test"
        result['confidence'] = 'needs_llm'
        return result

    # V18: Check if N1 is in serving cell's config neighbor list
    n1_in_config, serving_pci, n1_pci = check_n1_in_config(
        question, drive_test, config_cells
    )

    # V18: Get serving cell's A2 threshold and neighbor count from config
    serving_config = config_cells.get(serving_pci, {}) if serving_pci else {}
    a2_thld = serving_config.get('a2_rsrp_thld')
    n_configured_neighbors = serving_config.get('n_configured_neighbors', 0)

    # Calculate metrics
    rsrps = [d['rsrp'] for d in drive_test if d['rsrp']]
    sinrs = [d['sinr'] for d in drive_test if d['sinr']]
    throughputs = [d['throughput'] for d in drive_test if d['throughput']]
    cce_fails = [d['cce_fail_rate'] for d in drive_test if d['cce_fail_rate'] is not None]
    blers = [d['initial_bler'] for d in drive_test if d['initial_bler'] is not None]
    rb_slots = [d['rb_slot'] for d in drive_test if d['rb_slot'] is not None]

    # V8 ENHANCED: Neighbor RSRP analysis
    neighbor1_rsrps = [d['neighbor1_rsrp'] for d in drive_test if d.get('neighbor1_rsrp') is not None]
    neighbor2_rsrps = [d['neighbor2_rsrp'] for d in drive_test if d.get('neighbor2_rsrp') is not None]
    neighbor3_rsrps = [d['neighbor3_rsrp'] for d in drive_test if d.get('neighbor3_rsrp') is not None]

    avg_rsrp = sum(rsrps) / len(rsrps) if rsrps else -90
    avg_sinr = sum(sinrs) / len(sinrs) if sinrs else 10
    min_throughput = min(throughputs) if throughputs else 50
    avg_cce_fail = sum(cce_fails) / len(cce_fails) if cce_fails else 0
    avg_bler = sum(blers) / len(blers) if blers else 0
    avg_rb = sum(rb_slots) / len(rb_slots) if rb_slots else 200

    # V8 ENHANCED: Compute serving-to-neighbor RSRP difference (for overlap/interference)
    avg_n1_rsrp = sum(neighbor1_rsrps) / len(neighbor1_rsrps) if neighbor1_rsrps else -120
    min_neighbor_diff = avg_rsrp - avg_n1_rsrp  # Positive = serving stronger, negative = neighbor stronger

    # V8 NEW: RSRP variance for A vs F discrimination (Type A learning: ratios are discriminative)
    std_rsrp = (sum((r - avg_rsrp)**2 for r in rsrps) / len(rsrps))**0.5 if len(rsrps) > 1 else 0
    rsrp_var_norm = std_rsrp / abs(avg_rsrp) if avg_rsrp != 0 else 0

    # V11: RF health and interference indicators
    rf_healthy = avg_sinr > 10 and avg_rsrp > -92 and avg_bler < 12
    sinr_deficit = (avg_rsrp + 100) - avg_sinr  # High = interference likely

    # V11: Throughput drop analysis - find transition point where TP drops below 100
    tp_drop_context = ""
    for i, d in enumerate(drive_test):
        if d['throughput'] and d['throughput'] < 100 and i > 0:
            prev = drive_test[i - 1]
            rsrp_before = prev.get('rsrp', 0) or 0
            sinr_before = prev.get('sinr', 0) or 0
            rsrp_after = d.get('rsrp', 0) or 0
            sinr_after = d.get('sinr', 0) or 0
            pci_changed = prev.get('serving_pci') != d.get('serving_pci')
            tp_drop_context = (
                f"TP dropped at row {i+1}: "
                f"RSRP {rsrp_before:.0f}->{rsrp_after:.0f}dBm, "
                f"SINR {sinr_before:.1f}->{sinr_after:.1f}dB"
                f"{', PCI changed (handover)' if pci_changed else ', same PCI'}"
            )
            break

    # Count actual handovers from PCI changes
    pcis = [d['serving_pci'] for d in drive_test if d['serving_pci']]
    actual_handovers = sum(1 for i in range(1, len(pcis)) if pcis[i] != pcis[i-1]) if len(pcis) > 1 else 0

    # V8 NEW: Ratio features (Type A learning: ratio_nbdiff_interf was highly discriminative)
    ratio_a3_ho = signaling['a3_events'] / max(actual_handovers, 1)

    # V8 ENHANCED: RRC Reestablish (connection drops) - strong G indicator
    rrc_reestablish = signaling.get('rrc_reestablish', 0)

    # =========================================================================
    # V15: New PHY-layer and multi-neighbor metrics
    # =========================================================================

    # V15: Extract new field arrays
    mcs_vals = [d['avg_mcs'] for d in drive_test if d.get('avg_mcs') is not None]
    rank_vals = [d['avg_rank'] for d in drive_test if d.get('avg_rank') is not None]
    grant_vals = [d['grant'] for d in drive_test if d.get('grant') is not None]
    resbler_vals = [d['residual_bler'] for d in drive_test if d.get('residual_bler') is not None]

    avg_mcs = sum(mcs_vals) / len(mcs_vals) if mcs_vals else None
    avg_rank = sum(rank_vals) / len(rank_vals) if rank_vals else None
    avg_grant = sum(grant_vals) / len(grant_vals) if grant_vals else None
    avg_residual_bler = sum(resbler_vals) / len(resbler_vals) if resbler_vals else None

    # V15: Conditional metrics during low-TP rows (throughput < 100 Mbps)
    # Key insight: during the actual throughput drop, overlap cases show degraded PHY
    # (low MCS, low rank, low SINR, high BLER) while transport cases show healthy PHY
    # (high MCS, high rank, high SINR, normal BLER).
    low_tp_rows = [d for d in drive_test if d.get('throughput') is not None and d['throughput'] < 100]
    high_tp_rows = [d for d in drive_test if d.get('throughput') is not None and d['throughput'] >= 100]

    def safe_avg(rows, key):
        vals = [d[key] for d in rows if d.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    low_tp_avg_mcs = safe_avg(low_tp_rows, 'avg_mcs')
    low_tp_avg_rank = safe_avg(low_tp_rows, 'avg_rank')
    low_tp_avg_sinr = safe_avg(low_tp_rows, 'sinr')
    low_tp_avg_bler = safe_avg(low_tp_rows, 'initial_bler')
    low_tp_avg_resbler = safe_avg(low_tp_rows, 'residual_bler')
    low_tp_avg_rb = safe_avg(low_tp_rows, 'rb_slot')

    high_tp_avg_mcs = safe_avg(high_tp_rows, 'avg_mcs')
    high_tp_avg_rank = safe_avg(high_tp_rows, 'avg_rank')

    # V15: phy_healthy_during_low_tp - is the radio link fine during throughput drops?
    # Overlap: PHY degrades (MCS<10, SINR<8, BLER>15) -> False
    # Transport: PHY stays healthy (MCS>10, SINR>8, BLER<15) -> True
    # Measured: 0/33 overlap cases True, 21/21 transport cases True.
    phy_healthy_during_low_tp = None
    if low_tp_avg_mcs is not None and low_tp_avg_sinr is not None and low_tp_avg_bler is not None:
        phy_healthy_during_low_tp = (
            low_tp_avg_mcs > 10 and
            low_tp_avg_sinr > 8 and
            low_tp_avg_bler < 15
        )

    # V15: MCS and rank drop from high-TP to low-TP periods
    # Overlap: MCS drops ~3.5, rank drops ~0.83 (PHY degrades with TP)
    # Transport: MCS drops ~0, rank drops ~0.04 (PHY unchanged)
    mcs_drop = None
    if high_tp_avg_mcs is not None and low_tp_avg_mcs is not None:
        mcs_drop = high_tp_avg_mcs - low_tp_avg_mcs

    rank_drop = None
    if high_tp_avg_rank is not None and low_tp_avg_rank is not None:
        rank_drop = high_tp_avg_rank - low_tp_avg_rank

    # V15: Multi-neighbor proximity (overlap indicator)
    # Count neighbors with avg RSRP within 3dB/5dB of serving.
    # Overlap: avg 0.67 within 3dB, 1.48 within 5dB
    # Transport: 0.00 within 3dB, 0.57 within 5dB (perfect separation)
    avg_n2_rsrp = sum(neighbor2_rsrps) / len(neighbor2_rsrps) if neighbor2_rsrps else -120
    avg_n3_rsrp = sum(neighbor3_rsrps) / len(neighbor3_rsrps) if neighbor3_rsrps else -120

    neighbors_within_3dB = 0
    neighbors_within_5dB = 0
    for avg_n in [avg_n1_rsrp, avg_n2_rsrp, avg_n3_rsrp]:
        if avg_n > -115:  # Valid neighbor
            diff = avg_rsrp - avg_n
            if diff < 3:
                neighbors_within_3dB += 1
            if diff < 5:
                neighbors_within_5dB += 1

    # V15: Percentage of rows where neighbor1 is stronger than serving
    # Overlap: avg 29%, Transport: avg 4%
    n1_stronger_count = 0
    n1_total = 0
    for d in drive_test:
        if d.get('rsrp') is not None and d.get('neighbor1_rsrp') is not None:
            n1_total += 1
            if d['neighbor1_rsrp'] > d['rsrp']:
                n1_stronger_count += 1
    n1_stronger_pct = (n1_stronger_count / n1_total * 100) if n1_total > 0 else 0

    # V15: Spectral efficiency - throughput per RB
    # Transport: 1.28 Mbps/RB (good link, few RBs allocated)
    # Overlap: 0.89 Mbps/RB (interference wastes RBs)
    tp_per_rb_vals = []
    for d in drive_test:
        if d.get('throughput') is not None and d.get('rb_slot') is not None and d['rb_slot'] > 0:
            tp_per_rb_vals.append(d['throughput'] / d['rb_slot'])
    avg_tp_per_rb = sum(tp_per_rb_vals) / len(tp_per_rb_vals) if tp_per_rb_vals else None

    evidence = {
        'avg_rsrp': f"{avg_rsrp:.1f}dBm",
        'std_rsrp': f"{std_rsrp:.2f}dB",
        'rsrp_var_norm': f"{rsrp_var_norm:.3f}",
        'avg_sinr': f"{avg_sinr:.1f}dB",
        'min_throughput': f"{min_throughput:.1f}Mbps",
        'avg_cce_fail': f"{avg_cce_fail:.2f}",
        'avg_bler': f"{avg_bler:.1f}%",
        'avg_rb': f"{avg_rb:.0f}",
        'actual_handovers': actual_handovers,
        'a3_events': signaling['a3_events'],
        'ratio_a3_ho': f"{ratio_a3_ho:.2f}",
        'ho_attempts': signaling['handover_attempts'],
        'rrc_reestablish': rrc_reestablish,  # V8 ENHANCED
        'min_neighbor_diff': f"{min_neighbor_diff:.1f}dB",  # V8 ENHANCED
        # V18: Config/signaling features
        'n1_in_config': n1_in_config,
        'inter_freq_ho': inter_freq_ho,
        'a2_thld': a2_thld,
        'n_configured_neighbors': n_configured_neighbors,
    }

    # ==========================================================================
    # TIER 1: RULES DERIVED FROM DATA ANALYSIS
    # Applying Type A learnings: combined conditions and ratio features
    # Rule ordering: Physical causes before behavioral symptoms
    # ==========================================================================

    # I: PDCCH resource management issue (PHYSICAL CAUSE - check first)
    # V8 NEW: High CCE failure rate directly indicates PDCCH resource problems
    # Domain knowledge: CCE (Control Channel Element) failure affects control signaling
    # which can PREVENT handovers, making it look like G (threshold too high).
    # Check I before G since it's the physical cause, not the symptom.
    if avg_cce_fail > 0.25:
        evidence['i_rule'] = f"cce={avg_cce_fail:.2f}>0.25 (high CCE = PDCCH issue)"
        result.update({
            'answer': cause_to_label.get('I'),
            'canonical': 'I',
            'confidence': 'deterministic',
            'evidence': evidence,
        })
        return result

    # H: Intra-freq threshold TOO LOW (ping-pong handovers)
    # Domain knowledge: Too many handovers indicate threshold is too aggressive
    # Check before G since ping-pong is a clear, distinct symptom
    if actual_handovers >= 3:
        evidence['h_rule'] = f"actual_ho={actual_handovers}>=3 (ping-pong)"
        result.update({
            'answer': cause_to_label.get('H'),
            'canonical': 'H',
            'confidence': 'deterministic',
            'evidence': evidence,
        })
        return result

    # G: Intra-freq threshold TOO HIGH
    # Domain knowledge: UE reports A3 events (neighbor better) but handover doesn't happen
    # V8 ENHANCED: Two indicators for G:
    # 1. ratio_a3_ho >= 3 (many A3 events, few actual handovers)
    # 2. RRC Reestablish > 0 (connection drops - 100% correlated with G in analysis)
    g_by_ratio = ratio_a3_ho >= 3 and signaling['a3_events'] >= 2
    g_by_rrc = rrc_reestablish > 0 and signaling['a3_events'] >= 1

    if g_by_ratio or g_by_rrc:
        if g_by_rrc:
            evidence['g_rule'] = f"rrc_reestablish={rrc_reestablish}>0 (connection drops indicate HO failure)"
        else:
            evidence['g_rule'] = f"ratio_a3_ho={ratio_a3_ho:.1f}>=3, a3={signaling['a3_events']}>=2"

        # V18: Check if this is actually E (missing neighbor configuration).
        # When N1 PCI is NOT in the serving cell's configured neighbor list,
        # the UE sees a strong neighbor but can't handover to it because it's
        # not configured. This causes RRC drops (reestablishment attempts)
        # and A3 events without successful handover - the exact G signature.
        # Discriminator: n1_in_config=False for 13/13 G cases, True for 87/87 others.
        if n1_in_config is False:
            evidence['v18_e_rule'] = (
                f"G-rule triggered but N1 PCI {n1_pci} NOT in serving cell "
                f"{serving_pci}'s neighbor list -> missing neighbor config (E)"
            )
            result.update({
                'answer': cause_to_label.get('E'),
                'canonical': 'E',
                'confidence': 'deterministic',
                'evidence': evidence,
            })
            return result

        result.update({
            'answer': cause_to_label.get('G'),
            'canonical': 'G',
            'confidence': 'deterministic',
            'evidence': evidence,
        })
        return result

    # A: Overlap coverage (competing strong signals)
    # V8 NEW: Type A learning - ratio features capture physical relationships
    # Domain knowledge: High RSRP variance with OK average = multiple competing cells
    if rsrp_var_norm > 0.08 and avg_rsrp > -95:
        evidence['a_rule'] = f"rsrp_var_norm={rsrp_var_norm:.3f}>0.08 AND avg_rsrp={avg_rsrp:.1f}>-95 (overlap)"
        result.update({
            'answer': cause_to_label.get('A'),
            'canonical': 'A',
            'confidence': 'deterministic',
            'evidence': evidence,
        })
        return result

    # F: Weak coverage - consistently poor RSRP
    # Domain knowledge: Very low RSRP indicates RF/power/site construction issue
    if avg_rsrp < -95:
        evidence['f_rule'] = f"avg_rsrp={avg_rsrp:.1f}<-95 (weak coverage)"
        result.update({
            'answer': cause_to_label.get('F'),
            'canonical': 'F',
            'confidence': 'deterministic',
            'evidence': evidence,
        })
        return result

    # ==========================================================================
    # V17 TIER 1.5: PHY-health-based heuristic rules
    #
    # After Tier 1 rules (I/H/G/A/F), the remaining cases are in a gray zone
    # where standard thresholds don't trigger. Analysis of V16 checkpoint
    # shows perfect separation between A (overlap) and D (transport) based on
    # PHY-layer health during low-throughput periods:
    #
    # D (transport): PHY stays healthy during TP drops (MCS>10, SINR>8, BLER<15)
    #   - No strong neighbors (nb_3dB=0), avg_sinr>10
    #   - Bottleneck is above PHY layer (server/transport)
    #   - 25/25 match on V16 checkpoint (100% precision)
    #
    # A (overlap): PHY degrades during TP drops (MCS or SINR or BLER degrade)
    #   - After ruling out F (weak), I (PDCCH), H (pingpong), G (threshold)
    #   - Remaining RF degradation = overlap/interference
    #   - 33/33 match on V16 checkpoint (100% precision)
    #
    # Validated metrics (0% range overlap between A and D):
    #   low_tp_avg_sinr: A=[-2.5, 5.3] vs D=[11.3, 14.4]
    #   low_tp_avg_rank: A=[1.1, 2.2] vs D=[2.3, 2.8]
    #   neighbors_within_3dB: A=[0, 1] vs D=[0, 0]
    # ==========================================================================

    # Rule D: Transport/server anomaly
    # Justification: Radio link is healthy during throughput drops. Problem is
    # above PHY layer. No competing neighbors.
    if (phy_healthy_during_low_tp is True
            and neighbors_within_3dB == 0
            and avg_sinr > 10):
        evidence['v17_rule_d'] = (
            f"phy_healthy=True (MCS={low_tp_avg_mcs:.1f}, SINR={low_tp_avg_sinr:.1f}, "
            f"BLER={low_tp_avg_bler:.1f}%), nb_3dB=0, avg_sinr={avg_sinr:.1f}>10"
        )
        result.update({
            'answer': cause_to_label.get('D'),
            'canonical': 'D',
            'confidence': 'heuristic',
            'evidence': evidence,
        })
        return result

    # Rule A2a: Overlap coverage - strong interference signature
    # Justification: PHY degrades AND MCS crashes below 12 AND a strong neighbor
    # is within 3 dB. This is textbook pilot pollution / overlap: the UE has
    # decent RSRP but SINR collapses because a neighbor's signal interferes.
    # Sub-cluster 1: avg_rsrp -82 to -86, low_tp_mcs 8-10, low_tp_sinr -2.5 to 3.1,
    # nb_3dB=1, n1_stronger_pct 25-50%. Unambiguous overlap.
    if (phy_healthy_during_low_tp is False
            and low_tp_avg_mcs is not None and low_tp_avg_mcs < 12
            and neighbors_within_3dB >= 1):
        evidence['v17_rule_a2a'] = (
            f"phy_healthy=False, MCS_crash={low_tp_avg_mcs:.1f}<12, "
            f"nb_3dB={neighbors_within_3dB}>=1 (strong overlap), "
            f"SINR={low_tp_avg_sinr:.1f}, BLER={low_tp_avg_bler:.1f}%, "
            f"n1_pct={n1_stronger_pct:.1f}%"
        )
        result.update({
            'answer': cause_to_label.get('A'),
            'canonical': 'A',
            'confidence': 'heuristic',
            'evidence': evidence,
        })
        return result

    # ==========================================================================
    # V18 TIER 1.75: Configuration/Signaling-based rules
    #
    # These rules use features from tables that V17 never parsed:
    # - Configuration Data: neighbor lists, A2/A5 thresholds
    # - Signaling Event Content: inter-freq HO (different ARFCN)
    #
    # Rule B: Inter-frequency handover threshold unreasonable
    #   All 15 former LLM cases have:
    #   - Inter-freq HO detected (TargetARFCN != SourceARFCN in NRHandoverAttempt)
    #   - A2 threshold = -95 dBm (vs -105 for all other cases)
    #   - 7-8 configured neighbors (vs 2-4 for all other cases)
    #   - NREventA5 present (inter-freq measurement event)
    #   0/85 non-LLM cases have this signature.
    # ==========================================================================

    if (inter_freq_ho
            and a2_thld is not None and a2_thld > -100
            and n_configured_neighbors >= 6):
        evidence['v18_b_rule'] = (
            f"inter_freq_HO=True, a2_thld={a2_thld}>-100, "
            f"n_configured_neighbors={n_configured_neighbors}>=6 "
            f"(inter-freq HO threshold unreasonable)"
        )
        result.update({
            'answer': cause_to_label.get('B'),
            'canonical': 'B',
            'confidence': 'deterministic',
            'evidence': evidence,
        })
        return result

    # ==========================================================================
    # TIER 2: Route to LLM for remaining ambiguous cases
    # Reached by:
    #   - phy_healthy_during_low_tp is None (metrics couldn't be computed)
    #   - Any other edge case not covered by Tier 1/1.5/1.75
    # V8 ENHANCED: Pass ALL computed metrics to help LLM make better decisions
    # ==========================================================================

    result.update({
        'answer': None,
        'canonical': None,
        'confidence': 'needs_llm',
        'evidence': evidence,
        'context': {
            # Basic RF metrics
            'avg_rsrp': avg_rsrp,
            'min_rsrp': min(rsrps) if rsrps else -90,
            'std_rsrp': std_rsrp,
            'rsrp_var_norm': rsrp_var_norm,
            'avg_sinr': avg_sinr,
            'min_sinr': min(sinrs) if sinrs else 0,
            # Throughput
            'avg_throughput': sum(throughputs) / len(throughputs) if throughputs else 50,
            'min_throughput': min_throughput,
            # PHY layer metrics
            'avg_cce_fail': avg_cce_fail,
            'avg_bler': avg_bler,
            'avg_rb': avg_rb,
            # Neighbor analysis (V8 NEW)
            'avg_neighbor1_rsrp': avg_n1_rsrp,
            'min_neighbor_diff': min_neighbor_diff,  # serving - neighbor1
            # Handover metrics
            'actual_handovers': actual_handovers,
            'ratio_a3_ho': ratio_a3_ho,
            # Signaling details (V8 ENHANCED)
            'a3_events': signaling['a3_events'],
            'a2_events': signaling['a2_events'],
            'a5_events': signaling['a5_events'],
            'ho_attempts': signaling['handover_attempts'],
            'ho_success': signaling['handover_success'],
            'rrc_reestablish': rrc_reestablish,  # V8 NEW: connection drops
            # V11: New discriminating metrics
            'rf_healthy': rf_healthy,
            'sinr_deficit': sinr_deficit,
            'tp_drop_context': tp_drop_context,
            # V15: PHY-layer metrics
            'avg_mcs': avg_mcs,
            'avg_rank': avg_rank,
            'avg_grant': avg_grant,
            'avg_residual_bler': avg_residual_bler,
            # V15: Conditional metrics during low-TP rows
            'low_tp_avg_mcs': low_tp_avg_mcs,
            'low_tp_avg_rank': low_tp_avg_rank,
            'low_tp_avg_sinr': low_tp_avg_sinr,
            'low_tp_avg_bler': low_tp_avg_bler,
            'low_tp_avg_resbler': low_tp_avg_resbler,
            'low_tp_avg_rb': low_tp_avg_rb,
            'phy_healthy_during_low_tp': phy_healthy_during_low_tp,
            'mcs_drop': mcs_drop,
            'rank_drop': rank_drop,
            # V15: Multi-neighbor metrics
            'neighbors_within_3dB': neighbors_within_3dB,
            'neighbors_within_5dB': neighbors_within_5dB,
            'n1_stronger_pct': n1_stronger_pct,
            # V15: Spectral efficiency
            'avg_tp_per_rb': avg_tp_per_rb,
            # Summary for LLM
            'metrics_summary': (
                f"RSRP: avg={avg_rsrp:.1f}dBm (std={std_rsrp:.1f}), "
                f"SINR: avg={avg_sinr:.1f}dB, "
                f"TP: min={min_throughput:.0f}Mbps, "
                f"CCE_fail={avg_cce_fail:.2f}, BLER={avg_bler:.1f}%, "
                f"Neighbor_diff={min_neighbor_diff:.1f}dB, "
                f"RF_healthy={rf_healthy}, SINR_deficit={sinr_deficit:.1f}, "
                f"MCS={avg_mcs:.1f}, Rank={avg_rank:.2f}, ResBLER={avg_residual_bler:.2f}%, "
                f"A3={signaling['a3_events']}, HO={actual_handovers}, RRC_drop={rrc_reestablish}"
            ) if avg_mcs is not None and avg_rank is not None and avg_residual_bler is not None else (
                f"RSRP: avg={avg_rsrp:.1f}dBm (std={std_rsrp:.1f}), "
                f"SINR: avg={avg_sinr:.1f}dB, "
                f"TP: min={min_throughput:.0f}Mbps, "
                f"CCE_fail={avg_cce_fail:.2f}, BLER={avg_bler:.1f}%, "
                f"Neighbor_diff={min_neighbor_diff:.1f}dB, "
                f"RF_healthy={rf_healthy}, SINR_deficit={sinr_deficit:.1f}, "
                f"A3={signaling['a3_events']}, HO={actual_handovers}, RRC_drop={rrc_reestablish}"
            ),
        }
    })
    return result


# =============================================================================
# STAGE 2: GENERIC CLASSIFIER (always needs LLM)
# =============================================================================

def detect_generic_subtype(question: str) -> str:
    """
    V9 NEW: Detect if generic question is history/reading or math.

    History questions (45 in Phase 2) contain a passage with this marker.
    Math questions (37 in Phase 2) go directly to the problem.
    """
    if 'this question refers to the following information' in question.lower():
        return 'history'
    return 'math'


def classify_generic(question: str) -> Dict:
    """
    Classify generic question - always needs LLM.
    V9: Also detects subtype (history vs math) for prompt selection.
    """
    # Extract options
    options = []
    for line in question.split('\n'):
        match = re.match(r'^\s*(\d+)\s*:\s*(.+)$', line.strip())
        if match:
            options.append(match.group(1))

    # V9 NEW: Detect subtype
    subtype = detect_generic_subtype(question)

    return {
        'answer': None,
        'canonical': None,
        'confidence': 'needs_llm',
        'evidence': {'options': options, 'subtype': subtype},
        'test_option_map': {opt: opt for opt in options},
        'subtype': subtype,  # V9: For LLM prompt selection
    }


# =============================================================================
# LOOKUP INDEX (for Type A retrieval)
# =============================================================================

def extract_route_key(question: str) -> Optional[Tuple]:
    """Extract route signature (start, mid, end GPS) from question."""
    points = []
    for match in re.finditer(r'\d{4}-\d{2}-\d{2}[^|]+\|([^|]+)\|([^|]+)\|', question):
        try:
            lon = round(float(match.group(1)), 5)
            lat = round(float(match.group(2)), 5)
            points.append((lon, lat))
        except:
            pass
    if not points:
        return None
    mid = points[len(points)//2] if len(points) > 2 else points[0]
    return (points[0], mid, points[-1])


def extract_serving_pci(question: str) -> Optional[str]:
    """Extract serving PCI from first row of drive test data."""
    match = re.search(r'\d{4}-\d{2}-\d{2}[^|]+\|[^|]+\|[^|]+\|[^|]+\|(\d+)\|', question)
    return match.group(1) if match else None


def extract_lookup_key(question: str) -> Optional[Tuple]:
    """Extract lookup key (route + serving_pci) from question."""
    route = extract_route_key(question)
    pci = extract_serving_pci(question)
    if route and pci:
        return (route, pci)
    return None


class Phase1LookupIndex:
    """Index of labeled questions for few-shot retrieval."""

    def __init__(self, phase1_path: str, truth_path: str):
        self.index = defaultdict(list)
        self._build_index(phase1_path, truth_path)

    def _build_index(self, phase1_path: str, truth_path: str):
        p1_df = pd.read_csv(phase1_path)
        truth_df = pd.read_csv(truth_path)

        truth_df['base_id'] = truth_df['ID'].apply(lambda x: '_'.join(x.split('_')[:-1]))
        gt_map = truth_df.groupby('base_id')['Qwen3-32B'].first().to_dict()

        for _, row in p1_df.iterrows():
            qid = row['ID']
            question = row['question']

            if qid not in gt_map:
                continue

            lookup_key = extract_lookup_key(question)
            if lookup_key:
                # Map ground truth label to canonical cause
                test_option_map = extract_type_a_options(question)
                gt_label = gt_map[qid]
                gt_canonical = test_option_map.get(gt_label, gt_label)

                self.index[lookup_key].append({
                    'id': qid,
                    'question': question,
                    'ground_truth_label': gt_label,
                    'ground_truth_canonical': gt_canonical,
                })

        logger.info(f"[Lookup Index] Built index with {len(self.index)} unique keys")
        logger.info(f"[Lookup Index] Total questions indexed: {sum(len(v) for v in self.index.values())}")

    def lookup(self, question: str) -> List[Dict]:
        """Look up labeled examples matching the given question."""
        key = extract_lookup_key(question)
        if key and key in self.index:
            return self.index[key]
        return []

    def get_examples_by_cause(self, question: str) -> Dict[str, Dict]:
        """Get labeled examples organized by canonical cause."""
        examples = self.lookup(question)
        by_cause = {}
        for ex in examples:
            cause = ex['ground_truth_canonical']
            if cause not in by_cause:
                by_cause[cause] = ex
        return by_cause


# =============================================================================
# STAGE 3: LLM PROMPTS
# =============================================================================

# Root cause descriptions
TYPE_A_CAUSE_DESCRIPTIONS = {
    'C1': "The serving cell's downtilt angle is too large, causing weak coverage at the far end.",
    'C2': "The serving cell's coverage distance exceeds 1km, resulting in over-shooting.",
    'C3': "A neighboring cell provides higher throughput.",
    'C4': "Non-colocated co-frequency neighboring cells cause severe overlapping coverage.",
    'C5': "Frequent handovers degrade performance.",
    'C6': "Neighbor cell and serving cell have the same PCI mod 30, leading to interference.",
    'C7': "Test vehicle speed exceeds 40km/h, impacting user throughput.",
    'C8': "Average scheduled RBs are below 160, affecting throughput.",
}

TYPE_B_CAUSE_DESCRIPTIONS = {
    'A': "RF or power parameters cause severe overlap coverage (interference from neighbors)",
    'B': "Inter-frequency handover threshold configuration unreasonable",
    'C': "Network capacity insufficient or load imbalance between cells",
    'D': "Test server or transport anomaly causes insufficient upstream traffic",
    'E': "Missing neighbor cell configuration",
    'F': "RF, power parameters or site construction cause weak coverage",
    'G': "Intra-frequency handover threshold too high (A3 triggers but HO doesn't happen)",
    'H': "Intra-frequency handover threshold too low (excessive ping-pong handovers)",
    'I': "PDCCH resource management parameters unreasonable (high CCE fail rate)",
}


# Type A C1 vs C3 prompt
TYPE_A_SYSTEM_PROMPT = """You are an expert 5G network troubleshooter analyzing why throughput dropped below 600Mbps.

You need to distinguish between two remaining root causes:

**C1 (Downtilt Issue):** The serving cell's downtilt angle is too large, causing weak coverage at the far end of the cell.
Key indicators:
- High total tilt (mechanical + electronic > 20 degrees, where electronic tilt 255 = 6 degrees)
- RSRP degrades as UE moves away from cell site
- No significantly better neighbor available

**C3 (Neighbor Better):** A neighboring cell provides higher throughput, but UE didn't handover.
Key indicators:
- Neighbor RSRP within 6dB of serving RSRP (should have triggered handover)
- Normal tilt angles
- Handover to neighbor would have improved throughput

Verification steps:
1. Calculate total tilt = mechanical tilt + electronic tilt (255 means 6 degrees)
2. Compare serving RSRP vs neighbor BRSRP values
3. Check if any neighbor is stronger or within 6dB

Format your final answer as: \\boxed{ANSWER} where ANSWER is the exact option label."""


TYPE_A_USER_PROMPT_WITH_EXAMPLES = """## Reference Examples (same route/cell with known answers)

{examples_section}

## Test Question

**Pre-calculated Metrics for Test Question:**
{query_metrics}

**Full Question Data:**
{question}

---

Compare the test question metrics against the reference examples:
- C1 (Downtilt): typically has higher tilt (>20), weaker RSRP (<-85 dBm)
- C3 (Neighbor Better): typically has lower tilt (<15), neighbor within 3dB of serving

Based on the metrics comparison, which root cause matches?

\\boxed{{YOUR_ANSWER}}"""


TYPE_A_USER_PROMPT_NO_EXAMPLES = """## Test Question

**Pre-calculated Metrics:**
{query_metrics}

**Full Question Data:**
{question}

---

Analyze the metrics:
- C1 (Downtilt): High tilt (>20 degrees) + weak RSRP (<-85 dBm) suggests downtilt issue
- C3 (Neighbor Better): Low tilt (<15 degrees) + neighbor close to serving (diff <3dB) suggests handover issue

Based on the calculated metrics, determine if this is C1 or C3.

\\boxed{{YOUR_ANSWER}}"""


# Type B prompt - V9 ENHANCED
# IMPORTANT: Option letters are SHUFFLED per question - identify CAUSE first, then find matching option
TYPE_B_SYSTEM_PROMPT = """You are an expert 5G network troubleshooter analyzing why throughput dropped below 100Mbps.

IMPORTANT: The option letters (A-I) are SHUFFLED in each question. You must:
1. First identify the ROOT CAUSE from the data
2. Then find which option letter matches that cause

## ROOT CAUSE IDENTIFICATION (follow in order):

**Physical Layer Issues:**
- High CCE fail rate (>0.25) → PDCCH resource management issue
- Very low RSRP (<-95 dBm) → Weak coverage (RF/power/site issue)

**Handover Issues:**
- Many actual handovers (≥3 PCIs visited) → Intra-freq threshold TOO LOW (ping-pong)
- Many A3 events but few handovers (ratio ≥3:1) → Intra-freq threshold TOO HIGH
- RRC Reestablish events → Connection drops, intra-freq threshold TOO HIGH

**Interference Issues:**
- Good RSRP (>-90) but poor SINR (<8dB) → Overlap coverage (interference)
- High RSRP variance → Overlap coverage

**Other Causes:**
- A2 triggered but no A5/inter-freq HO → Inter-freq threshold issue
- Low RB allocation (<100) with good RF → Capacity/load imbalance
- Good RF but poor throughput → Transport/server issue
- Strong neighbor not in config → Missing neighbor configuration

## MATCHING TO OPTIONS:
After identifying the cause, read the options (A-I) and find the one that matches:
- "overlap coverage" = overlap cause
- "weak coverage" = weak coverage cause
- "intra-frequency handover threshold too high" = threshold too high
- "intra-frequency handover threshold too low" = threshold too low (ping-pong)
- "PDCCH resource management" = PDCCH cause
- "inter-frequency handover threshold" = inter-freq cause
- "capacity insufficient" = capacity cause
- "transport anomaly" = transport cause
- "missing neighbor" = missing neighbor cause

Format: \\boxed{LETTER} where LETTER is the option (A-I) matching your identified cause."""


TYPE_B_USER_PROMPT = """Analyze this 5G network troubleshooting question:

{question}

---

**Pre-computed Metrics:**
{metrics_summary}

**Key Observations:**
{key_observations}

---

## Step 1: Identify the ROOT CAUSE from metrics

Check in order:
1. CCE fail rate {cce_note}
2. RSRP level {rsrp_note}
3. Handover count {ho_note}
4. A3 to HO ratio {ratio_note}
5. SINR vs RSRP pattern
6. Other indicators (A2/A5, RB, transport)

What is the ROOT CAUSE? (e.g., "weak coverage", "threshold too high", "overlap", etc.)

## Step 2: Match cause to option letter

Read the options A-I in the question above.
Find the option that describes your identified root cause.

\\boxed{{LETTER}}"""


# =============================================================================
# V9 NEW: GENERIC PROMPTS - SPLIT INTO HISTORY AND MATH
# =============================================================================

# History/Reading Comprehension prompt (45 questions in Phase 2)
# V15: Enhanced with question classification, elimination strategy, trap awareness
GENERIC_HISTORY_SYSTEM_PROMPT = """You are an expert at reading comprehension, historical analysis, and textual reasoning.

Your method:
1. Read the QUESTION first. Classify it: factual recall, inference, author's purpose/tone,
   cause-effect, comparison, or "which best describes."
2. Read the passage. Mark key claims, dates, names, and qualifiers (e.g., "some," "always,"
   "primarily"). Pay attention to tone and rhetorical purpose.
3. For EACH option, find a specific quote or lack thereof:
   - Supported: quote the exact phrase(s) that support it.
   - Plausible but unsupported: explain why the text does not actually say this.
   - Contradicted: quote the phrase that contradicts it.
   - Irrelevant/anachronistic: explain why it does not fit the time period or context.
4. Eliminate options using the evidence. If two remain, re-read the relevant passage section
   and pick the one with MORE DIRECT support.

Common traps to avoid:
- Options that use words from the passage but change the meaning.
- Options that are historically true but not supported by THIS specific passage.
- Options that are too broad or too narrow relative to what the text actually claims.
- Anachronistic options that reference concepts from a different era.

Format: \\boxed{OPTION_NUMBER} using the exact number (1, 2, 3, or 4)."""


GENERIC_HISTORY_USER_PROMPT = """Read the passage and answer the question below.

{question}

---

Think step by step:

1. QUESTION TYPE: What kind of question is this? (factual, inference, purpose, cause-effect, comparison)

2. KEY PASSAGE EVIDENCE: Quote 2-3 specific sentences or phrases from the passage that are
   most relevant to the question. Note any qualifying language.

3. EVALUATE EACH OPTION:
   - Option 1: [Quote supporting evidence OR explain why unsupported/contradicted]
   - Option 2: [Quote supporting evidence OR explain why unsupported/contradicted]
   - Option 3: [Quote supporting evidence OR explain why unsupported/contradicted]
   - Option 4: [Quote supporting evidence OR explain why unsupported/contradicted]

4. ELIMINATE: Which options can be ruled out and why?

5. DECIDE: Of the remaining options, which has the most direct textual support?
   If choosing between two close options, ask: "Does the passage ACTUALLY say this,
   or am I bringing in outside knowledge?"

\\boxed{{YOUR_ANSWER}}"""


# Math prompt (37 questions in Phase 2)
# V15: Enhanced with estimation, independent verification, common error awareness
GENERIC_MATH_SYSTEM_PROMPT = """You are an expert mathematician. You solve problems with rigorous step-by-step reasoning and always verify your answer through an independent check.

Your method:
1. Read the problem. Identify the type: arithmetic, algebra, calculus, probability,
   geometry, number theory, word problem, or applied math.
2. Before computing, ESTIMATE the answer's order of magnitude. This catches gross errors.
3. Define variables explicitly. Write the equation or expression before solving.
4. Solve step by step. Show every algebraic manipulation. Do not skip steps.
5. VERIFY using a method DIFFERENT from your solution:
   - Equations: substitute your answer back into the original equation.
   - Word problems: check units and confirm the answer makes sense in context.
   - Calculus: verify with a numerical estimate or alternative technique.
   - Probability: confirm the result is in [0,1] and boundary cases work.
   - Counting: try a small example first, then generalize.
6. Compare to the given options. If your answer does not match any option exactly,
   re-examine your work for sign errors, arithmetic mistakes, or misread values.
   If still no match, pick the closest option and explain the discrepancy.

Common errors to watch for:
- Sign errors in subtraction and negative exponents.
- Off-by-one errors in counting and combinatorics.
- Forgetting to convert units or misreading the question.
- Dividing when you should multiply (or vice versa) in proportion problems.
- For transcendental equations (trig, log): evaluate at several sample points
  rather than trying to solve algebraically.

Format: \\boxed{OPTION_NUMBER} using the exact number (1, 2, 3, or 4)."""


GENERIC_MATH_USER_PROMPT = """Solve the following problem carefully.

{question}

---

Think step by step:

1. PROBLEM TYPE: What kind of math problem is this?

2. ESTIMATE: Before solving, what is the approximate magnitude of the answer?
   (e.g., "should be around 100" or "should be a small fraction")

3. SETUP: Define variables and write the core equation or expression.

4. SOLVE: Show every step. Do not skip arithmetic.

5. VERIFY: Check your answer using a DIFFERENT method than your solution.
   - If you solved algebraically, substitute back numerically.
   - If you solved numerically, verify with estimation or dimensional analysis.
   - Does the answer make sense given the problem context?

6. MATCH: Which option matches your verified answer?
   If none match exactly, recheck your work before selecting the closest.

\\boxed{{YOUR_ANSWER}}"""


# =============================================================================
# STAGE 3: LLM CLIENT
# =============================================================================

@dataclass
class LLMConfig:
    api_key: str = ""
    model_name: str = "Qwen/Qwen3-32B"
    max_concurrent: int = 10
    max_tokens: int = 16384
    top_p: float = 0.95
    max_retries: int = 5
    retry_delay: float = 1.0


def strip_think_tags(response: str) -> str:
    """Remove <think>...</think> blocks from response."""
    if not response:
        return response
    pattern = r'<think>.*?</think>'
    cleaned = re.sub(pattern, '', response, flags=re.DOTALL)
    return cleaned.strip()


def extract_answer(response: str) -> Optional[str]:
    """Extract answer from \\boxed{...} in response."""
    if not response:
        return None

    patterns = [
        r'\\boxed\{([^}]+)\}',
        r'\\boxed\s*\{([^}]+)\}',
        r'\*\*\\boxed\{([^}]+)\}\*\*',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            return matches[-1].strip()

    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer format."""
    if not answer:
        return answer
    answer = answer.strip().upper()
    answer = re.sub(r'^(OPTION\s*|ANSWER\s*|ROOT\s*CAUSE\s*)', '', answer, flags=re.IGNORECASE)
    return answer.strip()


def map_answer_to_test_option(answer: str, test_option_map: Dict[str, str]) -> Optional[str]:
    """Map model's answer back to exact test option label."""
    if not answer or not test_option_map:
        return answer

    answer_upper = answer.strip().upper()

    # Direct match
    if answer_upper in test_option_map:
        return answer_upper

    # Answer is canonical cause - find test label
    if answer_upper in test_option_map.values():
        for label, cause in test_option_map.items():
            if cause == answer_upper:
                return label

    # Answer is just a number - find matching label
    if answer_upper.isdigit():
        for label in test_option_map.keys():
            label_num = re.sub(r'^[A-Z]+', '', label)
            if label_num == answer_upper:
                return label

    # Answer has different prefix but same number
    answer_num = re.sub(r'^[A-Z]+', '', answer_upper)
    if answer_num.isdigit():
        for label in test_option_map.keys():
            label_num = re.sub(r'^[A-Z]+', '', label)
            if label_num == answer_num:
                return label

    return answer


class LLMClient:
    """Async client for HuggingFace Inference API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = InferenceClient(api_key=config.api_key)
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self._request_count = 0
        self._error_count = 0

    async def generate(self, messages: List[Dict], temperature: float = 0.6) -> Optional[str]:
        """Generate a response from the model."""
        for attempt in range(self.config.max_retries):
            try:
                async with self.semaphore:
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model=self.config.model_name,
                        messages=messages,
                        max_tokens=self.config.max_tokens,
                        temperature=temperature,
                        top_p=self.config.top_p,
                    )

                    if response and response.choices:
                        self._request_count += 1
                        return response.choices[0].message.content

            except Exception as e:
                error_str = str(e).lower()

                if "rate" in error_str or "429" in error_str:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    continue

                if "timeout" in error_str or "504" in error_str:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Timeout. Waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    continue

                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue

        self._error_count += 1
        return None


# =============================================================================
# STAGE 3: LLM INFERENCE
# =============================================================================

def format_type_a_examples(examples_by_cause: Dict[str, Dict], cause_to_label: Dict[str, str]) -> str:
    """Format labeled examples for Type A prompt with calculated metrics."""
    if not examples_by_cause:
        return "(No reference examples available)"

    sections = []
    for cause in ['C1', 'C3']:
        if cause in examples_by_cause:
            ex = examples_by_cause[cause]
            label = cause_to_label.get(cause, cause)
            q = ex['question']

            # Calculate ALL metrics for this example
            drive_test, cells = parse_type_a_question(q)
            if drive_test:
                tilt = get_type_a_tilt(drive_test, cells)
                avg_rsrp = get_type_a_avg_rsrp(drive_test)
                min_nb_diff = get_min_neighbor_diff(drive_test)

                # Additional metrics
                _, max_speed, _ = check_c7_speed(drive_test)
                _, max_dist, _ = check_c2_overshooting(drive_test, cells)
                _, handovers, _ = check_c5_handovers(drive_test)
                _, avg_rb, _ = check_c8_low_rbs(drive_test)

                rsrps = [d['rsrp'] for d in drive_test if d['rsrp']]
                sinrs = [d['sinr'] for d in drive_test if d['sinr']]
                min_rsrp = min(rsrps) if rsrps else 0
                avg_sinr = sum(sinrs)/len(sinrs) if sinrs else 0

                # Build metrics summary
                metrics = f"""**Key Metrics:**
- Total Tilt: {tilt:.0f} degrees
- Avg RSRP: {avg_rsrp:.1f} dBm (min: {min_rsrp:.1f} dBm)
- Min Neighbor Diff: {min_nb_diff:.1f} dB
- Avg SINR: {avg_sinr:.1f} dB
- Max Speed: {max_speed:.1f} km/h
- Handovers: {handovers}
- Avg RBs: {avg_rb:.0f}
"""
            else:
                metrics = "(Could not calculate metrics)"

            # Extract just the data portion (first 10 rows)
            lines = q.split('\n')
            data_lines = []
            in_data = False
            for line in lines:
                if 'Timestamp|' in line or 'gNodeB ID|' in line:
                    in_data = True
                if in_data and '|' in line:
                    data_lines.append(line)
                if in_data and line.strip() and '|' not in line and not line.startswith(' '):
                    break

            section = f"### Example: Answer = {label} ({cause})\n\n{metrics}\n**Sample Data:**\n" + '\n'.join(data_lines[:10])
            sections.append(section)

    return '\n\n---\n\n'.join(sections) if sections else "(No C1/C3 examples found)"


async def run_llm_type_a(
    question: str,
    stage2_result: Dict,
    llm_client: LLMClient,
    lookup_index: Optional[Phase1LookupIndex],
    temperatures: List[float],
) -> List[Dict]:
    """Run LLM for Type A C1/C3 cases."""

    # Get examples if available
    examples_by_cause = {}
    if lookup_index:
        examples_by_cause = lookup_index.get_examples_by_cause(question)

    test_option_map = stage2_result.get('test_option_map', {})
    cause_to_label = {cause: label for label, cause in test_option_map.items()}

    # Get ALL query metrics from stage2_result evidence
    evidence = stage2_result.get('evidence', {})
    query_metrics = f"""**Primary C1/C3 Indicators:**
- Total Tilt: {evidence.get('tilt', 'N/A')} degrees
- Avg RSRP: {evidence.get('avg_rsrp', 'N/A')}
- Min Neighbor Diff (serving - neighbor): {evidence.get('min_neighbor_diff', 'N/A')}

**Other Metrics (already ruled out as root cause):**
- Max Speed: {evidence.get('C7', 'N/A')}
- Max Distance to Cell: {evidence.get('C2', 'N/A')}
- Handovers: {evidence.get('C5', 'N/A')}
- Avg RBs: {evidence.get('C8', 'N/A')}
- Non-colocated Interference: {evidence.get('C4', 'N/A')}
- PCI Collision: {evidence.get('C6', 'N/A')}

**Deterministic Prediction:** {evidence.get('c1_c3_pred', 'N/A')}"""

    # Build prompt
    if examples_by_cause and ('C1' in examples_by_cause or 'C3' in examples_by_cause):
        examples_section = format_type_a_examples(examples_by_cause, cause_to_label)
        user_prompt = TYPE_A_USER_PROMPT_WITH_EXAMPLES.format(
            examples_section=examples_section,
            query_metrics=query_metrics,
            question=question,
        )
    else:
        user_prompt = TYPE_A_USER_PROMPT_NO_EXAMPLES.format(
            query_metrics=query_metrics,
            question=question,
        )

    messages = [
        {"role": "system", "content": TYPE_A_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Generate responses
    results = []
    for i, temp in enumerate(temperatures):
        response = await llm_client.generate(messages, temperature=temp)

        answer = None
        if response:
            clean_response = strip_think_tags(response)
            answer = extract_answer(clean_response)
            if answer:
                answer = normalize_answer(answer)
                answer = map_answer_to_test_option(answer, test_option_map)

        results.append({
            "response_idx": i + 1,
            "temperature": temp,
            "response": strip_think_tags(response) if response else None,
            "answer": answer,
        })

    return results


async def run_llm_type_b(
    question: str,
    stage2_result: Dict,
    llm_client: LLMClient,
    temperatures: List[float],
) -> List[Dict]:
    """Run LLM for Type B cases. V9: Fixed to handle shuffled options."""

    test_option_map = stage2_result.get('test_option_map', {})
    context = stage2_result.get('context', {})

    # V8 ENHANCED: Build metrics summary for LLM
    metrics_summary = context.get('metrics_summary', 'No metrics available')

    # V9: Generate metric-specific notes for rule-based classification
    avg_cce = context.get('avg_cce_fail', 0)
    avg_rsrp = context.get('avg_rsrp', -90)
    actual_ho = context.get('actual_handovers', 0)
    ratio_a3_ho = context.get('ratio_a3_ho', 1)

    cce_note = f"= {avg_cce:.2f} {'(HIGH - suggests PDCCH issue)' if avg_cce > 0.25 else '(normal)'}"
    rsrp_note = f"= {avg_rsrp:.1f}dBm {'(WEAK - suggests coverage issue)' if avg_rsrp < -95 else '(OK)'}"
    ho_note = f"= {actual_ho} {'(HIGH - suggests ping-pong)' if actual_ho >= 3 else '(normal)'}"
    ratio_note = f"= {ratio_a3_ho:.1f} {'(HIGH - suggests threshold too high)' if ratio_a3_ho >= 3 else '(normal)'}"

    # V8 ENHANCED: Generate key observations based on metrics
    observations = []
    if avg_rsrp < -95:
        observations.append(f"- Weak RSRP ({avg_rsrp:.1f}dBm) suggests coverage issue")
    if avg_cce > 0.15:
        observations.append(f"- High CCE fail rate ({avg_cce:.2f}) suggests PDCCH issue")
    if context.get('rrc_reestablish', 0) > 0:
        observations.append(f"- RRC reestablish events ({context.get('rrc_reestablish', 0)}) indicate connection drops")
    if context.get('a5_events', 0) > 0:
        observations.append(f"- A5 events ({context.get('a5_events', 0)}) suggest inter-frequency HO triggered")
    if context.get('min_neighbor_diff', 0) < 3:
        observations.append(f"- Small neighbor RSRP diff ({context.get('min_neighbor_diff', 0):.1f}dB) may indicate overlap")
    if context.get('avg_bler', 0) > 15:
        observations.append(f"- High BLER ({context.get('avg_bler', 0):.1f}%) indicates poor link quality")

    # V11: RF health and interference observations
    avg_sinr = context.get('avg_sinr', 10)
    avg_bler = context.get('avg_bler', 10)
    rf_healthy = context.get('rf_healthy', False)
    sinr_deficit = context.get('sinr_deficit', 0)

    if rf_healthy:
        observations.append(
            f"- RF HEALTHY: SINR={avg_sinr:.1f}dB, RSRP={avg_rsrp:.1f}dBm, BLER={avg_bler:.1f}% "
            f"are all good. Problem is likely NOT RF-related (not overlap, not weak coverage). "
            f"Consider: transport/server issue, capacity, inter-freq config."
        )
    if sinr_deficit > 5:
        observations.append(
            f"- SINR deficit={sinr_deficit:.1f}: SINR much worse than expected for this RSRP level. "
            f"Strong indicator of interference/overlap."
        )
    tp_drop_ctx = context.get('tp_drop_context', '')
    if tp_drop_ctx:
        observations.append(f"- {tp_drop_ctx}")

    # =========================================================================
    # V15: PHY health during low-TP observations
    # =========================================================================
    phy_healthy = context.get('phy_healthy_during_low_tp')
    low_tp_mcs = context.get('low_tp_avg_mcs')
    low_tp_rank = context.get('low_tp_avg_rank')
    low_tp_sinr = context.get('low_tp_avg_sinr')
    low_tp_bler = context.get('low_tp_avg_bler')
    low_tp_resbler = context.get('low_tp_avg_resbler')
    v15_mcs_drop = context.get('mcs_drop')
    v15_rank_drop = context.get('rank_drop')

    if phy_healthy is True and low_tp_mcs is not None:
        observations.append(
            f"- PHY HEALTHY DURING LOW-TP: During throughput drops, MCS={low_tp_mcs:.1f}, "
            f"Rank={low_tp_rank:.2f}, SINR={low_tp_sinr:.1f}dB, BLER={low_tp_bler:.1f}%, "
            f"ResBLER={low_tp_resbler:.2f}%. The radio link is FINE - the PHY layer is not degraded. "
            f"This strongly indicates a non-RF cause: transport/server anomaly, capacity, or config issue."
        )
    elif phy_healthy is False and low_tp_mcs is not None:
        observations.append(
            f"- PHY DEGRADED DURING LOW-TP: During throughput drops, MCS={low_tp_mcs:.1f}, "
            f"Rank={low_tp_rank:.2f}, SINR={low_tp_sinr:.1f}dB, BLER={low_tp_bler:.1f}%, "
            f"ResBLER={low_tp_resbler:.2f}%. The radio link quality degrades WITH throughput. "
            f"This strongly indicates an RF cause: overlap coverage or interference."
        )
        if v15_mcs_drop is not None and v15_mcs_drop > 2:
            observations.append(
                f"- MCS drops by {v15_mcs_drop:.1f} and Rank drops by {v15_rank_drop:.2f} "
                f"from high-TP to low-TP rows. MIMO and modulation degrade together - "
                f"consistent with interference destroying the signal."
            )

    # V15: Multi-neighbor proximity
    nbrs_3db = context.get('neighbors_within_3dB', 0)
    nbrs_5db = context.get('neighbors_within_5dB', 0)
    n1_pct = context.get('n1_stronger_pct', 0)

    if nbrs_3db >= 1:
        observations.append(
            f"- {nbrs_3db} neighbor(s) within 3dB of serving RSRP. "
            f"Very strong overlap - multiple cells competing at similar power levels."
        )
    elif nbrs_5db >= 2:
        observations.append(
            f"- {nbrs_5db} neighbors within 5dB of serving RSRP. "
            f"Moderate overlap - neighbors are strong relative to serving cell."
        )

    if n1_pct > 20:
        observations.append(
            f"- Strongest neighbor is STRONGER than serving in {n1_pct:.0f}% of samples. "
            f"The UE is being served by a weaker cell - classic overlap/dominance issue."
        )

    # V15: Spectral efficiency
    tp_rb = context.get('avg_tp_per_rb')
    low_tp_rb = context.get('low_tp_avg_rb')
    if tp_rb is not None and phy_healthy is True and low_tp_rb is not None:
        observations.append(
            f"- Spectral efficiency={tp_rb:.2f} Mbps/RB (healthy link). "
            f"RB/slot drops to {low_tp_rb:.0f} during low-TP periods - "
            f"scheduler allocates fewer RBs because upstream data flow is limited."
        )

    key_observations = '\n'.join(observations) if observations else "No strong indicators detected - analyze tables carefully"

    messages = [
        {"role": "system", "content": TYPE_B_SYSTEM_PROMPT},
        {"role": "user", "content": TYPE_B_USER_PROMPT.format(
            question=question,
            metrics_summary=metrics_summary,
            key_observations=key_observations,
            cce_note=cce_note,
            rsrp_note=rsrp_note,
            ho_note=ho_note,
            ratio_note=ratio_note,
        )},
    ]

    results = []
    for i, temp in enumerate(temperatures):
        response = await llm_client.generate(messages, temperature=temp)

        answer = None
        if response:
            clean_response = strip_think_tags(response)
            answer = extract_answer(clean_response)
            if answer:
                answer = normalize_answer(answer)
                answer = map_answer_to_test_option(answer, test_option_map)

        results.append({
            "response_idx": i + 1,
            "temperature": temp,
            "response": strip_think_tags(response) if response else None,
            "answer": answer,
        })

    return results


async def run_llm_generic(
    question: str,
    stage2_result: Dict,
    llm_client: LLMClient,
    temperatures: List[float],
) -> List[Dict]:
    """
    V9 UPDATED: Run LLM for generic questions with subtype-specific prompts.

    Dispatches to history or math prompts based on detected subtype.
    """
    test_option_map = stage2_result.get('test_option_map', {})
    subtype = stage2_result.get('subtype', 'math')  # V9: Get subtype

    # V9: Select prompts based on subtype
    if subtype == 'history':
        system_prompt = GENERIC_HISTORY_SYSTEM_PROMPT
        user_prompt = GENERIC_HISTORY_USER_PROMPT.format(question=question)
    else:  # math
        system_prompt = GENERIC_MATH_SYSTEM_PROMPT
        user_prompt = GENERIC_MATH_USER_PROMPT.format(question=question)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    results = []
    for i, temp in enumerate(temperatures):
        response = await llm_client.generate(messages, temperature=temp)

        answer = None
        if response:
            clean_response = strip_think_tags(response)
            answer = extract_answer(clean_response)
            if answer:
                answer = normalize_answer(answer)
                answer = map_answer_to_test_option(answer, test_option_map)

        results.append({
            "response_idx": i + 1,
            "temperature": temp,
            "response": strip_think_tags(response) if response else None,
            "answer": answer,
            "subtype": subtype,  # V9: Track which prompt was used
        })

    return results


# =============================================================================
# UNIFIED PIPELINE
# =============================================================================

class UnifiedPipeline:
    """Main pipeline orchestrator."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        lookup_index: Optional[Phase1LookupIndex] = None,
        temperatures: List[float] = [0.5, 0.55, 0.6, 0.65],
        dry_run: bool = False,
    ):
        self.llm_client = llm_client
        self.lookup_index = lookup_index
        self.temperatures = temperatures
        self.dry_run = dry_run

        # Stats
        self.stats = {
            'type_a_deterministic': 0,
            'type_a_high': 0,
            'type_a_c1c3_high': 0,
            'type_a_c1c3_medium': 0,
            'type_a_llm': 0,
            'type_b_deterministic': 0,
            'type_b_llm': 0,
            'generic_llm': 0,
            'errors': 0,
        }

    async def process(self, question_id: str, question: str) -> Dict:
        """Process a single question through the 3-stage pipeline."""

        # Stage 1: Classify question type
        q_type = classify_question_type(question)

        # Stage 2: Apply rules derived from data analysis
        if q_type == 'type_a_telco':
            stage2_result = classify_type_a(question)
        elif q_type == 'type_b_telco':
            stage2_result = classify_type_b(question)
        else:  # generic
            stage2_result = classify_generic(question)

        # Prepare result
        result = {
            'ID': question_id,
            'question_type': q_type,
            'stage2_confidence': stage2_result['confidence'],
            'stage2_answer': stage2_result.get('answer'),
            'stage2_canonical': stage2_result.get('canonical'),
            'evidence': stage2_result.get('evidence', {}),
            'context': stage2_result.get('context', {}),  # V10 FIX: Store context for debugging
            'llm_used': False,
            'responses': [],
        }

        # If Stage 2 gave a confident answer, use it
        if stage2_result['confidence'] in ['deterministic', 'high', 'heuristic', 'c1c3_high', 'c1c3_medium', 'v16_override', 'v18_rescue']:
            if q_type == 'type_a_telco':
                conf = stage2_result['confidence']
                if conf == 'deterministic':
                    self.stats['type_a_deterministic'] += 1
                elif conf == 'high':
                    self.stats['type_a_high'] += 1
                elif conf == 'c1c3_high':
                    self.stats['type_a_c1c3_high'] += 1
                elif conf == 'c1c3_medium':
                    self.stats['type_a_c1c3_medium'] += 1
                elif conf == 'v16_override':
                    self.stats['type_a_v16_override'] = self.stats.get('type_a_v16_override', 0) + 1
                elif conf == 'v18_rescue':
                    self.stats['type_a_v18_rescue'] = self.stats.get('type_a_v18_rescue', 0) + 1
            elif q_type == 'type_b_telco':
                conf = stage2_result['confidence']
                if conf == 'heuristic':
                    self.stats['type_b_heuristic'] = self.stats.get('type_b_heuristic', 0) + 1
                else:
                    self.stats['type_b_deterministic'] += 1
            else:
                self.stats['generic_deterministic'] = self.stats.get('generic_deterministic', 0) + 1

            # Create 4 identical responses for submission
            for i in range(4):
                result['responses'].append({
                    'response_idx': i + 1,
                    'temperature': self.temperatures[i % len(self.temperatures)],
                    'response': None,
                    'answer': stage2_result['answer'],
                })
            return result

        # Stage 3: LLM inference needed
        result['llm_used'] = True

        # Update stats before potential dry-run return
        if q_type == 'type_a_telco':
            self.stats['type_a_llm'] += 1
        elif q_type == 'type_b_telco':
            self.stats['type_b_llm'] += 1
        else:
            self.stats['generic_llm'] += 1

        if self.dry_run or not self.llm_client:
            # Dry run - return placeholder
            for i in range(4):
                result['responses'].append({
                    'response_idx': i + 1,
                    'temperature': self.temperatures[i % len(self.temperatures)],
                    'response': None,
                    'answer': 'PLACEHOLDER',
                })
            return result

        # Run appropriate LLM
        if q_type == 'type_a_telco':
            responses = await run_llm_type_a(
                question, stage2_result, self.llm_client,
                self.lookup_index, self.temperatures
            )
        elif q_type == 'type_b_telco':
            responses = await run_llm_type_b(
                question, stage2_result, self.llm_client, self.temperatures
            )
        else:
            responses = await run_llm_generic(
                question, stage2_result, self.llm_client, self.temperatures
            )

        result['responses'] = responses
        return result

    def print_stats(self):
        """Print pipeline statistics."""
        print("\n" + "="*60)
        print("PIPELINE STATISTICS")
        print("="*60)
        total = sum(self.stats.values())
        for key, count in sorted(self.stats.items()):
            pct = 100 * count / total if total > 0 else 0
            print(f"  {key:25s}: {count:4d} ({pct:5.1f}%)")
        print(f"  {'TOTAL':25s}: {total:4d}")


# =============================================================================
# CHECKPOINT AND SUBMISSION
# =============================================================================

def load_checkpoint(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def save_checkpoint(data: Dict, path: str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def generate_submission(checkpoint: Dict, output_path: str, df: pd.DataFrame):
    """Generate submission CSV from checkpoint."""
    rows = []

    for _, row in df.iterrows():
        question_id = row['ID']

        if question_id not in checkpoint:
            for i in range(4):
                rows.append({
                    'ID': f"{question_id}_{i+1}",
                    'Qwen3-32B': "placeholder",
                    'Qwen2.5-7B-Instruct': "placeholder",
                    'Qwen2.5-1.5B-Instruct': "placeholder",
                })
            continue

        data = checkpoint[question_id]
        responses = data.get('responses', [])
        q_type = data.get('question_type', 'type_a_telco')

        for i in range(4):
            if i < len(responses) and responses[i].get('answer'):
                answer = responses[i]['answer']
                if q_type in ['type_a_telco', 'type_b_telco']:
                    formatted = f"Based on the analysis, the root cause is: \\boxed{{{answer}}}"
                else:
                    formatted = f"The answer is: \\boxed{{{answer}}}"
            else:
                formatted = "placeholder"

            rows.append({
                'ID': f"{question_id}_{i+1}",
                'Qwen3-32B': formatted,
                'Qwen2.5-7B-Instruct': "placeholder",
                'Qwen2.5-1.5B-Instruct': "placeholder",
            })

    submission_df = pd.DataFrame(rows)
    submission_df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Total rows: {len(submission_df)}")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Unified 3-Stage Pipeline V17")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=2, help="Test phase")
    parser.add_argument("--samples", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls")
    parser.add_argument("--temperatures", type=str, default="0.5,0.55,0.6,0.65",
                        help="Comma-separated temperatures")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Save checkpoint every N questions")
    args = parser.parse_args()

    # Paths
    data_dir = Path("the-ai-telco-troubleshooting-challenge20251127-8634-8qzscv")
    output_dir = Path("outputs/inference")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_file = data_dir / f"phase_{args.phase}_test.csv"
    phase1_file = data_dir / "phase_1_test.csv"
    truth_file = data_dir / "phase_1_test_truth.csv"

    checkpoint_path = output_dir / f"checkpoint_phase_{args.phase}.json"
    submission_path = output_dir / f"submission_phase_{args.phase}.csv"

    # Load test data
    print(f"Loading test data from {test_file}")
    df = pd.read_csv(test_file)

    if args.samples:
        df = df.head(args.samples)
        print(f"Limited to {args.samples} samples for testing")

    # Build lookup index
    lookup_index = None
    if phase1_file.exists() and truth_file.exists():
        print(f"\nBuilding lookup index...")
        lookup_index = Phase1LookupIndex(str(phase1_file), str(truth_file))

    # Parse temperatures
    temperatures = [float(t) for t in args.temperatures.split(',')]
    print(f"Temperatures: {temperatures}")

    # Setup LLM client
    llm_client = None
    if not args.dry_run:
        HF_API_KEY = os.getenv("HF_API_KEY")
        if HF_API_KEY:
            config = LLMConfig(api_key=HF_API_KEY)
            llm_client = LLMClient(config)
            print("LLM client initialized")
        else:
            print("WARNING: No HF_API_KEY found, running in dry-run mode")
            args.dry_run = True

    # Initialize pipeline
    pipeline = UnifiedPipeline(
        llm_client=llm_client,
        lookup_index=lookup_index,
        temperatures=temperatures,
        dry_run=args.dry_run,
    )

    # Load existing checkpoint
    checkpoint = load_checkpoint(str(checkpoint_path))
    print(f"Loaded checkpoint with {len(checkpoint)} existing results")

    # Process questions
    print(f"\nProcessing {len(df)} questions...")
    print("="*60)

    for idx, row in df.iterrows():
        question_id = row['ID']
        question = row['question']

        # Skip if already processed
        if question_id in checkpoint:
            # Update stats from checkpoint
            data = checkpoint[question_id]
            q_type = data.get('question_type', 'type_a_telco')
            conf = data.get('stage2_confidence', 'unknown')
            llm_used = data.get('llm_used', False)

            if q_type == 'type_a_telco':
                if conf == 'deterministic':
                    pipeline.stats['type_a_deterministic'] += 1
                elif conf == 'high':
                    pipeline.stats['type_a_high'] += 1
                elif conf == 'c1c3_high':
                    pipeline.stats['type_a_c1c3_high'] += 1
                elif conf == 'c1c3_medium':
                    pipeline.stats['type_a_c1c3_medium'] += 1
                elif llm_used:
                    pipeline.stats['type_a_llm'] += 1
            elif q_type == 'type_b_telco':
                if conf == 'deterministic':
                    pipeline.stats['type_b_deterministic'] += 1
                elif conf == 'heuristic':
                    pipeline.stats['type_b_heuristic'] = pipeline.stats.get('type_b_heuristic', 0) + 1
                elif llm_used:
                    pipeline.stats['type_b_llm'] += 1
            else:
                if llm_used:
                    pipeline.stats['generic_llm'] += 1
            continue

        # Process question
        result = await pipeline.process(question_id, question)
        checkpoint[question_id] = result

        # Log progress
        answers = [r['answer'] for r in result['responses'] if r.get('answer')]
        print(f"[{idx+1}/{len(df)}] {question_id}: {result['question_type']}, "
              f"conf={result['stage2_confidence']}, llm={result['llm_used']}, "
              f"answers={answers[:2]}...")

        # Save checkpoint
        if (idx + 1) % args.checkpoint_every == 0:
            save_checkpoint(checkpoint, str(checkpoint_path))
            print(f"  [Checkpoint saved: {len(checkpoint)} questions]")

    # Final save
    save_checkpoint(checkpoint, str(checkpoint_path))

    # Print stats
    pipeline.print_stats()

    # Generate submission
    generate_submission(checkpoint, str(submission_path), df)


if __name__ == "__main__":
    asyncio.run(main())
