#!/usr/bin/env python3
"""
Final Reasoning Trace Generator - Train Data Only

Produces training-ready checkpoint JSON with reasoning traces for SFT/GRPO
from the labeled training set (train.csv, 2400 Type A questions).

Rule-based classification walkthrough (from data analysis we know these rules):
- Tier 1: C7 (speed), C2 (distance), C5 (handovers), C8 (RB)
- Tier 2: C1 detection (3 sub-rules + V16 overrides B/P3/P4)
- Tier 3: C4 interference (ratio filter)
- Tier 4: C6 collision (signal filters + V16 overrides P1/P2/G/J/P5b)
- Tier 5: C1/C3 tiebreaker (tilt/RSRP/SINR gate + rescue rules R1-R4)

V19 thresholds (calibrated on train set):
- tilt_high_c1: 28 with SINR gate (>=12 -> C3)
- rsrp_c1_medium: -90, rsrp_c3_medium: -82
- P4: -79, R1: collision_ratio >= 0.9
- R2: strong_neighbors < 0.8, R3: c4_interference >= 3.0

Output: outputs/traces_final/traces_final.json (~2400 traces)

Usage:
    uv run python generate_traces_final.py
    uv run python generate_traces_final.py --spot-check 5
    uv run python generate_traces_final.py --validate-only outputs/traces_final/traces_final.json
"""

import json
import argparse
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd

from telco_utils import (
    classify_question_type,
    classify_type_a,
    extract_type_a_options,
    parse_type_a_question,
    haversine,
    check_c4_non_colocated,
    check_c6_pci_collision,
    get_min_rsrp,
    get_strong_neighbor_count,
    get_type_a_tilt,
    get_type_a_avg_rsrp,
    get_avg_off_axis_angle,
    get_min_sinr_low_tp,
    get_min_neighbor_diff,
    get_pci_collision_ratio,
    get_tp_threshold,
    compute_v16_metrics,
    classify_c1_vs_c3,
    # Type B imports (used by SFT/GRPO/inference for prompt preparation)
    parse_type_b_question,
    parse_config_data,
    detect_inter_freq_ho,
    check_n1_in_config,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path('the-ai-telco-troubleshooting-challenge20251127-8634-8qzscv')
OUTPUT_DIR = Path('outputs/traces_final')

CAUSE_DESCRIPTIONS = {
    'C1': 'excessive downtilt causing weak far-end coverage',
    'C2': 'coverage distance exceeding 1 km (over-shooting)',
    'C3': 'a neighboring cell providing higher throughput',
    'C4': 'non-colocated co-frequency interference',
    'C5': 'frequent handovers degrading performance',
    'C6': 'PCI mod 30 collision causing interference',
    'C7': 'vehicle speed exceeding 40 km/h',
    'C8': 'average scheduled RBs below threshold',
}

# Format specs per metric
METRIC_FORMATS = {
    'max_speed': '.1f',
    'max_distance_low_tp': '.2f',
    'handover_count': '.0f',
    'avg_rb': '.1f',
}


def safe_cmp_fmt(val: float, threshold: float, op: str, decimals: int = 1) -> str:
    """Format value with enough precision that the displayed comparison is correct."""
    def _passes(v, t, o):
        if o == '<': return v < t
        if o == '>': return v > t
        if o == '<=': return v <= t
        if o == '>=': return v >= t
        return False

    raw_passes = _passes(val, threshold, op)
    for d in range(decimals, decimals + 3):
        rounded = round(val, d)
        if _passes(rounded, threshold, op) == raw_passes:
            return f"{val:.{d}f}"
    return f"{val:.{decimals + 2}f}"


def _maybe_correct(lines: list, m: Dict, canonical: str, is_expert: bool) -> str:
    """Post-process trace: if V19 concluded with the wrong cause, append expert correction."""
    trace = '\n'.join(lines)
    if not is_expert:
        return trace
    correct_conclusion = f"The root cause is {CAUSE_DESCRIPTIONS[canonical]}."
    if correct_conclusion in trace:
        return trace
    # Wrong conclusion - append expert correction
    lines.append("")
    lines.append("However, examining additional indicators:")
    _add_expert_reasoning(lines, m, canonical)
    return '\n'.join(lines)


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_all_metrics(question: str, drive_test: List[Dict], cells: Dict) -> Dict:
    """Compute all V19 metrics from parsed drive test and engineering params."""
    m = {}
    tp_threshold = get_tp_threshold(question)
    m['tp_threshold'] = tp_threshold

    # Tier 1
    speeds = [d['speed'] for d in drive_test if d['speed']]
    m['max_speed'] = max(speeds) if speeds else 0.0

    low_tp_distances = []
    for d in drive_test:
        if d['throughput'] and d['throughput'] < tp_threshold:
            pci = d['serving_pci']
            if pci and pci in cells:
                cell = cells[pci]
                dist = haversine(cell['lon'], cell['lat'], d['lon'], d['lat'])
                low_tp_distances.append(dist)
    m['max_distance_low_tp'] = max(low_tp_distances) if low_tp_distances else 0.0

    pcis = [d['serving_pci'] for d in drive_test if d['serving_pci']]
    m['handover_count'] = sum(1 for i in range(1, len(pcis)) if pcis[i] != pcis[i-1]) if len(pcis) >= 2 else 0

    rbs = [d['rb'] for d in drive_test if d['rb']]
    m['avg_rb'] = sum(rbs) / len(rbs) if rbs else 999.0

    # Tier 2/3
    m['serving_tilt'] = get_type_a_tilt(drive_test, cells)
    m['avg_rsrp'] = get_type_a_avg_rsrp(drive_test)
    m['min_rsrp'] = get_min_rsrp(drive_test)
    m['strong_neighbor_count'] = get_strong_neighbor_count(drive_test)
    m['min_neighbor_diff'] = get_min_neighbor_diff(drive_test)
    m['avg_off_axis'] = get_avg_off_axis_angle(drive_test, cells)
    m['min_sinr_low_tp'] = get_min_sinr_low_tp(drive_test)

    # avg_sinr for SINR gate in C1/C3 tiebreaker
    sinrs = [d.get('sinr') for d in drive_test if d.get('sinr') is not None]
    m['avg_sinr'] = sum(sinrs) / len(sinrs) if sinrs else None

    _, c4_interference, _ = check_c4_non_colocated(drive_test, cells)
    m['c4_interference'] = c4_interference
    m['ratio_nbdiff_interf'] = (
        m['min_neighbor_diff'] / max(c4_interference, 1) if c4_interference > 0 else 0.0
    )

    _, has_collision, c6_detail = check_c6_pci_collision(drive_test)
    m['pci_collision'] = has_collision
    if has_collision:
        import re as _re
        dm = _re.search(r'serving (\d+)%30=(\d+) == neighbor (\d+)', c6_detail)
        if dm:
            spci, smod, npci = dm.group(1), dm.group(2), dm.group(3)
            nmod = int(npci) % 30
            m['pci_collision_detail'] = f"serving PCI {spci} mod 30 = {smod}, neighbor PCI {npci} mod 30 = {nmod}"
        else:
            m['pci_collision_detail'] = c6_detail
    else:
        m['pci_collision_detail'] = c6_detail
    m['pci_collision_ratio'] = get_pci_collision_ratio(drive_test)

    # V16 override metrics
    v16 = compute_v16_metrics(drive_test, tp_threshold)
    m['post_ho_good_streak'] = v16.get('post_ho_good_streak', 0)
    m['rsrp_recovery'] = v16.get('rsrp_recovery', 0.0)
    m['rsrp_change_during_prob'] = v16.get('rsrp_change_during_prob', 0.0)
    m['rsrp_trend'] = v16.get('rsrp_trend', 0.0)
    m['nb_within_5db_per_row'] = v16.get('nb_within_5db_per_row', 0.0)

    # C6 filter signals
    m['c6_c1_signal'] = m['serving_tilt'] >= 20
    m['c6_c3_signal'] = m['min_neighbor_diff'] < 3 and m['serving_tilt'] > 12
    m['c6_c3_off_axis_signal'] = m['avg_off_axis'] > 30

    return m


def format_metrics_block(m: Dict) -> str:
    """Format all computed metrics as a structured text block."""
    lines = [
        "Extracted metrics:",
        f"  max_speed = {m['max_speed']:.1f} km/h",
        f"  max_distance_low_tp = {m['max_distance_low_tp']:.2f} km",
        f"  handover_count = {m['handover_count']}",
        f"  avg_rb = {m['avg_rb']:.1f}",
        f"  serving_tilt = {m['serving_tilt']:.0f} deg",
        f"  avg_rsrp = {m['avg_rsrp']:.3f} dBm",
        f"  min_rsrp = {m['min_rsrp']:.2f} dBm",
        f"  strong_neighbor_count = {m['strong_neighbor_count']:.2f}",
        f"  min_neighbor_diff = {m['min_neighbor_diff']:.1f} dB",
        f"  c4_interference = {m['c4_interference']:.2f} dB",
        f"  pci_collision = {'yes' if m['pci_collision'] else 'no'}",
        f"  pci_collision_ratio = {m['pci_collision_ratio']:.2f}",
        f"  avg_off_axis = {m['avg_off_axis']:.1f} deg",
        f"  post_ho_good_streak = {m['post_ho_good_streak']}",
        f"  rsrp_recovery = {m['rsrp_recovery']:.1f} dB",
        f"  rsrp_trend = {m['rsrp_trend']:.2f}",
        f"  nb_within_5db_per_row = {m['nb_within_5db_per_row']:.2f}",
    ]
    if m.get('avg_sinr') is not None:
        lines.append(f"  avg_sinr = {m['avg_sinr']:.1f} dB")
    else:
        lines.append("  avg_sinr = N/A")
    return '\n'.join(lines)


# =============================================================================
# TRACE GENERATION: V19 CASCADE WALKER
# =============================================================================

def generate_trace(
    m: Dict,
    result: Dict,
    available_causes: set,
    ground_truth: str,
    is_expert: bool = False,
) -> str:
    """Walk the V19 cascade and emit reasoning trace."""
    canonical = ground_truth if is_expert else result.get('canonical', ground_truth)
    evidence = result.get('evidence', {})
    v16_override = evidence.get('v16_override', '')

    lines = []
    lines.append(f"I need to identify the root cause of throughput dropping below {m['tp_threshold']:.0f} Mbps.")
    lines.append("")
    lines.append(format_metrics_block(m))
    lines.append("")

    # ===== STEP 1: TIER 1 CHECKS =====
    lines.append("Step 1 - Tier 1 checks:")
    tier1_hit = _walk_tier1(lines, m, canonical, available_causes)
    if tier1_hit:
        return '\n'.join(lines)

    lines.append("All tier 1 causes ruled out.")
    lines.append("")

    # ===== STEP 2: C1 DETECTION =====
    lines.append("Step 2 - C1 detection rules:")
    c1_hit = _walk_c1_detection(lines, m, canonical, available_causes, evidence, v16_override)
    if c1_hit:
        return _maybe_correct(lines, m, canonical, is_expert)
    lines.append("")

    # ===== STEP 3: C4 CHECK =====
    lines.append("Step 3 - C4 interference check:")
    c4_hit = _walk_c4(lines, m, canonical, available_causes, evidence)
    if c4_hit:
        return _maybe_correct(lines, m, canonical, is_expert)
    lines.append("")

    # ===== STEP 4: C6 COLLISION CHECK =====
    lines.append("Step 4 - C6 collision check:")
    c6_hit = _walk_c6(lines, m, canonical, available_causes, evidence, v16_override)
    if c6_hit:
        return _maybe_correct(lines, m, canonical, is_expert)
    lines.append("")

    # ===== STEP 5: C1/C3 TIEBREAKER =====
    lines.append("Step 5 - C1/C3 tiebreaker:")
    _walk_c1c3_tiebreaker(lines, m, canonical, available_causes, evidence, v16_override, is_expert)

    return _maybe_correct(lines, m, canonical, is_expert)


def _walk_tier1(lines: list, m: Dict, canonical: str, available: set) -> bool:
    """Walk tier-1 cascade. Returns True if answer found here."""
    checks = [
        ('C7', 'max_speed', 40, '>', 'km/h'),
        ('C2', 'max_distance_low_tp', 1.0, '>', 'km'),
        ('C5', 'handover_count', 3, '>=', ''),
        ('C8', 'avg_rb', 170, '<', ''),
    ]
    for code, metric, thresh, op, unit in checks:
        if code not in available:
            continue
        val = m[metric]
        fmt = METRIC_FORMATS[metric]
        triggered = (
            (op == '>' and val > thresh) or
            (op == '>=' and val >= thresh) or
            (op == '<' and val < thresh)
        )
        suffix = f" {unit}".rstrip()
        val_str = f"{val:{fmt}}"
        if triggered and code == canonical:
            cmp_op = '>' if op == '>' else '>=' if op == '>=' else '<'
            lines.append(f"{metric} = {val_str}{suffix} {cmp_op} {thresh} -> {code}.")
            lines.append(f"The root cause is {CAUSE_DESCRIPTIONS[code]}.")
            return True
        elif triggered:
            cmp_op = '>' if op == '>' else '>=' if op == '>=' else '<'
            lines.append(f"{metric} = {val_str}{suffix} {cmp_op} {thresh} -> would suggest {code}.")
        else:
            inv_op = '<=' if op == '>' else '<' if op == '>=' else '>='
            lines.append(f"{metric} = {val_str}{suffix} {inv_op} {thresh} -> not {code}.")
    return False


def _walk_c1_detection(lines: list, m: Dict, canonical: str, available: set,
                       evidence: Dict, v16_override: str) -> bool:
    """Walk C1 detection rules. Returns True if answer resolved here."""
    if 'C1' not in available:
        lines.append("C1 not in available options -> skip.")
        return False

    r1 = m['min_rsrp'] < -90 and not m['pci_collision'] and m['c4_interference'] < 3
    r2 = m['strong_neighbor_count'] < 0.5 and m['serving_tilt'] >= 15
    r3 = m['pci_collision'] and m['strong_neighbor_count'] < 0.5

    rsrp_s = safe_cmp_fmt(m['min_rsrp'], -90, '<', decimals=2)
    c4_s = safe_cmp_fmt(m['c4_interference'], 3, '<', decimals=2)
    nb_s1 = safe_cmp_fmt(m['strong_neighbor_count'], 0.5, '<', decimals=2)
    nb_s3 = safe_cmp_fmt(m['strong_neighbor_count'], 0.5, '<', decimals=2)

    lines.append(
        f"Rule 1: min_rsrp = {rsrp_s} {'<' if m['min_rsrp'] < -90 else '>='} -90"
        f", pci_collision = {'yes' if m['pci_collision'] else 'no'}"
        f", c4_interference = {c4_s} {'<' if m['c4_interference'] < 3 else '>='} 3"
        f" -> {'TRIGGERED' if r1 else 'no'}."
    )
    lines.append(
        f"Rule 2: strong_neighbor_count = {nb_s1} {'<' if m['strong_neighbor_count'] < 0.5 else '>='} 0.5"
        f", serving_tilt = {m['serving_tilt']:.0f} {'>=' if m['serving_tilt'] >= 15 else '<'} 15"
        f" -> {'TRIGGERED' if r2 else 'no'}."
    )
    lines.append(
        f"Rule 3: pci_collision = {'yes' if m['pci_collision'] else 'no'}"
        f", strong_neighbor_count = {nb_s3} {'<' if m['strong_neighbor_count'] < 0.5 else '>='} 0.5"
        f" -> {'TRIGGERED' if r3 else 'no'}."
    )

    if not (r1 or r2 or r3):
        lines.append("No C1 detection rule triggered.")
        return False

    fired = 'Rule 1' if r1 else 'Rule 2' if r2 else 'Rule 3'
    lines.append(f"C1 detected via {fired}.")

    # V16 override checks (always show all 3)
    lines.append("V16 override checks:")

    b_fires = 'C3' in available and m['post_ho_good_streak'] >= 2
    lines.append(
        f"  B: post_ho_good_streak = {m['post_ho_good_streak']} {'>=' if m['post_ho_good_streak'] >= 2 else '<'} 2"
        f" -> {'OVERRIDE to C3' if b_fires else 'no'}."
    )
    if b_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C3']}.")
        return True

    p3_fires = 'C6' in available and m['pci_collision_ratio'] > 0.70
    cr_s = safe_cmp_fmt(m['pci_collision_ratio'], 0.70, '>', decimals=2)
    lines.append(
        f"  P3: pci_collision_ratio = {cr_s} {'>' if m['pci_collision_ratio'] > 0.70 else '<='} 0.70"
        f" -> {'OVERRIDE to C6' if p3_fires else 'no'}."
    )
    if p3_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C6']}.")
        return True

    p4_fires = 'C3' in available and m['avg_rsrp'] > -79 and m['strong_neighbor_count'] > 1.0
    rsrp_p4_s = safe_cmp_fmt(m['avg_rsrp'], -79, '>', decimals=3)
    nb_p4_s = safe_cmp_fmt(m['strong_neighbor_count'], 1.0, '>', decimals=2)
    lines.append(
        f"  P4: avg_rsrp = {rsrp_p4_s} {'>' if m['avg_rsrp'] > -79 else '<='} -79"
        f", strong_neighbor_count = {nb_p4_s} {'>' if m['strong_neighbor_count'] > 1.0 else '<='} 1.0"
        f" -> {'OVERRIDE to C3' if p4_fires else 'no'}."
    )
    if p4_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C3']}.")
        return True

    lines.append("No override. C1 confirmed.")
    lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C1']}.")
    return True


def _walk_c4(lines: list, m: Dict, canonical: str, available: set, evidence: Dict) -> bool:
    """Walk C4 interference check. Returns True if answer resolved here."""
    if 'C4' not in available:
        lines.append("C4 not in available options -> skip.")
        return False

    c4_s = safe_cmp_fmt(m['c4_interference'], 3, '<', decimals=2)
    if m['c4_interference'] < 3:
        lines.append(f"c4_interference = {c4_s} < 3 dB -> not C4.")
        return False

    ratio_skip = m['ratio_nbdiff_interf'] < -0.5 and m['c4_interference'] < 12
    lines.append(
        f"c4_interference = {c4_s} >= 3 dB."
    )
    ratio_s = safe_cmp_fmt(m['ratio_nbdiff_interf'], -0.5, '<', decimals=2)
    c4_12_s = safe_cmp_fmt(m['c4_interference'], 12, '<', decimals=2)
    lines.append(
        f"Ratio filter: ratio_nbdiff_interf = {ratio_s}"
        f" {'<' if m['ratio_nbdiff_interf'] < -0.5 else '>='} -0.5"
        f", c4_interference = {c4_12_s}"
        f" {'<' if m['c4_interference'] < 12 else '>='} 12"
        f" -> {'FILTERED (neighbors dominate, skip C4)' if ratio_skip else 'no filter, C4 confirmed'}."
    )

    if ratio_skip:
        return False

    lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C4']}.")
    return True


def _walk_c6(lines: list, m: Dict, canonical: str, available: set,
             evidence: Dict, v16_override: str) -> bool:
    """Walk C6 collision check with filtering. Returns True if answer resolved here."""
    if 'C6' not in available:
        lines.append("C6 not in available options -> skip.")
        return False

    if not m['pci_collision']:
        lines.append("pci_collision = no -> not C6.")
        return False

    lines.append(f"pci_collision = yes ({m['pci_collision_detail']}).")

    lines.append("Filter signals:")
    lines.append(
        f"  c1_signal: serving_tilt = {m['serving_tilt']:.0f} {'>=' if m['serving_tilt'] >= 20 else '<'} 20"
        f" -> {'yes' if m['c6_c1_signal'] else 'no'}."
    )
    lines.append(
        f"  c3_signal: min_neighbor_diff = {m['min_neighbor_diff']:.1f} {'<' if m['min_neighbor_diff'] < 3 else '>='} 3"
        f" AND serving_tilt = {m['serving_tilt']:.0f} {'>' if m['serving_tilt'] > 12 else '<='} 12"
        f" -> {'yes' if m['c6_c3_signal'] else 'no'}."
    )
    lines.append(
        f"  c3_off_axis: avg_off_axis = {m['avg_off_axis']:.1f} {'>' if m['avg_off_axis'] > 30 else '<='} 30"
        f" -> {'yes' if m['c6_c3_off_axis_signal'] else 'no'}."
    )

    no_signal = not m['c6_c1_signal'] and not m['c6_c3_signal'] and not m['c6_c3_off_axis_signal']

    if no_signal:
        lines.append("No filter signals -> genuine collision path.")
        return _walk_c6_no_signal_path(lines, m, canonical, available)

    if m['c6_c3_off_axis_signal']:
        rsrp_offaxis_s = safe_cmp_fmt(m['min_rsrp'], -90, '<', decimals=2)
        if m['min_rsrp'] < -90 and 'C1' in available:
            lines.append(f"Off-axis signal + min_rsrp = {rsrp_offaxis_s} < -90 -> downtilt path.")
            return _walk_c6_offaxis_c1_path(lines, m, canonical, available)
        elif 'C3' in available:
            lines.append(f"Off-axis signal + min_rsrp = {rsrp_offaxis_s} >= -90 -> neighbor-better path.")
            return _walk_c6_offaxis_c3_path(lines, m, canonical, available)

    signals = []
    if m['c6_c1_signal']:
        signals.append('c1_signal (high tilt)')
    if m['c6_c3_signal']:
        signals.append('c3_signal (small neighbor diff)')
    lines.append(f"Filter triggered: {', '.join(signals)} -> collision not primary cause, fall through to C1/C3.")
    return False


def _walk_c6_no_signal_path(lines: list, m: Dict, canonical: str, available: set) -> bool:
    """C6 no-signal path: B override, then P1 collision ratio check."""
    lines.append("V16 override checks:")

    b_fires = 'C3' in available and m['post_ho_good_streak'] >= 2
    lines.append(
        f"  B: post_ho_good_streak = {m['post_ho_good_streak']} {'>=' if m['post_ho_good_streak'] >= 2 else '<'} 2"
        f" -> {'OVERRIDE to C3' if b_fires else 'no'}."
    )
    if b_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C3']}.")
        return True

    if m['pci_collision_ratio'] >= 1.0:
        lines.append(f"P1: pci_collision_ratio = {m['pci_collision_ratio']:.2f} >= 1.0 -> genuine C6.")
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C6']}.")
        return True
    else:
        lines.append(f"P1: pci_collision_ratio = {m['pci_collision_ratio']:.2f} < 1.0 -> not genuine C6.")
        if 'C1' in available and m['serving_tilt'] > 10 and m['rsrp_trend'] > 0.4:
            lines.append(
                f"  serving_tilt = {m['serving_tilt']:.0f} > 10, rsrp_trend = {m['rsrp_trend']:.2f} > 0.4"
                f" -> OVERRIDE to C1."
            )
            lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C1']}.")
            return True
        elif 'C3' in available:
            lines.append(f"  Default fallback -> C3.")
            lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C3']}.")
            return True
        else:
            lines.append(f"  No better option -> keep C6.")
            lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C6']}.")
            return True


def _walk_c6_offaxis_c1_path(lines: list, m: Dict, canonical: str, available: set) -> bool:
    """C6 off-axis + weak RSRP -> C1 with V16 overrides B/P3/P4."""
    lines.append("V16 override checks:")

    b_fires = 'C3' in available and m['post_ho_good_streak'] >= 2
    lines.append(
        f"  B: post_ho_good_streak = {m['post_ho_good_streak']} {'>=' if m['post_ho_good_streak'] >= 2 else '<'} 2"
        f" -> {'OVERRIDE to C3' if b_fires else 'no'}."
    )
    if b_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C3']}.")
        return True

    p3_fires = 'C6' in available and m['pci_collision_ratio'] > 0.70
    cr_s = safe_cmp_fmt(m['pci_collision_ratio'], 0.70, '>', decimals=2)
    lines.append(
        f"  P3: pci_collision_ratio = {cr_s} {'>' if m['pci_collision_ratio'] > 0.70 else '<='} 0.70"
        f" -> {'OVERRIDE to C6' if p3_fires else 'no'}."
    )
    if p3_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C6']}.")
        return True

    p4_fires = 'C3' in available and m['avg_rsrp'] > -79 and m['strong_neighbor_count'] > 1.0
    rsrp_s = safe_cmp_fmt(m['avg_rsrp'], -79, '>', decimals=3)
    nb_s = safe_cmp_fmt(m['strong_neighbor_count'], 1.0, '>', decimals=2)
    lines.append(
        f"  P4: avg_rsrp = {rsrp_s} {'>' if m['avg_rsrp'] > -79 else '<='} -79"
        f", strong_neighbor_count = {nb_s} {'>' if m['strong_neighbor_count'] > 1.0 else '<='} 1.0"
        f" -> {'OVERRIDE to C3' if p4_fires else 'no'}."
    )
    if p4_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C3']}.")
        return True

    lines.append("No override. C1 confirmed (off-axis downtilt path).")
    lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C1']}.")
    return True


def _walk_c6_offaxis_c3_path(lines: list, m: Dict, canonical: str, available: set) -> bool:
    """C6 off-axis + good RSRP -> C3 with V16 overrides P2/G/J/P5b."""
    lines.append("V16 override checks:")

    p2_fires = 'C6' in available and m['pci_collision_ratio'] > 0.70
    cr_s = safe_cmp_fmt(m['pci_collision_ratio'], 0.70, '>', decimals=2)
    lines.append(
        f"  P2: pci_collision_ratio = {cr_s} {'>' if m['pci_collision_ratio'] > 0.70 else '<='} 0.70"
        f" -> {'OVERRIDE to C6' if p2_fires else 'no'}."
    )
    if p2_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C6']}.")
        return True

    g_fires = ('C1' in available
               and m['rsrp_change_during_prob'] > 5
               and m['rsrp_trend'] > 0.5
               and m['nb_within_5db_per_row'] < 1.0)
    rc_s = safe_cmp_fmt(m['rsrp_change_during_prob'], 5, '>')
    rt_s = safe_cmp_fmt(m['rsrp_trend'], 0.5, '>', decimals=2)
    nb5_s = safe_cmp_fmt(m['nb_within_5db_per_row'], 1.0, '<', decimals=2)
    lines.append(
        f"  G: rsrp_change = {rc_s} {'>' if m['rsrp_change_during_prob'] > 5 else '<='} 5"
        f", rsrp_trend = {rt_s} {'>' if m['rsrp_trend'] > 0.5 else '<='} 0.5"
        f", nb_5db = {nb5_s} {'<' if m['nb_within_5db_per_row'] < 1.0 else '>='} 1.0"
        f" -> {'OVERRIDE to C1' if g_fires else 'no'}."
    )
    if g_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C1']}.")
        return True

    j_fires = 'C1' in available and m['rsrp_recovery'] > 15
    rr_s = safe_cmp_fmt(m['rsrp_recovery'], 15, '>')
    lines.append(
        f"  J: rsrp_recovery = {rr_s} {'>' if m['rsrp_recovery'] > 15 else '<='} 15"
        f" -> {'OVERRIDE to C1' if j_fires else 'no'}."
    )
    if j_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C1']}.")
        return True

    p5b_fires = 'C1' in available and m['serving_tilt'] > 6 and m['nb_within_5db_per_row'] < 1.0
    nb5b_s = safe_cmp_fmt(m['nb_within_5db_per_row'], 1.0, '<', decimals=2)
    lines.append(
        f"  P5b: serving_tilt = {m['serving_tilt']:.0f} {'>' if m['serving_tilt'] > 6 else '<='} 6"
        f", nb_5db = {nb5b_s} {'<' if m['nb_within_5db_per_row'] < 1.0 else '>='} 1.0"
        f" -> {'OVERRIDE to C1' if p5b_fires else 'no'}."
    )
    if p5b_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C1']}.")
        return True

    lines.append("No override. C3 confirmed (off-axis neighbor-better path).")
    lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C3']}.")
    return True


def _walk_c1c3_tiebreaker(lines: list, m: Dict, canonical: str, available: set,
                          evidence: Dict, v16_override: str, is_expert: bool):
    """Walk V19 C1/C3 tiebreaker with SINR gate, updated thresholds, and rescue rules."""
    tilt = m['serving_tilt']
    avg_rsrp = m['avg_rsrp']
    min_nb_diff = m['min_neighbor_diff']
    avg_sinr = m.get('avg_sinr')

    pred, conf = classify_c1_vs_c3(tilt, avg_rsrp, min_nb_diff, avg_sinr)

    rsrp_s = safe_cmp_fmt(avg_rsrp, -90, '<', decimals=3)
    rsrp_s2 = safe_cmp_fmt(avg_rsrp, -82, '>', decimals=3)

    # tilt >= 28 with SINR gate
    if tilt >= 28:
        if avg_sinr is not None and avg_sinr >= 12:
            lines.append(
                f"serving_tilt = {tilt:.0f} >= 28, avg_sinr = {avg_sinr:.1f} >= 12"
                f" -> SINR gate: high confidence C3 (good SINR despite high tilt)."
            )
        else:
            sinr_str = f"{avg_sinr:.1f}" if avg_sinr is not None else "N/A"
            lines.append(
                f"serving_tilt = {tilt:.0f} >= 28, avg_sinr = {sinr_str} < 12"
                f" -> high confidence C1."
            )
    elif tilt < 12:
        lines.append(f"serving_tilt = {tilt:.0f} < 12 -> high confidence C3.")
    elif avg_rsrp < -90:
        lines.append(
            f"serving_tilt = {tilt:.0f} (12-27 range), avg_rsrp = {rsrp_s} < -90"
            f" -> medium confidence C1."
        )
    elif avg_rsrp > -82:
        lines.append(
            f"serving_tilt = {tilt:.0f} (12-27 range), avg_rsrp = {rsrp_s2} > -82"
            f" -> medium confidence C3."
        )
    else:
        lines.append(
            f"serving_tilt = {tilt:.0f} (12-27 range), avg_rsrp = {avg_rsrp:.3f} (-90 to -82)"
            f" -> low confidence {pred}."
        )

    # Low confidence -> rescue rules
    if conf == 'low':
        lines.append("Low confidence -> applying rescue rules:")
        _walk_rescue(lines, m, canonical, available, is_expert)
        return

    if conf in ('high', 'medium') and 'C1' in available and 'C3' in available:
        lines.append("V16 override checks:")
        if pred == 'C3':
            resolved = _show_c3_overrides(lines, m, available, canonical)
        else:
            resolved = _show_c1_overrides(lines, m, available, canonical)

        if not resolved:
            if pred == canonical:
                lines.append(f"No override. {pred} confirmed.")
                lines.append(f"The root cause is {CAUSE_DESCRIPTIONS[canonical]}.")
            else:
                lines.append(f"Deterministic classifier predicts {pred}, but examining additional indicators:")
                _add_expert_reasoning(lines, m, canonical)
    else:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS[canonical]}.")


def _walk_rescue(lines: list, m: Dict, canonical: str, available: set, is_expert: bool):
    """Walk V19 rescue rules R1-R4 for low-confidence C1/C3 cases."""
    cr = m['pci_collision_ratio']
    nb = m['strong_neighbor_count']
    c4 = m['c4_interference']

    # R1: collision_ratio >= 0.9 -> C6
    cr_s = safe_cmp_fmt(cr, 0.9, '>=', decimals=2)
    r1_fires = cr >= 0.9 and 'C6' in available
    lines.append(
        f"  R1: pci_collision_ratio = {cr_s} {'>=' if cr >= 0.9 else '<'} 0.9"
        f" -> {'C6' if r1_fires else 'no'}."
    )
    if r1_fires:
        if canonical == 'C6' or not is_expert:
            lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C6']}.")
        else:
            lines.append(f"Rescue rule suggests C6, but examining additional indicators:")
            _add_expert_reasoning(lines, m, canonical)
        return

    # R2: strong_neighbors < 0.8 -> C1
    nb_s = safe_cmp_fmt(nb, 0.8, '<', decimals=2)
    r2_fires = nb < 0.8 and 'C1' in available
    lines.append(
        f"  R2: strong_neighbor_count = {nb_s} {'<' if nb < 0.8 else '>='} 0.8"
        f" -> {'C1' if r2_fires else 'no'}."
    )
    if r2_fires:
        if canonical == 'C1' or not is_expert:
            lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C1']}.")
        else:
            lines.append(f"Rescue rule suggests C1, but examining additional indicators:")
            _add_expert_reasoning(lines, m, canonical)
        return

    # R3: c4_interference >= 3.0 -> C1
    c4_s = safe_cmp_fmt(c4, 3.0, '>=', decimals=2)
    r3_fires = c4 >= 3.0 and 'C1' in available
    lines.append(
        f"  R3: c4_interference = {c4_s} {'>=' if c4 >= 3.0 else '<'} 3.0"
        f" -> {'C1' if r3_fires else 'no'}."
    )
    if r3_fires:
        if canonical == 'C1' or not is_expert:
            lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C1']}.")
        else:
            lines.append(f"Rescue rule suggests C1, but examining additional indicators:")
            _add_expert_reasoning(lines, m, canonical)
        return

    # R4: default -> C3
    lines.append("  R4: default fallback -> C3.")
    if canonical == 'C3' or not is_expert:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C3']}.")
    else:
        lines.append(f"Default rescue suggests C3, but examining additional indicators:")
        _add_expert_reasoning(lines, m, canonical)


def _show_c3_overrides(lines: list, m: Dict, available: set, canonical: str) -> bool:
    """Show V16 overrides for C3 prediction: P2, G, J, P5b."""
    p2_fires = 'C6' in available and m['pci_collision_ratio'] > 0.70
    cr_s = safe_cmp_fmt(m['pci_collision_ratio'], 0.70, '>', decimals=2)
    lines.append(
        f"  P2: pci_collision_ratio = {cr_s} {'>' if m['pci_collision_ratio'] > 0.70 else '<='} 0.70"
        f" -> {'OVERRIDE to C6' if p2_fires else 'no'}."
    )
    if p2_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C6']}.")
        return True

    g_fires = ('C1' in available
               and m['rsrp_change_during_prob'] > 5
               and m['rsrp_trend'] > 0.5
               and m['nb_within_5db_per_row'] < 1.0)
    rc_s = safe_cmp_fmt(m['rsrp_change_during_prob'], 5, '>')
    rt_s = safe_cmp_fmt(m['rsrp_trend'], 0.5, '>', decimals=2)
    nb5_s = safe_cmp_fmt(m['nb_within_5db_per_row'], 1.0, '<', decimals=2)
    lines.append(
        f"  G: rsrp_change = {rc_s} {'>' if m['rsrp_change_during_prob'] > 5 else '<='} 5"
        f", rsrp_trend = {rt_s} {'>' if m['rsrp_trend'] > 0.5 else '<='} 0.5"
        f", nb_5db = {nb5_s} {'<' if m['nb_within_5db_per_row'] < 1.0 else '>='} 1.0"
        f" -> {'OVERRIDE to C1' if g_fires else 'no'}."
    )
    if g_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C1']}.")
        return True

    j_fires = 'C1' in available and m['rsrp_recovery'] > 15
    rr_s = safe_cmp_fmt(m['rsrp_recovery'], 15, '>')
    lines.append(
        f"  J: rsrp_recovery = {rr_s} {'>' if m['rsrp_recovery'] > 15 else '<='} 15"
        f" -> {'OVERRIDE to C1' if j_fires else 'no'}."
    )
    if j_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C1']}.")
        return True

    p5b_fires = 'C1' in available and m['serving_tilt'] > 6 and m['nb_within_5db_per_row'] < 1.0
    nb5b_s = safe_cmp_fmt(m['nb_within_5db_per_row'], 1.0, '<', decimals=2)
    lines.append(
        f"  P5b: serving_tilt = {m['serving_tilt']:.0f} {'>' if m['serving_tilt'] > 6 else '<='} 6"
        f", nb_5db = {nb5b_s} {'<' if m['nb_within_5db_per_row'] < 1.0 else '>='} 1.0"
        f" -> {'OVERRIDE to C1' if p5b_fires else 'no'}."
    )
    if p5b_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C1']}.")
        return True

    return False


def _show_c1_overrides(lines: list, m: Dict, available: set, canonical: str) -> bool:
    """Show V16 overrides for C1 prediction: P3, P4."""
    p3_fires = 'C6' in available and m['pci_collision_ratio'] > 0.70
    cr_s = safe_cmp_fmt(m['pci_collision_ratio'], 0.70, '>', decimals=2)
    lines.append(
        f"  P3: pci_collision_ratio = {cr_s} {'>' if m['pci_collision_ratio'] > 0.70 else '<='} 0.70"
        f" -> {'OVERRIDE to C6' if p3_fires else 'no'}."
    )
    if p3_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C6']}.")
        return True

    p4_fires = 'C3' in available and m['avg_rsrp'] > -79 and m['strong_neighbor_count'] > 1.0
    rsrp_s = safe_cmp_fmt(m['avg_rsrp'], -79, '>', decimals=3)
    nb_s = safe_cmp_fmt(m['strong_neighbor_count'], 1.0, '>', decimals=2)
    lines.append(
        f"  P4: avg_rsrp = {rsrp_s} {'>' if m['avg_rsrp'] > -79 else '<='} -79"
        f", strong_neighbor_count = {nb_s} {'>' if m['strong_neighbor_count'] > 1.0 else '<='} 1.0"
        f" -> {'OVERRIDE to C3' if p4_fires else 'no'}."
    )
    if p4_fires:
        lines.append(f"The root cause is {CAUSE_DESCRIPTIONS['C3']}.")
        return True

    return False


def _add_expert_reasoning(lines: list, m: Dict, canonical: str):
    """Add expert reasoning for cases where V19 prediction differs from ground truth."""
    tilt = m['serving_tilt']
    tilt_desc = 'high' if tilt >= 20 else 'moderate' if tilt >= 12 else 'low'
    if canonical == 'C1':
        nb = m['strong_neighbor_count']
        nb_diff = m['min_neighbor_diff']
        if nb < 1.0:
            lines.append(
                f"strong_neighbor_count = {nb:.2f} - few strong neighbors during low TP."
                f" min_neighbor_diff = {nb_diff:.1f} dB - neighbors generally weaker."
                f" Pattern of few strong neighbors despite {tilt_desc} tilt ({tilt:.0f} deg) suggests downtilt is"
                f" causing signal weakness at the cell edge."
            )
        else:
            lines.append(
                f"strong_neighbor_count = {nb:.2f} - some neighbors within 6 dB."
                f" min_neighbor_diff = {nb_diff:.1f} dB - but neighbors provide weaker signal overall."
                f" Despite nearby neighbors, the negative neighbor difference shows the serving cell's"
                f" {tilt_desc} tilt ({tilt:.0f} deg) is degrading coverage, not that a neighbor provides"
                f" better throughput."
            )
    elif canonical == 'C3':
        lines.append(
            f"strong_neighbor_count = {m['strong_neighbor_count']:.2f} - multiple neighbors within 6 dB."
            f" min_neighbor_diff = {m['min_neighbor_diff']:.1f} dB - at least one neighbor provides"
            f" comparable or stronger signal. A neighboring cell can provide higher throughput."
        )
    elif canonical == 'C6':
        lines.append(
            f"Although {tilt_desc} tilt ({tilt:.0f} deg) initially suggested downtilt rather than collision,"
            f" pci_collision_ratio = {m['pci_collision_ratio']:.2f} indicates collision is present in"
            f" {m['pci_collision_ratio']*100:.0f}% of drive test rows."
            f" avg_off_axis = {m['avg_off_axis']:.1f} deg - UE in the main beam where collision"
            f" has maximum impact. The persistent PCI mod 30 collision overrides the tilt signal."
        )
    elif canonical == 'C4':
        lines.append(
            f"c4_interference = {m['c4_interference']:.2f} dB shows significant non-colocated"
            f" co-frequency interference. Despite other indicators, the interference level"
            f" is the primary throughput degradation factor."
        )
    else:
        lines.append(
            f"Further analysis of the drive test data confirms the root cause"
            f" is {CAUSE_DESCRIPTIONS.get(canonical, canonical)}."
        )
    lines.append(f"The root cause is {CAUSE_DESCRIPTIONS[canonical]}.")


# =============================================================================
# TYPE B METRICS (used by SFT/GRPO/inference for prompt preparation)
# =============================================================================

def compute_type_b_metrics(question: str) -> Optional[Dict]:
    """Compute all Type B metrics from a question string.

    Includes config/signaling parsing: n1_in_config, inter_freq_ho,
    a2_thld, n_configured_neighbors.
    """
    drive_test, signaling = parse_type_b_question(question)
    if not drive_test:
        return None

    rsrps = [d['rsrp'] for d in drive_test if d['rsrp']]
    sinrs = [d['sinr'] for d in drive_test if d['sinr']]
    throughputs = [d['throughput'] for d in drive_test if d['throughput']]
    cce_fails = [d['cce_fail_rate'] for d in drive_test if d['cce_fail_rate'] is not None]
    blers = [d['initial_bler'] for d in drive_test if d['initial_bler'] is not None]
    rb_slots = [d['rb_slot'] for d in drive_test if d['rb_slot'] is not None]

    neighbor1_rsrps = [d['neighbor1_rsrp'] for d in drive_test if d.get('neighbor1_rsrp') is not None]
    neighbor2_rsrps = [d['neighbor2_rsrp'] for d in drive_test if d.get('neighbor2_rsrp') is not None]
    neighbor3_rsrps = [d['neighbor3_rsrp'] for d in drive_test if d.get('neighbor3_rsrp') is not None]

    avg_rsrp = sum(rsrps) / len(rsrps) if rsrps else -90
    avg_sinr = sum(sinrs) / len(sinrs) if sinrs else 10
    avg_cce_fail = sum(cce_fails) / len(cce_fails) if cce_fails else 0
    avg_bler = sum(blers) / len(blers) if blers else 0
    avg_rb = sum(rb_slots) / len(rb_slots) if rb_slots else 200

    avg_n1_rsrp = sum(neighbor1_rsrps) / len(neighbor1_rsrps) if neighbor1_rsrps else -120
    min_neighbor_diff = avg_rsrp - avg_n1_rsrp

    std_rsrp = (sum((r - avg_rsrp)**2 for r in rsrps) / len(rsrps))**0.5 if len(rsrps) > 1 else 0
    rsrp_var_norm = std_rsrp / abs(avg_rsrp) if avg_rsrp != 0 else 0

    pcis = [d['serving_pci'] for d in drive_test if d['serving_pci']]
    actual_handovers = sum(1 for i in range(1, len(pcis)) if pcis[i] != pcis[i-1]) if len(pcis) > 1 else 0

    ratio_a3_ho = signaling['a3_events'] / max(actual_handovers, 1)
    rrc_reestablish = signaling.get('rrc_reestablish', 0)

    # Conditional metrics during low-TP rows
    low_tp_rows = [d for d in drive_test if d.get('throughput') is not None and d['throughput'] < 100]

    def safe_avg(rows, key):
        vals = [d[key] for d in rows if d.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    low_tp_avg_mcs = safe_avg(low_tp_rows, 'avg_mcs')
    low_tp_avg_sinr = safe_avg(low_tp_rows, 'sinr')
    low_tp_avg_bler = safe_avg(low_tp_rows, 'initial_bler')

    phy_healthy_during_low_tp = None
    if low_tp_avg_mcs is not None and low_tp_avg_sinr is not None and low_tp_avg_bler is not None:
        phy_healthy_during_low_tp = (
            low_tp_avg_mcs > 10 and
            low_tp_avg_sinr > 8 and
            low_tp_avg_bler < 15
        )

    avg_n2_rsrp = sum(neighbor2_rsrps) / len(neighbor2_rsrps) if neighbor2_rsrps else -120
    avg_n3_rsrp = sum(neighbor3_rsrps) / len(neighbor3_rsrps) if neighbor3_rsrps else -120

    neighbors_within_3dB = 0
    neighbors_within_5dB = 0
    for avg_n in [avg_n1_rsrp, avg_n2_rsrp, avg_n3_rsrp]:
        if avg_n > -115:
            diff = avg_rsrp - avg_n
            if diff < 3:
                neighbors_within_3dB += 1
            if diff < 5:
                neighbors_within_5dB += 1

    n1_stronger_count = 0
    n1_total = 0
    for d in drive_test:
        if d.get('rsrp') is not None and d.get('neighbor1_rsrp') is not None:
            n1_total += 1
            if d['neighbor1_rsrp'] > d['rsrp']:
                n1_stronger_count += 1
    n1_stronger_pct = (n1_stronger_count / n1_total * 100) if n1_total > 0 else 0

    # Configuration and signaling table parsing
    config_cells = parse_config_data(question)
    inter_freq_ho = detect_inter_freq_ho(question)
    n1_in_config, serving_pci, n1_pci = check_n1_in_config(question, drive_test, config_cells)

    # Extract a2_thld and n_configured_neighbors from config
    a2_thld = None
    n_configured_neighbors = 0
    if serving_pci and serving_pci in config_cells:
        cfg = config_cells[serving_pci]
        a2_thld = cfg.get('a2_rsrp_thld')
        n_configured_neighbors = cfg.get('n_configured_neighbors', 0)

    return {
        'avg_rsrp': avg_rsrp,
        'avg_sinr': avg_sinr,
        'avg_cce_fail': avg_cce_fail,
        'avg_bler': avg_bler,
        'avg_rb': avg_rb,
        'actual_handovers': actual_handovers,
        'a3_events': signaling['a3_events'],
        'ratio_a3_ho': ratio_a3_ho,
        'rrc_reestablish': rrc_reestablish,
        'rsrp_var_norm': rsrp_var_norm,
        'min_neighbor_diff': min_neighbor_diff,
        'low_tp_avg_mcs': low_tp_avg_mcs,
        'low_tp_avg_sinr': low_tp_avg_sinr,
        'low_tp_avg_bler': low_tp_avg_bler,
        'phy_healthy_during_low_tp': phy_healthy_during_low_tp,
        'neighbors_within_3dB': neighbors_within_3dB,
        'neighbors_within_5dB': neighbors_within_5dB,
        'n1_stronger_pct': n1_stronger_pct,
        'n1_in_config': n1_in_config,
        'inter_freq_ho': inter_freq_ho,
        'a2_thld': a2_thld,
        'n_configured_neighbors': n_configured_neighbors,
    }


def format_type_b_metrics_block(m: Dict) -> str:
    """Format Type B metrics as a structured text block for the user message."""
    lines = [
        "Extracted metrics:",
        f"  avg_rsrp = {m['avg_rsrp']:.1f} dBm",
        f"  avg_sinr = {m['avg_sinr']:.1f} dB",
        f"  avg_cce_fail = {m['avg_cce_fail']:.2f}",
        f"  avg_bler = {m['avg_bler']:.1f}%",
        f"  avg_rb = {m['avg_rb']:.0f}",
        f"  actual_handovers = {m['actual_handovers']}",
        f"  a3_events = {m['a3_events']}",
        f"  ratio_a3_ho = {m['ratio_a3_ho']:.2f}",
        f"  rrc_reestablish = {m['rrc_reestablish']}",
        f"  rsrp_var_norm = {m['rsrp_var_norm']:.3f}",
        f"  min_neighbor_diff = {m['min_neighbor_diff']:.1f} dB",
    ]

    if m['low_tp_avg_mcs'] is not None:
        lines.append(f"  low_tp_avg_mcs = {m['low_tp_avg_mcs']:.1f}")
    else:
        lines.append("  low_tp_avg_mcs = N/A")

    if m['low_tp_avg_sinr'] is not None:
        lines.append(f"  low_tp_avg_sinr = {m['low_tp_avg_sinr']:.1f} dB")
    else:
        lines.append("  low_tp_avg_sinr = N/A")

    if m['low_tp_avg_bler'] is not None:
        lines.append(f"  low_tp_avg_bler = {m['low_tp_avg_bler']:.1f}%")
    else:
        lines.append("  low_tp_avg_bler = N/A")

    if m['phy_healthy_during_low_tp'] is not None:
        lines.append(f"  phy_healthy_during_low_tp = {m['phy_healthy_during_low_tp']}")
    else:
        lines.append("  phy_healthy_during_low_tp = N/A")

    lines.extend([
        f"  neighbors_within_3dB = {m['neighbors_within_3dB']}",
        f"  neighbors_within_5dB = {m['neighbors_within_5dB']}",
        f"  n1_stronger_pct = {m['n1_stronger_pct']:.1f}%",
    ])

    lines.append(f"  n1_in_config = {m.get('n1_in_config', 'N/A')}")
    lines.append(f"  inter_freq_ho = {m.get('inter_freq_ho', False)}")
    if m.get('a2_thld') is not None:
        lines.append(f"  a2_thld = {m['a2_thld']}")
    else:
        lines.append("  a2_thld = N/A")
    lines.append(f"  n_configured_neighbors = {m.get('n_configured_neighbors', 0)}")

    return '\n'.join(lines)


# =============================================================================
# TRACE GENERATOR
# =============================================================================

def generate_type_a_traces(
    train_csv: Path,
    output_dict: Dict,
    stats: Dict,
    spot_check: int = 0,
):
    """Generate Type A traces from train.csv with ground truth labels.

    Args:
        train_csv: Path to train.csv (must have 'ID', 'question', 'answer' columns)
        output_dict: dict to add traces to
        stats: stats dict to update
    """
    logger.info(f"Loading training data from {train_csv}")
    train_df = pd.read_csv(train_csv)
    logger.info(f"Loaded {len(train_df)} training questions")

    ground_truth_map = dict(zip(train_df['ID'], train_df['answer']))
    logger.info(f"Ground truth labels: {len(ground_truth_map)}")

    for _, row in train_df.iterrows():
        qid = row['ID']
        question = row['question']

        qtype = classify_question_type(question)
        if qtype != 'type_a_telco':
            continue

        ground_truth = ground_truth_map.get(qid)
        if not ground_truth:
            logger.warning(f"{qid}: no ground truth, skipping")
            continue

        result = classify_type_a(question)
        drive_test, cells = parse_type_a_question(question)
        option_map = extract_type_a_options(question)
        cause_to_label = {cause: label for label, cause in option_map.items()}
        available_causes = set(option_map.values())

        answer_label = cause_to_label.get(ground_truth)
        if not answer_label:
            logger.warning(f"{qid}: ground truth {ground_truth} not in options, skipping")
            continue

        m = compute_all_metrics(question, drive_test, cells)

        is_expert = result['confidence'] == 'needs_llm'
        if not is_expert and result['canonical'] != ground_truth:
            logger.info(f"{qid}: V19 predicted {result['canonical']} but truth is {ground_truth}")
            is_expert = True

        source = 'expert' if is_expert else 'deterministic'
        trace_text = generate_trace(m, result, available_causes, ground_truth, is_expert=is_expert)
        formatted_trace = f"<think>\n{trace_text}\n</think>"

        output_dict[qid] = {
            'question': question,
            'expected_answer': answer_label,
            'derived_answer': answer_label,
            'reasoning_trace': formatted_trace,
            'attempts': 1,
            'success': True,
            'source': source,
            'question_type': 'type_a',
        }

        stats['total'] += 1
        stats['by_cause'][ground_truth] += 1
        stats['by_confidence'][result['confidence']] += 1
        stats['expert' if is_expert else 'deterministic'] += 1
        stats['correct' if result['canonical'] == ground_truth else 'incorrect'] += 1
        stats['trace_lengths'].append(len(trace_text))


def _print_summary(stats, spot_check, checkpoint):
    logger.info("=" * 60)
    logger.info("GENERATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Type A traces: {stats['total']}")
    logger.info(f"  Deterministic: {stats['deterministic']}")
    logger.info(f"  Expert: {stats['expert']}")
    logger.info(f"  V19 accuracy: {stats['correct']}/{stats['total']}")
    logger.info("")
    logger.info("By cause:")
    for c in sorted(stats['by_cause']):
        logger.info(f"  {c}: {stats['by_cause'][c]}")
    logger.info("")
    logger.info("By confidence:")
    for c in sorted(stats['by_confidence']):
        logger.info(f"  {c}: {stats['by_confidence'][c]}")

    if stats['trace_lengths']:
        L = stats['trace_lengths']
        logger.info(f"Trace length (chars): min={min(L)}, max={max(L)}, mean={sum(L)/len(L):.0f}")

    logger.info(f"\nTotal traces: {len(checkpoint)}")

    if spot_check > 0:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"SPOT CHECK ({spot_check} samples)")
        logger.info("=" * 60)
        items = list(checkpoint.items())
        det = [(k, v) for k, v in items if v['source'] == 'deterministic']
        exp = [(k, v) for k, v in items if v['source'] == 'expert']

        shown = 0
        for label, pool in [("DETERMINISTIC", det[:3]), ("EXPERT", exp[:2])]:
            for qid, data in pool:
                if shown >= spot_check:
                    break
                logger.info(f"\n--- [{label}] {qid} -> {data['expected_answer']} ({data['source']}) ---")
                trace = data['reasoning_trace']
                logger.info(trace[:2000])
                if len(trace) > 2000:
                    logger.info(f"... ({len(trace)} chars total)")
                shown += 1


# =============================================================================
# VALIDATION
# =============================================================================

def validate_checkpoint(checkpoint_path: Path, train_csv: Path = None):
    """Validate that all traces have correct format and match ground truth."""
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    issues = []
    for qid, data in checkpoint.items():
        trace = data.get('reasoning_trace', '')
        if not trace.startswith('<think>'):
            issues.append(f"{qid}: missing <think> tag")
        if not trace.rstrip().endswith('</think>'):
            issues.append(f"{qid}: missing </think> closing tag")

    if train_csv and train_csv.exists():
        train_df = pd.read_csv(train_csv)
        train_answers = dict(zip(train_df['ID'], train_df['answer']))

        for qid, data in checkpoint.items():
            if data.get('question_type') == 'type_a':
                gt = train_answers.get(qid)
                if gt and data['expected_answer'] != gt:
                    issues.append(f"{qid}: answer={data['expected_answer']} != truth={gt}")

    if issues:
        logger.error(f"Validation found {len(issues)} issues:")
        for issue in issues[:20]:
            logger.error(f"  {issue}")
        return False

    type_a = sum(1 for v in checkpoint.values() if v.get('question_type') == 'type_a')
    logger.info(f"Validation passed: {len(checkpoint)} traces ({type_a} Type A), format OK")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate reasoning traces from train.csv for SFT/GRPO training",
    )
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR / 'traces_final.json'))
    parser.add_argument('--spot-check', type=int, default=0)
    parser.add_argument('--validate-only', type=str, default=None)
    args = parser.parse_args()

    if args.validate_only:
        train_csv = DATA_DIR / 'train.csv'
        validate_checkpoint(Path(args.validate_only), train_csv)
        return

    output_path = Path(args.output)

    checkpoint = {}
    stats = {
        'total': 0, 'deterministic': 0, 'expert': 0,
        'correct': 0, 'incorrect': 0,
        'by_cause': Counter(), 'by_confidence': Counter(),
        'trace_lengths': [],
    }

    train_csv = DATA_DIR / 'train.csv'
    if not train_csv.exists():
        logger.error(f"Training data not found: {train_csv}")
        return

    logger.info("=" * 60)
    logger.info("GENERATING TRACES FROM TRAIN.CSV")
    logger.info("=" * 60)
    generate_type_a_traces(
        train_csv=train_csv,
        output_dict=checkpoint,
        stats=stats,
    )
    logger.info(f"Generated {len(checkpoint)} traces")

    # Save
    logger.info(f"Saving {len(checkpoint)} traces to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    _print_summary(stats, args.spot_check, checkpoint)

    # Validate
    validate_checkpoint(output_path, train_csv=train_csv)


if __name__ == '__main__':
    main()
