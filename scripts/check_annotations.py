#!/usr/bin/env python3
"""Detect and optionally fix annotation outliers in data/captured/.

Uses temporal smoothing: for each frame, compares its PH/Redux values against
the median of its closest neighbours (within ±10 min) to detect single-frame
spikes caused by vision-model misclassification.

Two-tier flagging:
  CRITICAL  — deviation is so large or value so implausible it must be wrong.
              Auto-fixable if the corrected value is obvious, else nulled.
  WARNING   — unusual jump, needs human review.

Usage:
    # Dry run — only print report
    python scripts/check_annotations.py

    # Apply fixes (nulls critical errors, applies high-confidence corrections)
    python scripts/check_annotations.py --fix

    # More aggressive: also null WARNING-level anomalies
    python scripts/check_annotations.py --fix --null-warnings
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime, timedelta

import yaml

ROOT = pathlib.Path(__file__).parent.parent
CAPTURED_DIR = ROOT / "data" / "captured"

# Thresholds
CRITICAL_PH   = 0.30   # single-frame PH jump vs. local median → critical
WARNING_PH    = 0.07   # single-frame PH jump → warning (catches 1-digit tenths/hundredths errors)

CRITICAL_RX   = 40     # single-frame Redux jump → critical
WARNING_RX    = 15     # single-frame Redux jump → warning (catches 1-digit tens misreads)

WINDOW_MIN    = 10     # look ±N minutes for neighbours
MIN_NEIGHBOURS = 3     # need at least this many neighbours to compare

# Impossible range checks (always critical regardless of neighbours)
PH_MIN, PH_MAX   = 5.0, 10.0
RX_MIN, RX_MAX   = 200, 900


def load_annotations() -> list[dict]:
    """Load all valid annotations sorted by timestamp."""
    records = []
    for yf in sorted(CAPTURED_DIR.glob("*.yaml")):
        try:
            dt = datetime.strptime(yf.stem, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        meta = yaml.safe_load(open(yf)) or {}
        ph = meta.get("PH")
        rx = meta.get("Redux")
        if ph is None or rx is None:
            continue
        ph = float(ph)
        rx = int(rx)
        if ph == 0.0 or rx == 0:
            continue
        records.append({"path": yf, "dt": dt, "ph": ph, "rx": rx, "meta": meta})
    return records


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def neighbours(records: list[dict], idx: int) -> list[dict]:
    """Return records within ±WINDOW_MIN of records[idx], excluding idx itself."""
    centre = records[idx]["dt"]
    window = timedelta(minutes=WINDOW_MIN)
    return [
        r for i, r in enumerate(records)
        if i != idx and abs(r["dt"] - centre) <= window
    ]


def try_correct_redux(rx: int, median_rx: float) -> int | None:
    """Try to infer the correct Redux value from a known OCR error pattern.

    Patterns observed:
      - Hundreds digit misread: 274→774, 260→760, 277→777 (add 500)
      - Hundreds digit misread: 342→742 (add 400)
      - Tens or ones dropped: 4→774? not reliably inferrable

    Only returns a correction if the candidate is within 15 of the local median
    (high confidence). Otherwise returns None and the caller will null the annotation.
    """
    CONFIDENCE = 15  # max allowed distance from median to trust correction

    # Pattern: hundreds digit off by a fixed offset
    for delta in (700, 600, 500, 400, 300, 200):
        candidate = rx + delta
        if abs(candidate - median_rx) <= CONFIDENCE and RX_MIN <= candidate <= RX_MAX:
            return candidate
    # Pattern: leading hundreds digit completely wrong — replace it
    median_hundreds = int(round(median_rx / 100)) * 100
    candidate = median_hundreds + (rx % 100)
    if abs(candidate - median_rx) <= CONFIDENCE and RX_MIN <= candidate <= RX_MAX:
        return candidate
    return None


def try_correct_ph(ph: float, median_ph: float) -> float | None:
    """Try to infer the correct PH from a known OCR error pattern.

    Patterns observed:
      - Tenths digit misread: 7.12 → 7.17? hard to infer reliably.
      - Impossible value: 8.97 during dosing not necessarily wrong.
    """
    return None  # PH corrections are not safe to auto-apply


def is_isolated_spike(records: list[dict], idx: int, field: str, threshold: float) -> bool:
    """Return True only if this frame is an isolated spike (not part of a trend).

    A spike is isolated when the two immediately adjacent frames (prev and next)
    both deviate less than threshold from their own local medians in the same field.
    If the neighbours are also deviating, the whole group is a genuine trend.
    """
    prev_idx = idx - 1 if idx > 0 else None
    next_idx = idx + 1 if idx < len(records) - 1 else None

    # Require at least one adjacent frame that is NOT itself a spike
    adjacent_normal = 0
    for adj_idx in filter(None.__ne__, [prev_idx, next_idx]):
        adj_rec  = records[adj_idx]
        adj_nbrs = neighbours(records, adj_idx)
        if len(adj_nbrs) < MIN_NEIGHBOURS:
            continue
        if field == "ph":
            adj_val = adj_rec["ph"]
            adj_med = median([n["ph"] for n in adj_nbrs])
        else:
            adj_val = adj_rec["rx"]
            adj_med = median([n["rx"] for n in adj_nbrs])
        if abs(adj_val - adj_med) < threshold:
            adjacent_normal += 1
    return adjacent_normal >= 1


def analyse(records: list[dict]) -> list[dict]:
    """Return list of flagged items with diagnosis."""
    flags = []
    for i, rec in enumerate(records):
        nbrs = neighbours(records, i)
        if len(nbrs) < MIN_NEIGHBOURS:
            continue

        ph, rx = rec["ph"], rec["rx"]
        nbr_ph = [n["ph"] for n in nbrs]
        nbr_rx = [n["rx"] for n in nbrs]
        med_ph = median(nbr_ph)
        med_rx = median(nbr_rx)
        dev_ph = abs(ph - med_ph)
        dev_rx = abs(rx - med_rx)

        issues = []

        # --- PH ---
        if ph < PH_MIN or ph > PH_MAX:
            issues.append({"field": "PH", "level": "CRITICAL",
                            "value": ph, "median": med_ph,
                            "dev": dev_ph, "fix": None,
                            "reason": f"impossible value {ph}"})
        elif dev_ph >= CRITICAL_PH:
            fix_ph = try_correct_ph(ph, med_ph)
            issues.append({"field": "PH", "level": "CRITICAL",
                            "value": ph, "median": round(med_ph, 3),
                            "dev": round(dev_ph, 3), "fix": fix_ph,
                            "reason": f"Δ={dev_ph:.2f} vs median {med_ph:.2f}"})
        elif dev_ph >= WARNING_PH and is_isolated_spike(records, i, "ph", WARNING_PH):
            issues.append({"field": "PH", "level": "WARNING",
                            "value": ph, "median": round(med_ph, 3),
                            "dev": round(dev_ph, 3), "fix": None,
                            "reason": f"isolated spike Δ={dev_ph:.2f} vs median {med_ph:.2f}"})

        # --- Redux ---
        if rx < RX_MIN or rx > RX_MAX:
            fix_rx = try_correct_redux(rx, med_rx)
            issues.append({"field": "Redux", "level": "CRITICAL",
                            "value": rx, "median": round(med_rx, 1),
                            "dev": round(dev_rx, 1), "fix": fix_rx,
                            "reason": f"impossible value {rx} (range {RX_MIN}-{RX_MAX})"})
        elif dev_rx >= CRITICAL_RX:
            fix_rx = try_correct_redux(rx, med_rx)
            issues.append({"field": "Redux", "level": "CRITICAL",
                            "value": rx, "median": round(med_rx, 1),
                            "dev": round(dev_rx, 1), "fix": fix_rx,
                            "reason": f"Δ={dev_rx:.0f} vs median {med_rx:.0f}"})
        elif dev_rx >= WARNING_RX and is_isolated_spike(records, i, "rx", WARNING_RX):
            fix_rx = try_correct_redux(rx, med_rx)
            issues.append({"field": "Redux", "level": "WARNING",
                            "value": rx, "median": round(med_rx, 1),
                            "dev": round(dev_rx, 1), "fix": fix_rx,
                            "reason": f"isolated spike Δ={dev_rx:.0f} vs median {med_rx:.0f}"})

        if issues:
            # Show a few neighbours for context
            ctx = [(n["dt"].strftime("%H:%M:%S"), n["ph"], n["rx"])
                   for n in sorted(nbrs, key=lambda x: x["dt"])[:5]]
            flags.append({"rec": rec, "issues": issues, "context": ctx})

    return flags


def apply_fix(rec: dict, issue: dict, null_warnings: bool) -> bool:
    """Modify the YAML file according to the issue diagnosis. Returns True if changed."""
    path = rec["path"]
    meta = dict(rec["meta"])
    changed = False

    level = issue["level"]
    field = issue["field"]
    fix_val = issue.get("fix")

    if level == "CRITICAL":
        if fix_val is not None:
            meta[field] = fix_val
            changed = True
        else:
            # Null out the entire annotation — exclude from training
            meta["PH"] = None
            meta["Redux"] = None
            changed = True
    elif level == "WARNING" and null_warnings:
        meta["PH"] = None
        meta["Redux"] = None
        changed = True

    if changed:
        with open(path, "w") as f:
            yaml.dump(meta, f, default_flow_style=False, allow_unicode=True)

    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true",
                        help="Apply fixes (auto-correct or null critical issues).")
    parser.add_argument("--null-warnings", action="store_true",
                        help="Also null out WARNING-level anomalies (requires --fix).")
    args = parser.parse_args()

    print("Loading annotations…")
    records = load_annotations()
    print(f"  {len(records)} valid annotated frames loaded.")

    print("Analysing for temporal outliers…\n")
    flags = analyse(records)

    if not flags:
        print("No anomalies detected.")
        return

    # Summary counts
    n_critical = sum(1 for f in flags for iss in f["issues"] if iss["level"] == "CRITICAL")
    n_warning  = sum(1 for f in flags for iss in f["issues"] if iss["level"] == "WARNING")
    print(f"Found {len(flags)} frames with issues  "
          f"({n_critical} CRITICAL, {n_warning} WARNING)\n")
    print("─" * 72)

    applied = nulled = corrected = 0

    for flag in flags:
        rec = flag["rec"]
        print(f"\n[{rec['dt'].strftime('%Y-%m-%d %H:%M:%S')}]  "
              f"PH={rec['ph']:.2f}  Redux={rec['rx']}   "
              f"file: {rec['path'].name}")

        for iss in flag["issues"]:
            marker = "🔴 CRITICAL" if iss["level"] == "CRITICAL" else "🟡 WARNING"
            fix_str = (f"  → auto-correct to {iss['fix']}" if iss["fix"] is not None
                       else "  → null annotation")
            print(f"  {marker}  {iss['field']}: {iss['reason']}{fix_str if iss['level']=='CRITICAL' else ''}")

        ctx_str = "  Neighbours: " + ", ".join(
            f"{t} PH={p:.2f}/Rx={r}" for t, p, r in flag["context"]
        )
        print(ctx_str)

        if args.fix:
            ph_issues  = [iss for iss in flag["issues"] if iss["field"] == "PH"]
            rx_issues  = [iss for iss in flag["issues"] if iss["field"] == "Redux"]

            meta = dict(rec["meta"])
            file_changed = False

            for iss in ph_issues + rx_issues:
                level = iss["level"]
                field = iss["field"]
                fix_val = iss.get("fix")

                if level == "CRITICAL":
                    file_changed = True
                    if fix_val is not None:
                        meta[field] = fix_val
                        print(f"  ✔ corrected {field}: {iss['value']} → {fix_val}")
                        corrected += 1
                    else:
                        meta["PH"] = None
                        meta["Redux"] = None
                        print(f"  ✖ nulled entire annotation")
                        nulled += 1
                        break   # no point continuing, both fields are null
                elif level == "WARNING" and args.null_warnings:
                    file_changed = True
                    meta["PH"] = None
                    meta["Redux"] = None
                    print(f"  ✖ nulled (WARNING)")
                    nulled += 1
                    break

            if file_changed:
                with open(rec["path"], "w") as f:
                    yaml.dump(meta, f, default_flow_style=False, allow_unicode=True)
                applied += 1

    print("\n" + "─" * 72)
    if args.fix:
        print(f"\nSummary: {applied} files modified  "
              f"({corrected} field corrections, {nulled} annotations nulled)")
    else:
        print(f"\nDry run — no files changed.  "
              f"Run with --fix to apply corrections.")


if __name__ == "__main__":
    main()
