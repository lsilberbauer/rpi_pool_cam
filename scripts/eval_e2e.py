#!/usr/bin/env python3
"""End-to-end evaluation of the digit CNN on full captured LCD images.

For every annotated PNG in data/captured/, runs the complete pipeline
(extract 6 digit ROIs, classify with ONNX model) and compares the
predicted PH and Redux values against the YAML ground-truth labels.

This gives a realistic measure of production accuracy – not just
per-digit accuracy on isolated crops.

Usage:
    # Evaluate with the current best model (default)
    python scripts/eval_e2e.py

    # Evaluate a specific ONNX model
    python scripts/eval_e2e.py --model models/digit_cnn.onnx

    # Save per-image results to CSV
    python scripts/eval_e2e.py --csv results/e2e_baseline.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import sys

import cv2
import numpy as np
import yaml

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.cam_image_processor.lcd_detector import extract_digit_rois

CAPTURED_DIR = ROOT / "data" / "captured"
DEFAULT_MODEL = ROOT / "models" / "digit_cnn.onnx"
CONFIG_PATH = ROOT / "config.yaml"


# ── helpers ───────────────────────────────────────────────────────────────────


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_digit_labels(ph: float, redux: int) -> dict[int, int]:
    """ROI index → ground-truth digit (same mapping as build_dataset.py)."""
    ph_str = f"{ph:.2f}".replace(".", "")
    redux_str = f"{int(redux):03d}"
    return {
        0: int(ph_str[0]),
        2: int(ph_str[1]),
        3: int(ph_str[2]),
        5: int(redux_str[0]),
        6: int(redux_str[1]),
        7: int(redux_str[2]),
    }


def classify_rois(session, rois: list[np.ndarray]) -> list[int]:
    """Run ONNX session on the 6 active ROIs (indices 0,2,3,5,6,7)."""
    digits = []
    for i in (0, 2, 3, 5, 6, 7):
        roi = rois[i].astype(np.float32) / 255.0
        roi = (roi - 0.5) / 0.5
        inp = roi[np.newaxis, np.newaxis]          # (1, 1, H, W)
        logits = session.run(["logits"], {"image": inp})[0]
        digits.append(int(logits.argmax()))
    return digits


def digits_to_readings(digits: list[int]) -> tuple[float, int]:
    """Convert 6-digit list [ph_int, ph_t, ph_h, rx_100, rx_10, rx_1] to (ph, redux)."""
    ph = round(digits[0] + digits[1] * 0.1 + digits[2] * 0.01, 2)
    redux = digits[3] * 100 + digits[4] * 10 + digits[5]
    return ph, redux


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end evaluation on captured LCD images.")
    parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL,
                        help="Path to ONNX model (default: models/digit_cnn.onnx)")
    parser.add_argument("--csv", type=pathlib.Path, default=None,
                        help="Optional path to write per-image CSV results.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every mis-classified image.")
    args = parser.parse_args()

    import onnxruntime as ort
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    print(f"Model: {args.model}")

    config = load_config()
    digit_config = config["lcd"]["digits"]

    yaml_files = sorted(CAPTURED_DIR.glob("*.yaml"))

    # Counters
    total = skipped = errors = 0
    ph_exact = redux_exact = both_exact = 0
    digit_correct = digit_total = 0
    # per-position accuracy: indices 0,2,3,5,6,7 → positions 0-5
    pos_correct = [0] * 6
    pos_total   = [0] * 6
    ph_errors: list[float] = []
    redux_errors: list[int] = []

    csv_rows = []

    for yf in yaml_files:
        with open(yf) as f:
            meta = yaml.safe_load(f) or {}
        ph_gt = meta.get("PH")
        rx_gt = meta.get("Redux")
        if ph_gt is None or rx_gt is None:
            skipped += 1
            continue
        ph_gt = float(ph_gt)
        rx_gt = int(rx_gt)
        if ph_gt == 0.0:
            skipped += 1
            continue

        png_path = yf.with_suffix(".png")
        if not png_path.exists():
            png_path = yf.with_suffix(".jpg")
        if not png_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        # Overexposure check
        if (img > 240).mean() > 0.5:
            skipped += 1
            continue

        try:
            rois = extract_digit_rois(img, digit_config)
        except Exception as e:
            errors += 1
            if args.verbose:
                print(f"  ROI extraction failed on {yf.name}: {e}")
            continue

        try:
            pred_digits = classify_rois(session, rois)
        except Exception as e:
            errors += 1
            if args.verbose:
                print(f"  Classification failed on {yf.name}: {e}")
            continue

        ph_pred, rx_pred = digits_to_readings(pred_digits)

        gt_labels = get_digit_labels(ph_gt, rx_gt)
        gt_digit_list = [gt_labels[i] for i in (0, 2, 3, 5, 6, 7)]

        ph_ok   = abs(ph_pred - ph_gt) < 0.005
        rx_ok   = (rx_pred == rx_gt)
        both_ok = ph_ok and rx_ok

        if ph_ok:   ph_exact   += 1
        if rx_ok:   redux_exact += 1
        if both_ok: both_exact  += 1

        for k, (gt_d, pr_d) in enumerate(zip(gt_digit_list, pred_digits)):
            pos_total[k] += 1
            if gt_d == pr_d:
                pos_correct[k] += 1
                digit_correct += 1
            digit_total += 1

        ph_err   = abs(ph_pred - ph_gt)
        redux_err = abs(rx_pred - rx_gt)
        ph_errors.append(ph_err)
        redux_errors.append(redux_err)

        if args.verbose and not both_ok:
            print(f"  {yf.stem}  PH gt={ph_gt:.2f} pred={ph_pred:.2f}  "
                  f"Redux gt={rx_gt} pred={rx_pred}")

        if args.csv:
            csv_rows.append({
                "file": yf.stem,
                "ph_gt": ph_gt,
                "ph_pred": ph_pred,
                "redux_gt": rx_gt,
                "redux_pred": rx_pred,
                "ph_ok": int(ph_ok),
                "redux_ok": int(rx_ok),
            })

        total += 1

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Images evaluated : {total}  (skipped={skipped}, errors={errors})")
    print(f"{'─'*60}")

    if total == 0:
        print("No images to evaluate.")
        return

    print(f"\nReading-level accuracy")
    print(f"  PH exact match     : {ph_exact:5d}/{total}  ({100*ph_exact/total:.1f}%)")
    print(f"  Redux exact match  : {redux_exact:5d}/{total}  ({100*redux_exact/total:.1f}%)")
    print(f"  Both exact         : {both_exact:5d}/{total}  ({100*both_exact/total:.1f}%)")

    print(f"\nDigit-level accuracy")
    pos_names = ["D0(PH int)", "D2(PH .1)", "D3(PH .01)",
                 "D5(Rx 100)", "D6(Rx 10)", "D7(Rx 1)"]
    for k, name in enumerate(pos_names):
        n = pos_total[k]
        c = pos_correct[k]
        print(f"  {name:<14}: {c:5d}/{n}  ({100*c/n:.1f}%)")
    print(f"  {'Overall':<14}: {digit_correct:5d}/{digit_total}  "
          f"({100*digit_correct/digit_total:.1f}%)")

    ph_arr   = np.array(ph_errors)
    rx_arr   = np.array(redux_errors)
    print(f"\nError statistics")
    print(f"  PH   mean abs error : {ph_arr.mean():.4f}  "
          f"max={ph_arr.max():.4f}  >0.01={( ph_arr>0.01).sum()}")
    print(f"  Redux mean abs error: {rx_arr.mean():.2f}   "
          f"max={rx_arr.max()}  >0={(rx_arr>0).sum()}")
    print(f"{'─'*60}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nCSV results written to: {args.csv}")


if __name__ == "__main__":
    main()
