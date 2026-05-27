#!/usr/bin/env python3
"""Build a digit image dataset from annotated LCD captures.

Step 1 — Extract:
    For every annotated YAML in data/captured/ (and data/ for the older
    hand-labelled images), run extract_digit_rois on the unwarped LCD image
    and save individual crops to data/dataset/{0-9}/.

Step 2 — Balance:
    Sample an equal number of images per digit class (capped at the size of the
    smallest class) and copy them to data/dataset_balanced/{0-9}/.

Digit-to-position mapping (matches the notebook convention):
    Row 0: [D0] [D1] [D2] [D3]   ← D0=PH integer, D1=decimal point (skip),
                                    D2=PH tenth, D3=PH hundredth
    Row 1: [D4] [D5] [D6] [D7]   ← D4=unused separator (skip), D5=Redux hundreds,
                                    D6=Redux tens, D7=Redux ones

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --no-balance   # extract only
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import random
import shutil
import sys

import cv2
import yaml

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.cam_image_processor.cam_image_processor import crop_rectangle
from src.cam_image_processor.lcd_detector import detect_and_unwarp, extract_digit_rois

CAPTURED_DIR = ROOT / "data" / "captured"
LEGACY_DIR = ROOT / "data"          # older hand-labelled JPGs
CONFIG_PATH = ROOT / "config.yaml"
DATASET_DIR = ROOT / "data" / "dataset"
BALANCED_DIR = ROOT / "data" / "dataset_balanced"


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def is_overexposed(img: "cv2.Mat", threshold: float = 0.5) -> bool:
    """Return True if more than *threshold* fraction of pixels exceed 240.

    This reliably catches images taken when the cellar light was on, which
    wash out the LCD display completely.
    """
    return (img > 240).mean() > threshold


def get_digit_labels(ph: float, redux: int) -> dict[int, int]:
    """Map PH and Redux to {roi_index: digit_class}.

    Positions 1 (decimal point) and 4 (separator) are skipped.
    """
    ph_str = f"{ph:.2f}".replace(".", "")   # "7.16" → "716"
    redux_str = f"{int(redux):03d}"          # 790    → "790"
    return {
        0: int(ph_str[0]),    # D1 — PH integer part
        2: int(ph_str[1]),    # D3 — PH first decimal
        3: int(ph_str[2]),    # D4 — PH second decimal
        5: int(redux_str[0]), # D6 — Redux hundreds
        6: int(redux_str[1]), # D7 — Redux tens
        7: int(redux_str[2]), # D8 — Redux ones
    }


def iter_annotated_pairs():
    """Yield (image_path, meta_dict) for all annotated captures."""
    # 1. New captured PNGs (unwarped LCD images, grayscale)
    for yf in sorted(CAPTURED_DIR.glob("*.yaml")):
        with open(yf) as f:
            meta = yaml.safe_load(f) or {}
        if meta.get("PH") is None or meta.get("Redux") is None:
            continue
        for ext in (".png", ".jpg"):
            img_path = yf.with_suffix(ext)
            if img_path.exists():
                yield img_path, meta
                break

    # 2. Legacy hand-labelled JPGs — full frames from data/
    #    The builder will crop and unwarp them using the config.
    for yf in sorted(LEGACY_DIR.glob("*.yaml")):
        with open(yf) as f:
            meta = yaml.safe_load(f) or {}
        if meta.get("PH") is None or meta.get("Redux") is None:
            continue
        for ext in (".png", ".jpg"):
            img_path = yf.with_suffix(ext)
            if img_path.exists():
                yield img_path, meta
                break


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and balance digit dataset.")
    parser.add_argument("--no-balance", action="store_true", help="Skip balancing step.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    config = load_config()
    digit_config = config["lcd"]["digits"]

    # Create per-class directories
    for d in range(10):
        (DATASET_DIR / str(d)).mkdir(parents=True, exist_ok=True)

    saved_paths: dict[int, list[str]] = collections.defaultdict(list)
    success = skipped = errors = 0

    for img_path, meta in iter_annotated_pairs():
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                skipped += 1
                continue

            # Reject overexposed frames (cellar light on)
            if is_overexposed(img):
                skipped += 1
                continue

            # Reject bad AI annotations of overexposed frames (PH == 0.0)
            ph_val = float(meta["PH"])
            if ph_val == 0.0:
                skipped += 1
                continue

            # Decide whether this is already an unwarped LCD crop or a full frame.
            # Captured PNGs (~106×45) are already unwarped; full frames are much larger.
            if img.shape[1] > 300:
                # Legacy full-frame image — crop the LCD region then unwarp
                lcd_crop = crop_rectangle(
                    img,
                    config["lcd"]["rectangle_origin"],
                    config["lcd"]["rectangle_size"],
                )
                try:
                    img, _ = detect_and_unwarp(lcd_crop)
                except RuntimeError:
                    skipped += 1
                    continue

            # The captured PNGs are already unwarped; extract digit ROIs directly.
            rois = extract_digit_rois(img, digit_config)
            digit_labels = get_digit_labels(ph_val, int(meta["Redux"]))

            for idx, label in digit_labels.items():
                roi = rois[idx]
                if roi.size == 0:
                    continue
                fname = f"{img_path.stem}_D{idx}.png"
                out_path = DATASET_DIR / str(label) / fname
                if not out_path.exists():
                    cv2.imwrite(str(out_path), roi)
                saved_paths[label].append(str(out_path))

            success += 1

        except Exception as e:
            print(f"  Error on {img_path.name}: {e}")
            errors += 1

    print(f"\nExtracted from {success} images  (skipped={skipped}, errors={errors})")
    print("\nDigit class distribution (data/dataset/):")
    for d in range(10):
        n = len(saved_paths[d])
        bar = "█" * (n // 5)
        print(f"  {d}: {n:5d}  {bar}")

    missing = [d for d in range(10) if not saved_paths[d]]
    if missing:
        print(f"\n  WARNING: no samples for digit(s): {missing}")
        print("  Collect more data during the day when pool chemistry changes.")

    if args.no_balance:
        print("\nSkipping balancing (--no-balance).")
        return

    # ── Balance ──────────────────────────────────────────────────────────────
    present = {d: v for d, v in saved_paths.items() if v}
    if not present:
        print("No annotated data found.")
        return

    min_count = min(len(v) for v in present.values())
    print(f"\nBalancing to {min_count} samples per present class → {BALANCED_DIR.relative_to(ROOT)}")

    random.seed(args.seed)
    for d in range(10):
        out_dir = BALANCED_DIR / str(d)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = saved_paths[d]
        if not paths:
            print(f"  {d}: 0 samples (digit not seen in data)")
            continue

        sampled = random.sample(paths, min(min_count, len(paths)))
        for src in sampled:
            dst = out_dir / pathlib.Path(src).name
            shutil.copy2(src, dst)
        print(f"  {d}: {len(sampled):5d} samples")

    print(f"\nBalanced dataset saved to {BALANCED_DIR}")


if __name__ == "__main__":
    main()
