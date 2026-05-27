#!/usr/bin/env python3
"""Annotate captured images using OpenRouter vision API.

For each PNG in data/captured/ whose YAML has PH: null, crops the LCD region,
sends it to a cheap vision model, parses PH and Redux, and writes them back.

Usage:
    # Preview first 5 images without writing
    python scripts/annotate_captures.py --preview 5

    # Annotate all unannotated captures
    python scripts/annotate_captures.py

    # Annotate with higher concurrency
    python scripts/annotate_captures.py --delay 0.05
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import sys
import time
from datetime import datetime, time as dtime

import cv2
import requests
import yaml

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Configuration ────────────────────────────────────────────────────────────

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# qwen/qwen3-vl-8b-instruct — cheap vision model (~$0.08/M), ~$0.03 for all 674 images
MODEL = "qwen/qwen3-vl-8b-instruct"

CAPTURED_DIR = ROOT / "data" / "captured"

PROMPT = (
    "This image is a perspective-corrected crop of an LCD panel from a pool "
    "dosing unit. The top row displays the pH value (format X.XX, e.g. '7.16') "
    "and the bottom row displays the ORP/Redux value as an integer (e.g. '796'). "
    "Respond ONLY with a JSON object and nothing else, like: "
    '{"PH": 7.16, "Redux": 796}'
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def encode_image(img: "cv2.Mat") -> str:
    """Scale up 6× (nearest-neighbour keeps segments crisp) and encode as base64 JPEG."""
    large = cv2.resize(img, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    _, buf = cv2.imencode(".jpg", large, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buf.tobytes()).decode()


def _time_bucket(yf: pathlib.Path, night_min: int, day_min: int) -> tuple:
    """Return a bucket key so that only one file per interval is kept.

    Night (18:00–08:00): one sample per *night_min* minutes.
    Day   (08:00–18:00): one sample per *day_min* minutes.
    """
    try:
        dt = datetime.strptime(yf.stem, "%Y%m%d_%H%M%S")
    except ValueError:
        return (yf.stem,)   # non-timestamped file — always include
    t = dt.time()
    is_night = t >= dtime(18, 0) or t < dtime(8, 0)
    interval = night_min if is_night else day_min
    bucket = (dt.hour * 60 + dt.minute) // interval
    return (dt.date(), bucket)


def subsample(yaml_files: list[pathlib.Path], night_min: int, day_min: int) -> list[pathlib.Path]:
    """Keep the first YAML in each time bucket."""
    seen: set = set()
    result = []
    for yf in yaml_files:
        key = _time_bucket(yf, night_min, day_min)
        if key not in seen:
            seen.add(key)
            result.append(yf)
    return result


def query_openrouter(b64_image: str) -> dict:
    """Call OpenRouter and return parsed JSON with PH and Redux."""
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            "max_tokens": 60,
        },
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    # Robustly extract the JSON object even if the model adds prose
    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in response: {content!r}")
    return json.loads(content[start:end])


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate captured LCD images via OpenRouter.")
    parser.add_argument(
        "--preview",
        type=int,
        metavar="N",
        default=0,
        help="Dry-run: process only N images and print results without saving.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        help="Seconds to wait between API requests (default: 0.15).",
    )
    parser.add_argument(
        "--night-interval",
        type=int,
        default=60,
        metavar="MIN",
        help="Minutes between samples during 18:00–08:00 (default: 60).",
    )
    parser.add_argument(
        "--day-interval",
        type=int,
        default=5,
        metavar="MIN",
        help="Minutes between samples during 08:00–18:00 (default: 5).",
    )
    args = parser.parse_args()

    yaml_files = sorted(CAPTURED_DIR.glob("*.yaml"))

    unannotated: list[pathlib.Path] = []
    for yf in yaml_files:
        with open(yf) as f:
            data = yaml.safe_load(f) or {}
        if data.get("PH") is None:
            unannotated.append(yf)

    # Apply time-based subsampling to unannotated files
    selected = subsample(unannotated, args.night_interval, args.day_interval)

    print(f"Total YAML files:   {len(yaml_files)}")
    print(f"Unannotated:        {len(unannotated)}")
    print(f"After subsampling (night≥{args.night_interval}min, day≥{args.day_interval}min): {len(selected)}")

    if args.preview:
        print(f"\n[PREVIEW MODE — first {args.preview} images, no writes]\n")
        selected = selected[: args.preview]

    success = 0
    failed: list[tuple[str, str]] = []

    for i, yf in enumerate(selected, 1):
        png_path = yf.with_suffix(".png")
        if not png_path.exists():
            png_path = yf.with_suffix(".jpg")
        if not png_path.exists():
            failed.append((yf.name, "image file not found"))
            continue

        try:
            # Captures are already perspective-corrected LCD images (~106×45 px grayscale)
            img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                failed.append((yf.name, "cv2.imread returned None"))
                continue

            b64 = encode_image(img)
            result = query_openrouter(b64)

            ph = round(float(result["PH"]), 2)
            redux = int(result["Redux"])

            print(f"[{i:4d}/{len(selected)}] {yf.stem}  PH={ph}  Redux={redux}")

            if not args.preview:
                with open(yf) as f:
                    meta = yaml.safe_load(f) or {}
                meta["PH"] = ph
                meta["Redux"] = redux
                with open(yf, "w") as f:
                    yaml.dump(meta, f, default_flow_style=False, allow_unicode=True, sort_keys=True)

            success += 1

        except requests.HTTPError as e:
            msg = f"HTTP {e.response.status_code}"
            if e.response.status_code == 429:
                print(f"  Rate-limited — sleeping 5s …")
                time.sleep(5)
                failed.append((yf.name, msg))
            else:
                failed.append((yf.name, msg))
                print(f"  FAILED {yf.name}: {msg}")

        except Exception as e:
            failed.append((yf.name, str(e)))
            print(f"  FAILED {yf.name}: {e}")

        time.sleep(args.delay)

    print(f"\n{'[PREVIEW] ' if args.preview else ''}Done: {success} annotated, {len(failed)} failed.")
    if failed:
        print("Failed items:")
        for name, reason in failed[:20]:
            print(f"  {name}: {reason}")
        if len(failed) > 20:
            print(f"  … and {len(failed) - 20} more")


if __name__ == "__main__":
    main()
