# rpi_pool_cam

Monitor swimming pool chemistry (pH and ORP/Redux) using a Raspberry Pi camera pointed at a pool dosing unit LCD display.

---

## Overview

A Raspberry Pi with a PiCamera photographs a pool dosing unit display and LED-based fill level indicator at one-minute intervals. The captured image is processed on-device to:

- **Read the LCD display** — perspective-correct the panel and classify each digit with a tiny CNN (ONNX, ~30 k parameters, runs on RPi Zero)
- **Filter outliers** — a rolling-median filter rejects single-frame CNN misreads caused by digit superimposition during LCD transitions
- **Read LED status** — detect which indicator LEDs are active to determine pump state, fill level, and error status
- **Serve a JSON API** — polled by Home Assistant every minute

> The camera mount flexes with humidity, so pixel coordinates drift.
> The server compensates by detecting the LCD panel's quadrilateral each frame
> and applying a perspective transform to anchor everything to the display geometry.

---

## Architecture

```
PiCamera (1 fps, 2592×1952)
        │
        ▼
 CamImageProcessor
  ├── detect LCD quad ──► perspective unwarp ──► digit ROI extraction
  └── offset LED crop (anchored to LCD position)
        │
        ▼
  DigitCNN (ONNX via onnxruntime)
  Six digit positions → PH (X.XX) + Redux (XXX mV)
        │
        ▼
  PoolValueFilter  (rolling-median spike rejection)
        │
        ▼
  Flask server (port 8000)
  ├── GET /leds.json          ← polled by Home Assistant
  ├── GET /chart.png          live PH/Redux chart (raw vs filtered)
  ├── GET /history.json       last 60 readings (raw + filtered + accepted flag)
  ├── GET /image.jpg          full frame
  ├── GET /digital.jpg        unwarped LCD image
  ├── GET /lcd_digits.jpg     grid of 8 digit crops
  └── GET /leds_annotated.jpg annotated LED region
```

---

## Project Structure

```
rpi_pool_cam/
├── config.yaml                       camera ROI and digit grid configuration
├── requirements.txt                  runtime dependencies (RPi deployment)
├── requirements-train.txt            training-only dependencies
├── models/
│   ├── digit_cnn.onnx                ONNX model used at runtime
│   ├── digit_cnn.torchscript.pt      TorchScript export (backup)
│   └── digit_cnn_best.pt             PyTorch weights of best checkpoint
├── data/
│   └── captured/                     auto-saved LCD crops + annotation YAMLs
├── src/
│   ├── cam_image_processor/
│   │   ├── cam_image_processor.py    CamImageProcessor class
│   │   └── lcd_detector.py           perspective-unwarp and digit ROI utilities
│   └── server/
│       └── server.py                 Flask server + PoolValueFilter
├── scripts/
│   ├── annotate_captures.py          auto-annotate captures via OpenRouter vision API
│   ├── build_dataset.py              extract digit crops from annotated captures
│   ├── check_annotations.py          temporal outlier detection / cleanup
│   ├── eval_e2e.py                   end-to-end accuracy evaluation on full images
│   ├── export_onnx.py                re-export PyTorch weights to ONNX
│   ├── filter_demo.py                generate raw-vs-filtered charts from E2E CSV
│   └── train_cnn.py                  train the digit CNN (image-level E2E validation)
├── tests/
│   ├── conftest.py
│   ├── test_cam_image_processor.py
│   └── test_lcd_detector.py
└── jupyter/
    ├── check_cam_image_processor.ipynb
    ├── lcd_digit_extraction.ipynb
    └── regions_of_interest_definition.ipynb
```

---

## Deployment (Raspberry Pi)

```bash
# First time
pip install -r requirements.txt

# Pull latest (model + server updates)
git pull
python -m src.server.server --save-captures
```

The `--save-captures` flag writes every processed LCD crop plus a YAML stub to `data/captured/` for later use as training data.

---

## HTTP Endpoints

| Endpoint | Description |
|---|---|
| `GET /leds.json` | LED states, fill level, PH, Redux, accepted flag |
| `GET /chart.png` | Dark-mode time-series chart: raw CNN output vs filtered |
| `GET /history.json` | Last 60 readings with raw, filtered and accepted fields |
| `GET /image.jpg` | Full camera frame |
| `GET /digital.jpg` | Perspective-corrected LCD crop |
| `GET /lcd_digits.jpg` | 4× scaled grid of all 8 digit ROIs |
| `GET /leds_annotated.jpg` | LED region with detection circles |

Example `/leds.json` response:

```json
{
  "Valid": "True",
  "Overexposed": "False",
  "K1": "True", "K2": "False", "K3": "False",
  "Error": "False",
  "FillLevel": "75",
  "PH": 7.18,
  "Redux": 774,
  "PHReduxAccepted": "True"
}
```

`PHReduxAccepted` is `"False"` when the rolling-median filter rejects a reading as an isolated spike. Home Assistant uses this as an availability condition so rejected readings never reach the UI.

---

## Spike Filter

`PoolValueFilter` in `server.py` keeps a rolling window of the last 5 accepted readings and rejects any new reading that deviates more than **0.20 pH** or **35 mV** from their median. This catches:

- Digit superimposition during LCD transitions (e.g. Redux reading of 296 instead of 796)
- Partial LCD updates where one digit has not yet settled
- Random CNN misclassifications on borderline images

Genuine chemistry changes (e.g. acid dosing) are typically ≤0.05 pH/min, so they accumulate gradually through the filter rather than being rejected.

---

## Training Pipeline

### 1. Collect data

```bash
python -m src.server.server --save-captures
```

### 2. Auto-annotate captures

```bash
export OPENROUTER_API_KEY=sk-or-...
python scripts/annotate_captures.py --day-interval 1 --night-interval 10
```

Uses a cheap vision model (~$0.03/1000 images) to fill in PH and Redux ground-truth labels.

### 3. Clean annotations

```bash
# Dry run — review flagged frames
python scripts/check_annotations.py

# Apply fixes (auto-correct obvious OCR errors, null isolated spikes)
python scripts/check_annotations.py --fix --null-warnings
```

### 4. Build digit crop dataset

```bash
python scripts/build_dataset.py
```

Extracts 6 digit ROIs per image into `data/dataset/{0-9}/` and produces a balanced subset in `data/dataset_balanced/`.

### 5. Train

```bash
conda run -n gs2026 python scripts/train_cnn.py --epochs 200
```

Uses **image-level train/val split** — validation runs the full E2E pipeline (full image → ROI extraction → CNN → PH/Redux comparison) on held-out images, not just digit crops. Best checkpoint selected on E2E "both exact" accuracy.

### 6. Evaluate

```bash
python scripts/eval_e2e.py --csv results/e2e.csv
```

### 7. Deploy

Copy `models/digit_cnn.onnx` to the RPi (or `git pull` if the model is committed).

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration (`config.yaml`)

| Key | Description |
|---|---|
| `lcd.rectangle_origin` | Top-left of the coarse LCD crop in the full frame |
| `lcd.rectangle_size` | Size of the coarse LCD crop |
| `lcd.screen` | Four corner points of the LCD panel for perspective warp |
| `lcd.digits.origin` | Top-left of the first digit in the unwarped LCD |
| `lcd.digits.size` | Width × height of each digit ROI (12×18 px) |
| `lcd.digits.num_cols` / `num_rows` | Digit grid layout (4 × 2) |
| `leds.rectangle_origin` | Top-left of the LED region in the full frame |
| `leds.{name}.origin` | Centre of each LED within the LED crop |
| `leds.{name}.radius` | Detection circle radius |
