# rpi_pool_cam

Monitor swimming pool chemistry (pH and Redux) and fill level using a Raspberry Pi camera.

---

## Overview

A Raspberry Pi with a PiCamera photographs a pool dosing unit display and a led based fill level indicator at one-minute intervals.

The captured image is processed on-device to:

- **Read the LCD display** — perspective-correct the display and extract eight individual digit crops for later classification.
- **Read LED status** — detect which indicator LEDs are active to determine pump state, fill level, and error status.

The LED states are exposed via a JSON endpoint that can be polled by Home Assistant.
The digit crops are saved on disk to build a labelled dataset for training a digit classification model.

---

## Hardware

- Raspberry Pi (any model with a CSI camera port)
- Raspberry Pi Camera Module
- Camera mounted on a wooden frame above the pool dosing unit

> The wooden mount flexes with humidity, so pixel coordinates drift between captures.
> The server compensates for this by detecting the LCD panel's quadrilateral in every frame
> and applying a perspective transform — both to straighten the display and to anchor the LED
> region relative to the detected LCD position.

---

## Architecture

```
PiCamera (1 fps, 2592×1952)
        │
        ▼
 CamImageProcessor
  ├── detect LCD quad  ──► perspective unwarp ──► digit ROI extraction
  └── offset LED crop (anchored to LCD position)
        │
        ▼
  Flask server (port 8000)
  ├── GET /image.jpg          full frame
  ├── GET /digital.jpg        unwarped LCD image
  ├── GET /lcd_digits.jpg     grid of 8 digit crops
  ├── GET /leds_annotated.jpg annotated LED region
  └── GET /leds.json          LED states → Home Assistant
```

---

## Project Structure

```
rpi_pool_cam/
├── config.yaml                      camera ROI and digit grid configuration
├── requirements.txt
├── data/                            labelled test images + ground-truth YAMLs
│   └── captured/                    auto-saved captures (--save-captures mode)
├── src/
│   ├── cam_image_processor/
│   │   ├── cam_image_processor.py   CamImageProcessor class
│   │   └── lcd_detector.py          perspective-unwarp and digit ROI utilities
│   └── server/
│       └── server.py                Flask web server
├── tests/
│   ├── conftest.py                  shared pytest fixtures
│   ├── test_cam_image_processor.py  LED detection tests
│   └── test_lcd_detector.py         LCD unwarp and digit ROI tests
└── jupyter/
    ├── lcd_digit_extraction.ipynb   batch digit extraction for training data
    └── regions_of_interest_definition.ipynb
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the Server

```bash
# Normal operation
python -m src.server.server

# Save every capture to data/captured/ for training data collection
python -m src.server.server --save-captures

# Custom port
python -m src.server.server --port 8080
```

Each saved capture produces:
- `data/captured/{timestamp}.jpg` — full-resolution frame
- `data/captured/{timestamp}.yaml` — auto-detected LED states + empty `PH`/`Redux` fields for manual labelling

---

## HTTP Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | HTML dashboard |
| `GET /image.jpg` | Full camera frame |
| `GET /digital.jpg` | Perspective-corrected LCD |
| `GET /lcd_digits.jpg` | Grid of 8 digit crops (4× scaled) |
| `GET /leds_annotated.jpg` | LED region with detection circles |
| `GET /leds.json` | LED states and fill level as JSON |

Example `/leds.json` response:

```json
{"Valid": "True", "K1": "True", "K2": "True", "K3": "False", "Error": "False", "FillLevel": "75"}
```

---

## Building the Training Dataset

1. Run the server with `--save-captures` for several days to collect diverse images.
2. Manually fill in the `PH` and `Redux` fields in each generated YAML stub.
3. Open `jupyter/lcd_digit_extraction.ipynb` and run all cells — it reads every labelled
   JPEG + YAML pair and saves individual digit crops to `data/digits/{0-9}/`.
4. Use the collected crops to train a digit classification model.

---

## Running Tests

Tests use the images and ground-truth YAMLs in `data/` as fixtures.

```bash
pytest tests/ -v
```

---

## Configuration (`config.yaml`)

| Key | Description |
|---|---|
| `lcd.rectangle_origin` | Top-left of the coarse LCD crop in the full frame |
| `lcd.rectangle_size` | Width/height of the coarse LCD crop |
| `lcd.digits.origin` | Top-left of the first digit in the unwarped LCD |
| `lcd.digits.size` | Width/height of each digit ROI |
| `lcd.digits.num_cols` / `num_rows` | Digit grid layout (4 × 2 = 8 digits) |
| `leds.rectangle_origin` | Top-left of the LED region in the full frame |
| `leds.{name}.origin` | Centre of each LED within the LED crop |
| `leds.{name}.radius` | Detection circle radius for each LED |
