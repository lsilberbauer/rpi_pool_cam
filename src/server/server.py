# Copyright (c) 2026 Lukas Silberbauer. All rights reserved.

"""Flask web server exposing live camera images and pool sensor data.

Run directly::

    python -m src.server.server [--save-captures]

The ``--save-captures`` flag writes every captured frame as a JPEG and a
corresponding YAML stub (with auto-detected LED states, empty PH/Redux fields)
to ``data/captured/``.  The YAML stubs can be filled in manually later and then
fed to the digit-extraction notebook to build the training dataset.

Endpoints:
    GET /                   HTML dashboard linking all image endpoints.
    GET /image.jpg          Full-resolution camera frame.
    GET /digital.jpg        Perspective-corrected (unwarped) LCD image.
    GET /lcd_digits.jpg     Grid visualisation of the eight digit ROIs.
    GET /leds.jpg           Raw LED region crop (no annotation).
    GET /leds_annotated.jpg LED region with annotated circles.
    GET /leds.json          LED states and derived fill level as JSON.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import threading
import time
from pathlib import Path
from threading import Thread
from time import localtime, strftime
from typing import Any

import cv2
import numpy as np
import yaml
from flask import Flask, Response, jsonify

from src.cam_image_processor import CamImageProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path("config.yaml")
_CAPTURED_DIR = Path("data/captured")

_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, must-revalidate",
    "Expires": "0",
}

# ---------------------------------------------------------------------------
# Image stream — thread-safe frame + processor store
# ---------------------------------------------------------------------------


class ImageStream:
    """Thread-safe container for the latest camera frame and its processor.

    A background capture thread calls :meth:`update` after each capture.
    Request handlers call :meth:`frame` and :meth:`processor` to read the
    latest data without blocking each other.
    """

    def __init__(self) -> None:
        self._frame: np.ndarray | None = None
        self._processor: CamImageProcessor | None = None
        self._lock = threading.Lock()

    def update(self, frame: np.ndarray, config: dict[str, Any]) -> None:
        """Store a new frame and rebuild the image processor.

        Args:
            frame: Full-resolution BGR camera frame.
            config: Parsed ``config.yaml`` mapping.
        """
        processor = CamImageProcessor(config, frame)
        with self._lock:
            self._frame = frame
            self._processor = processor
        logger.debug("ImageStream updated.")

    def frame(self) -> np.ndarray | None:
        """Return the latest frame, or None if no frame has been captured yet."""
        with self._lock:
            return self._frame

    def processor(self) -> CamImageProcessor | None:
        """Return the latest processor, or None if not yet initialised."""
        with self._lock:
            return self._processor


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------


def capture_loop(stream: ImageStream, config: dict[str, Any], save_captures: bool) -> None:
    """Continuously capture frames from the PiCamera and push them to the stream.

    Camera settings (shutter speed, ISO, rotation) are fixed to the values
    established during initial project development.  The camera is opened and
    closed on every iteration to avoid resource leaks over long runtimes.

    Args:
        stream: Shared :class:`ImageStream` instance.
        config: Parsed ``config.yaml`` mapping.
        save_captures: When True, each frame is saved as a JPEG together with
            a YAML stub to ``data/captured/``.
    """
    # Deferred import: picamera is only available on the Raspberry Pi.
    from picamera import PiCamera  # type: ignore[import]

    if save_captures:
        _CAPTURED_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            camera = PiCamera(resolution=(2592, 1952), framerate=1, sensor_mode=3)
            camera.led = False
            camera.shutter_speed = 1_000_000
            camera.iso = 800
            camera.rotation = 180
            time.sleep(5)
            camera.exposure_mode = "off"

            raw = np.empty((1952 * 2592 * 3,), dtype=np.uint8)
            camera.annotate_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            camera.capture(raw, format="bgr")
            image = raw.reshape((1952, 2592, 3))
            camera.close()
        except Exception:
            logger.error("Camera capture failed.", exc_info=True)
            time.sleep(60)
            continue

        stream.update(image, config)

        if save_captures:
            _save_capture(stream.processor())

        time.sleep(60)


def _save_capture(processor: CamImageProcessor | None) -> None:
    """Write the unwarped LCD image and a YAML stub to ``data/captured/``.

    Saves only the perspective-corrected LCD crop (grayscale, ~44×106 px) rather
    than the full frame.  This keeps disk usage to a few KB per capture instead
    of ~1.4 MB, which is critical for long-running operation on a Raspberry Pi.

    The YAML stub contains auto-detected LED states and empty ``PH``/``Redux``
    fields for manual labelling.  If the processor is unavailable or LCD
    unwarping fails, the capture is skipped and the error is logged — no fallback
    to full-frame saving.

    Args:
        processor: Processor built from the current frame, or None.
    """
    if processor is None:
        logger.error("Skipping capture save: processor not available.")
        return

    try:
        unwarped = processor.get_unwarped_lcd()
    except RuntimeError:
        logger.error("Skipping capture save: LCD unwarp failed.", exc_info=True)
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = _CAPTURED_DIR / f"{timestamp}.png"
    yaml_path = _CAPTURED_DIR / f"{timestamp}.yaml"

    success = cv2.imwrite(str(png_path), unwarped)
    if not success:
        logger.error("Failed to write unwarped LCD image: %s", png_path)
        return

    led_states: dict[str, Any] = {}
    led_dict = processor.get_led_dict()
    if led_dict:
        led_states = {k: bool(v) for k, v in led_dict.items()}

    stub: dict[str, Any] = {**led_states, "PH": None, "Redux": None}
    yaml_path.write_text(yaml.dump(stub, default_flow_style=False))
    logger.info("Saved capture: %s (unwarped LCD %dx%d)", timestamp, unwarped.shape[1], unwarped.shape[0])


# ---------------------------------------------------------------------------
# Flask application factory
# ---------------------------------------------------------------------------


def create_app(stream: ImageStream, config: dict[str, Any]) -> Flask:
    """Create and configure the Flask application.

    Args:
        stream: Shared :class:`ImageStream` instance populated by the capture
            thread.
        config: Parsed configuration dictionary from config.yaml.

    Returns:
        Configured Flask application.
    """
    app = Flask(__name__)

    def _jpeg_response(image: np.ndarray) -> Response:
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            return Response("Failed to encode image.", status=500)
        resp = Response(encoded.tobytes(), mimetype="image/jpeg")
        resp.headers.update(_NO_CACHE_HEADERS)
        return resp

    @app.route("/")
    def index() -> Response:
        html = (
            "<html><body>"
            '<p><a href="/image.jpg">Full frame</a></p>'
            '<img src="/digital.jpg"><br>'
            '<img src="/lcd_digits.jpg"><br>'
            '<img src="/leds_annotated.jpg"><br>'
            "</body></html>"
        )
        return Response(html, mimetype="text/html")

    @app.route("/image.jpg")
    def full_image() -> Response:
        frame = stream.frame()
        if frame is None:
            return Response("No frame available.", status=503)
        return _jpeg_response(frame)

    @app.route("/digital.jpg")
    def digital() -> Response:
        """Return the perspective-corrected (unwarped) LCD as a JPEG."""
        cip = stream.processor()
        if cip is None:
            return Response("No frame available.", status=503)
        try:
            unwarped = cip.get_unwarped_lcd()
        except RuntimeError as exc:
            logger.error("LCD unwarp failed: %s", exc)
            return Response(str(exc), status=422)

        # Fit the unwarped LCD into a black canvas matching the LED crop dimensions
        # so both images have identical resolution and timestamp style.
        canvas_h: int = config["leds"]["rectangle_size"][1]
        canvas_w: int = config["leds"]["rectangle_size"][0]
        fit_scale = min(canvas_w / unwarped.shape[1], canvas_h / unwarped.shape[0])
        new_w = int(unwarped.shape[1] * fit_scale)
        new_h = int(unwarped.shape[0] * fit_scale)
        resized = cv2.resize(unwarped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        y_off = (canvas_h - new_h) // 2
        x_off = (canvas_w - new_w) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

        display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        timestamp = strftime("%H:%M:%S", localtime())
        cv2.putText(display, timestamp, (2, display.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 155, 155), 1)
        return _jpeg_response(display)

    @app.route("/lcd_digits.jpg")
    def lcd_digits() -> Response:
        """Return a grid image showing all eight digit ROIs side by side."""
        cip = stream.processor()
        if cip is None:
            return Response("No frame available.", status=503)
        try:
            rois = cip.get_digit_rois()
        except RuntimeError as exc:
            logger.error("Digit ROI extraction failed: %s", exc)
            return Response(str(exc), status=422)

        # Lay out all ROIs in a single row, separated by a 1-pixel gap
        gap = np.zeros((rois[0].shape[0], 1), dtype=np.uint8)
        row_parts: list[np.ndarray] = []
        for roi in rois:
            row_parts.append(roi)
            row_parts.append(gap)
        grid = np.hstack(row_parts[:-1])  # drop trailing gap

        # Scale up 4× for visibility
        scale = 4
        grid_scaled = cv2.resize(grid, (grid.shape[1] * scale, grid.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        return _jpeg_response(cv2.cvtColor(grid_scaled, cv2.COLOR_GRAY2BGR))

    @app.route("/leds.jpg")
    def leds_raw() -> Response:
        """Return the raw LED region crop with timestamp."""
        frame = stream.frame()
        if frame is None:
            return Response("No frame available.", status=503)
        origin = config["leds"]["rectangle_origin"]
        size = config["leds"]["rectangle_size"]
        x, y = origin[0], origin[1]
        w, h = size[0], size[1]
        crop = frame[y : y + h, x : x + w].copy()
        timestamp = strftime("%H:%M:%S", localtime())
        cv2.putText(crop, timestamp, (2, crop.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 155, 155), 1)
        return _jpeg_response(crop)

    @app.route("/leds_annotated.jpg")
    def leds_annotated() -> Response:
        cip = stream.processor()
        if cip is None:
            return Response("No frame available.", status=503)
        annotated = cip.get_led_annotations()
        if annotated is None:
            return Response("Image invalid — cannot annotate LEDs.", status=422)
        timestamp = strftime("%H:%M:%S", localtime())
        cv2.putText(annotated, timestamp, (2, annotated.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 155, 155), 1)
        return _jpeg_response(annotated)

    @app.route("/leds.json")
    @app.route("/leds_json")  # legacy alias — kept for Home Assistant compatibility
    def leds_json() -> Response:
        cip = stream.processor()
        if cip is None:
            return Response("No frame available.", status=503)
        led_json = cip.get_led_json()
        if not led_json:
            return Response("{}", mimetype="application/json", headers=_NO_CACHE_HEADERS)
        resp = Response(led_json, mimetype="application/json")
        resp.headers.update(_NO_CACHE_HEADERS)
        return resp

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pool camera web server")
    parser.add_argument(
        "--save-captures",
        action="store_true",
        help="Save each captured frame and a LED YAML stub to data/captured/.",
    )
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    return parser.parse_args()


def main() -> None:
    """Start the capture thread and serve the Flask app."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    args = _parse_args()

    with open(_CONFIG_PATH) as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    stream = ImageStream()

    capture_thread = Thread(target=capture_loop, args=(stream, config, args.save_captures), daemon=True)
    capture_thread.start()
    logger.info("Capture thread started (save_captures=%s).", args.save_captures)

    app = create_app(stream, config)
    logger.info("Serving on http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
