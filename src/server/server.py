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
    GET /leds.json          LED states, fill level, PH, Redux and overexposure state as JSON.
    GET /history.json       Last N raw CNN readings + their accepted/rejected status.
    GET /chart.png          Time-series chart of raw vs filtered PH and Redux.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import io
import json
import logging
import statistics
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
# Pool-value plausibility filter — rolling-median spike rejection
# ---------------------------------------------------------------------------


class PoolValueFilter:
    """Reject CNN digit readings that are isolated spikes relative to recent history.

    Uses the same principle as the annotation cleaner: each new reading is
    compared against the median of the last ``WINDOW`` *accepted* readings.
    A single-frame spike (e.g. from a superimposed-digit exposure artifact) will
    deviate far from that median and be rejected.  Genuine gradual chemistry
    changes move slowly enough that each accepted step is small and the rolling
    median tracks them correctly.

    Every call to :meth:`update` records one raw reading (accepted or not).
    Use :meth:`history` to retrieve the full log for the chart endpoint.
    """

    WINDOW: int = 5           # number of accepted readings for the rolling median
    MAX_GAP_SEC: float = 600  # reset window after a gap longer than this
    PH_MIN, PH_MAX = 5.0, 10.0
    RX_MIN, RX_MAX = 0, 999
    # Spike thresholds — must catch OCR errors (Δ≥0.3 pH, Δ≥20 Redux)
    # while tracking genuine gradual chemistry changes (≤0.08 pH/min, ≤5 Redux/min)
    PH_SPIKE: float = 0.20    # max deviation from rolling median before rejection
    RX_SPIKE: int   = 35      # max deviation from rolling median before rejection
    # History kept for charting
    MAX_HISTORY: int = 1440   # ~24 h at 1 reading/min

    def __init__(self) -> None:
        # Rolling buffer of accepted (ph, rx, ts) for median computation
        self._accepted: collections.deque[tuple[float, int, datetime.datetime]] = \
            collections.deque(maxlen=self.WINDOW)
        # Full log: each entry is a dict with keys ts, ph_raw, rx_raw, ph_filtered,
        # rx_filtered, accepted (bool)
        self._history: collections.deque[dict] = \
            collections.deque(maxlen=self.MAX_HISTORY)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, ph: float, redux: int,
               ts: datetime.datetime | None = None) -> bool:
        """Record a new CNN reading; return True if it passes the filter."""
        ts = ts or datetime.datetime.now()

        # Absolute range check first
        if not (self.PH_MIN <= ph <= self.PH_MAX) or \
           not (self.RX_MIN <= redux <= self.RX_MAX):
            logger.warning("PoolFilter: out-of-range ph=%.2f rx=%d — rejected", ph, redux)
            self._record(ts, ph, redux, accepted=False)
            return False

        with self._lock:
            accepted = self._spike_check(ph, redux, ts)
            ph_f, rx_f = self._filtered_values()
            if accepted:
                self._accepted.append((ph, redux, ts))
                ph_f, rx_f = ph, redux

        self._record(ts, ph, redux, accepted=accepted, ph_f=ph_f, rx_f=rx_f)
        if not accepted:
            logger.info(
                "PoolFilter: spike rejected ph=%.2f rx=%d (median ph=%.2f rx=%d)",
                ph, redux, ph_f if ph_f is not None else float("nan"),
                rx_f if rx_f is not None else -1,
            )
        return accepted

    @property
    def last_ph(self) -> float | None:
        with self._lock:
            return self._accepted[-1][0] if self._accepted else None

    @property
    def last_rx(self) -> int | None:
        with self._lock:
            return self._accepted[-1][1] if self._accepted else None

    def history(self, n: int | None = None) -> list[dict]:
        """Return the last *n* recorded readings (or all if n is None)."""
        items = list(self._history)
        return items[-n:] if n is not None else items

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spike_check(self, ph: float, redux: int,
                     ts: datetime.datetime) -> bool:
        """Return True if the reading passes the spike test."""
        if not self._accepted:
            return True   # no history yet — accept unconditionally

        # Check for time gap: if last accepted reading is old, reset
        last_ts = self._accepted[-1][2]
        if (ts - last_ts).total_seconds() > self.MAX_GAP_SEC:
            self._accepted.clear()
            return True

        ph_vals  = [a[0] for a in self._accepted]
        rx_vals  = [a[1] for a in self._accepted]
        med_ph   = statistics.median(ph_vals)
        med_rx   = statistics.median(rx_vals)

        if abs(ph - med_ph) > self.PH_SPIKE:
            logger.debug("PoolFilter: PH spike %.2f vs median %.2f", ph, med_ph)
            return False
        if abs(redux - med_rx) > self.RX_SPIKE:
            logger.debug("PoolFilter: Rx spike %d vs median %.0f", redux, med_rx)
            return False
        return True

    def _filtered_values(self) -> tuple[float | None, int | None]:
        """Return the last accepted ph/rx (for charting rejected readings)."""
        if not self._accepted:
            return None, None
        return self._accepted[-1][0], self._accepted[-1][1]

    def _record(self, ts: datetime.datetime, ph: float, redux: int,
                accepted: bool,
                ph_f: float | None = None,
                rx_f: int | None = None) -> None:
        self._history.append({
            "ts":          ts.isoformat(timespec="seconds"),
            "ph_raw":      ph,
            "rx_raw":      redux,
            "ph_filtered": ph_f,
            "rx_filtered": rx_f,
            "accepted":    accepted,
        })

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
    pool_filter = PoolValueFilter()

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
            '<p><a href="/chart.png">📈 PH/Redux chart (raw vs filtered)</a></p>'
            '<img src="/chart.png" style="max-width:100%"><br>'
            '<p><a href="/history.json">history.json</a></p>'
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

        overexposed: bool = getattr(cip, "overexposed", False)
        led_states = cip.get_led_dict()
        valid = bool(led_states)

        fill_level = 0
        if led_states.get("S1"):
            fill_level = 100
        elif led_states.get("S2"):
            fill_level = 75
        elif led_states.get("S3"):
            fill_level = 50
        elif led_states.get("S4"):
            fill_level = 25

        payload: dict[str, Any] = {
            "Valid": str(valid),
            "Overexposed": str(overexposed),
            "K1": str(led_states.get("K1", False)),
            "K2": str(led_states.get("K2", False)),
            "K3": str(led_states.get("K3", False)),
            "Error": str(led_states.get("Error", False)),
            "FillLevel": str(fill_level),
            "PH": None,
            "Redux": None,
            "PHReduxAccepted": "False",
        }

        if cip.valid_image:
            try:
                ph, redux = cip.get_ph_redux()
                accepted = pool_filter.update(ph, redux)
                payload["PH"] = ph
                payload["Redux"] = redux
                payload["PHReduxAccepted"] = str(accepted)
                if not accepted:
                    logger.info("leds.json: CNN read ph=%.2f redux=%d but filter rejected it", ph, redux)
            except Exception as exc:
                logger.warning("leds.json: digit CNN inference failed: %s", exc)

        resp = Response(json.dumps(payload), mimetype="application/json")
        resp.headers.update(_NO_CACHE_HEADERS)
        return resp

    @app.route("/history.json")
    def history_json() -> Response:
        """Return the last 60 raw + filtered readings as JSON."""
        data = pool_filter.history(n=60)
        resp = Response(json.dumps(data), mimetype="application/json")
        resp.headers.update(_NO_CACHE_HEADERS)
        return resp

    @app.route("/chart.png")
    def chart_png() -> Response:
        """Return a PNG time-series chart: raw CNN output vs filtered output."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        data = pool_filter.history()
        if not data:
            # Return a blank 1×1 PNG
            buf = io.BytesIO()
            plt.figure(figsize=(1, 1))
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            return Response(buf.read(), mimetype="image/png",
                            headers=_NO_CACHE_HEADERS)

        ts       = [datetime.datetime.fromisoformat(d["ts"]) for d in data]
        ph_raw   = [d["ph_raw"]      for d in data]
        rx_raw   = [d["rx_raw"]      for d in data]
        ph_filt  = [d["ph_filtered"] for d in data]
        rx_filt  = [d["rx_filtered"] for d in data]
        accepted = [d["accepted"]    for d in data]

        # Split into accepted / rejected for scatter
        ts_acc = [t for t, a in zip(ts, accepted) if a]
        ph_acc = [v for v, a in zip(ph_raw, accepted) if a]
        rx_acc = [v for v, a in zip(rx_raw, accepted) if a]
        ts_rej = [t for t, a in zip(ts, accepted) if not a]
        ph_rej = [v for v, a in zip(ph_raw, accepted) if not a]
        rx_rej = [v for v, a in zip(rx_raw, accepted) if not a]

        # Filtered line (skip None)
        ts_f  = [t for t, v in zip(ts, ph_filt) if v is not None]
        ph_f  = [v for v in ph_filt  if v is not None]
        rx_f  = [v for v in rx_filt  if v is not None]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.patch.set_facecolor("#1e1e2e")
        for ax in (ax1, ax2):
            ax.set_facecolor("#1e1e2e")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")

        # PH panel
        ax1.scatter(ts_acc, ph_acc, s=6, color="#6688cc", alpha=0.5, label="raw (accepted)", zorder=2)
        ax1.scatter(ts_rej, ph_rej, s=20, color="#ff4444", marker="x", linewidths=1.2,
                    label="raw (rejected)", zorder=3)
        if ts_f:
            ax1.plot(ts_f, ph_f, color="#88ddff", linewidth=1.5, label="filtered", zorder=4)
        ax1.set_ylabel("pH", color="white")
        ax1.legend(fontsize=8, facecolor="#2a2a3e", labelcolor="white",
                   loc="upper left", framealpha=0.7)
        ax1.grid(color="#333355", linewidth=0.5)
        if ph_acc:
            ax1.set_ylim(min(ph_acc) - 0.15, max(ph_acc) + 0.15)

        # Redux panel
        ax2.scatter(ts_acc, rx_acc, s=6, color="#66bb88", alpha=0.5, label="raw (accepted)", zorder=2)
        ax2.scatter(ts_rej, rx_rej, s=20, color="#ff4444", marker="x", linewidths=1.2,
                    label="raw (rejected)", zorder=3)
        if ts_f:
            ax2.plot(ts_f, rx_f, color="#aaffcc", linewidth=1.5, label="filtered", zorder=4)
        ax2.set_ylabel("Redux (mV)", color="white")
        ax2.legend(fontsize=8, facecolor="#2a2a3e", labelcolor="white",
                   loc="upper left", framealpha=0.7)
        ax2.grid(color="#333355", linewidth=0.5)
        if rx_acc:
            ax2.set_ylim(min(rx_acc) - 10, max(rx_acc) + 10)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=30)
        fig.suptitle("Pool chemistry — raw CNN vs filtered", color="white", fontsize=11)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return Response(buf.read(), mimetype="image/png",
                        headers=_NO_CACHE_HEADERS)

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
