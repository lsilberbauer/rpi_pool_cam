# Copyright (c) 2026 Lukas Silberbauer. All rights reserved.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.cam_image_processor.lcd_detector import detect_and_unwarp, extract_digit_rois

# Lazily loaded ONNX inference session for digit classification (loaded once, shared across instances).
_digit_session: Any = None

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def draw_rectangle(
    image: np.ndarray,
    rectangle_origin: list[int],
    rectangle_size: list[int],
) -> np.ndarray:
    """Draw a rectangle outline on an image and return the result.

    Args:
        image: Input image (BGR).
        rectangle_origin: Top-left corner as [x, y].
        rectangle_size: Dimensions as [width, height].

    Returns:
        Copy of the image with the rectangle drawn in cyan.
    """
    color = (0, 255, 255)
    thickness = 2
    start_point = (rectangle_origin[0], rectangle_origin[1])
    end_point = (rectangle_origin[0] + rectangle_size[0], rectangle_origin[1] + rectangle_size[1])
    return cv2.rectangle(image, start_point, end_point, color, thickness)


def crop_rectangle(
    image: np.ndarray,
    rectangle_origin: list[int],
    rectangle_size: list[int],
) -> np.ndarray:
    """Crop a rectangular region from an image.

    Args:
        image: Source image.
        rectangle_origin: Top-left corner as [x, y].
        rectangle_size: Dimensions as [width, height].

    Returns:
        Cropped image region (view, not a copy).
    """
    x, y = rectangle_origin
    w, h = rectangle_size
    return image[y : y + h, x : x + w]


def annotate_led(
    img: np.ndarray,
    origin: list[int],
    radius: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw a circle annotation for an LED on a copy of the image.

    Args:
        img: Source image (BGR).
        origin: Circle centre as [x, y].
        radius: Circle radius in pixels.
        color: BGR colour tuple.

    Returns:
        New image with the circle drawn.
    """
    annotated = img.copy()
    cv2.circle(annotated, (origin[0], origin[1]), radius, color, 1)
    return annotated


def get_led_color(
    img: np.ndarray,
    origin: list[int],
    radius: int,
) -> tuple[float, float, float]:
    """Compute the mean BGR colour inside a circular LED mask.

    Args:
        img: Source image (BGR).
        origin: Circle centre as [x, y].
        radius: Circle radius in pixels.

    Returns:
        Mean (B, G, R) colour values as a three-element tuple.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    cv2.circle(mask, (origin[0], origin[1]), radius, 255, -1)
    return cv2.mean(img, mask=mask)[:3]


class CamImageProcessor:
    """Processes a single camera frame to extract LCD and LED state information.

    On construction the LCD region is located, the LED region is offset-corrected
    using the detected LCD anchor, and all derived data is computed.  If the image
    is unsuitable (e.g. multiple LCD contours), ``valid_image`` is set to ``False``
    and all data accessors return empty/``None`` results rather than raising.

    Args:
        config: Parsed ``config.yaml`` dictionary.
        img: Full BGR camera frame as a NumPy array.
    """

    def __init__(self, config: dict[str, Any], img: np.ndarray) -> None:
        self.config = config
        self.valid_image = True
        self.overexposed = False

        self.img = img
        self._lcd_gray: np.ndarray | None = None
        self._unwarped_lcd: np.ndarray | None = None
        self._lcd_quad: np.ndarray | None = None
        self._leds: np.ndarray | None = None

        # Crop the coarse LCD region as configured
        lcd_crop = crop_rectangle(
            img,
            self.config["lcd"]["rectangle_origin"],
            self.config["lcd"]["rectangle_size"],
        )

        # --- LED region anchor correction (existing mechanism, unchanged) ---
        gray = cv2.cvtColor(lcd_crop, cv2.COLOR_BGR2GRAY)

        # Detect overexposure before contour analysis (cellar light on → all pixels white).
        # approxPolyDP may still find a contour on a uniformly bright crop, but the
        # unwarped image would be meaningless, so we bail out early.
        if (gray > 240).mean() > 0.5:
            logger.warning("LCD crop is overexposed (%.0f%% bright pixels) — skipping processing.",
                           (gray > 240).mean() * 100)
            self.overexposed = True
            self.valid_image = False
            return

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 1:
            logger.error("Expected exactly 1 LCD contour, found %d. Marking image invalid.", len(contours))
            self.valid_image = False
            return

        perimeter = cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], 0.04 * perimeter, True)

        lcd_offset = approx[0][0] - (17, 17)
        offsetted_origin = tuple(map(sum, zip(self.config["leds"]["rectangle_origin"], lcd_offset)))
        self._leds = crop_rectangle(img, list(offsetted_origin), self.config["leds"]["rectangle_size"])

        # Grayscale LCD crop stored for lazy perspective-correction
        self._lcd_gray = gray

    # ------------------------------------------------------------------
    # LCD accessors
    # ------------------------------------------------------------------

    def get_unwarped_lcd(self) -> np.ndarray:
        """Return the perspective-corrected (unwarped) grayscale LCD image.

        The LCD display is detected as a quadrilateral in the coarse crop and
        transformed to a straight rectangle using a full perspective transform.

        Returns:
            Grayscale unwarped LCD image.

        Raises:
            RuntimeError: If the image is invalid or no LCD quad can be detected.
        """
        if not self.valid_image:
            raise RuntimeError("Cannot unwarp LCD: image was marked invalid during construction.")

        if self._unwarped_lcd is None:
            assert self._lcd_gray is not None
            self._unwarped_lcd, self._lcd_quad = detect_and_unwarp(self._lcd_gray)

        return self._unwarped_lcd

    def get_digit_rois(self) -> list[np.ndarray]:
        """Return the eight individual digit ROI images from the unwarped LCD.

        Calls :meth:`get_unwarped_lcd` internally; raises the same errors on
        failure.

        Returns:
            List of eight grayscale ROI images (indexed left-to-right,
            top-to-bottom, sized width x height from config["lcd"]["digits"]).

        Raises:
            RuntimeError: If the image is invalid, the LCD cannot be detected,
                or a digit ROI falls outside the unwarped image bounds.
        """
        unwarped = self.get_unwarped_lcd()
        return extract_digit_rois(unwarped, self.config["lcd"]["digits"])

    # ------------------------------------------------------------------
    # LED accessors
    # ------------------------------------------------------------------

    def get_led_dict(self) -> dict[str, bool]:
        """Return a dictionary mapping each LED name to its on/off state.

        The image is considered valid only when the Status LED is on and
        the Black reference LED is off.

        Returns:
            Dictionary of LED states, or an empty dict if the image is invalid
            or the validity check fails.
        """
        if not self.valid_image or self._leds is None:
            return {}

        led_states: dict[str, bool] = {}
        for led in list(self.config["leds"].keys())[2:]:
            color = get_led_color(self._leds, self.config["leds"][led]["origin"], self.config["leds"][led]["radius"])
            led_states[led] = sum(color) > 50

        valid = led_states.get("Status", False) and not led_states.get("Black", True)
        return led_states if valid else {}

    def get_led_json(self) -> str:
        """Return LED states and derived fill level as a JSON string.

        The fill level is derived from the highest active fill sensor (S1-S4).
        S1 = 100 %, S2 = 75 %, S3 = 50 %, S4 = 25 %.

        Returns:
            JSON-encoded string with keys Valid, K1, K2, K3, Error, and
            FillLevel, or an empty string if the image is invalid.
        """
        led_states = self.get_led_dict()

        if not led_states:
            return ""

        fill_level = 0
        if led_states.get("S1"):
            fill_level = 100
        elif led_states.get("S2"):
            fill_level = 75
        elif led_states.get("S3"):
            fill_level = 50
        elif led_states.get("S4"):
            fill_level = 25

        payload = {
            "Valid": "True",
            "K1": str(led_states["K1"]),
            "K2": str(led_states["K2"]),
            "K3": str(led_states["K3"]),
            "Error": str(led_states["Error"]),
            "FillLevel": str(fill_level),
        }
        return json.dumps(payload)

    def get_led_annotations(self) -> np.ndarray | None:
        """Return a copy of the LED region image with circles drawn on each LED.

        Returns:
            Annotated BGR image, or None if the image is invalid.
        """
        if not self.valid_image or self._leds is None:
            return None

        annotated = self._leds.copy()
        draw_rectangle(annotated, self.config["leds"]["rectangle_origin"], self.config["leds"]["rectangle_size"])
        for led in list(self.config["leds"].keys())[2:]:
            annotated = annotate_led(
                annotated,
                self.config["leds"][led]["origin"],
                self.config["leds"][led]["radius"],
                (200, 0, 200),
            )
        return annotated

    def get_ph_redux(self) -> tuple[float, int]:
        """Classify the six digit ROIs with the ONNX CNN and return (ph, redux).

        The onnxruntime InferenceSession is loaded from ``models/digit_cnn.onnx``
        (relative to the project root) on the first call and cached for subsequent
        calls.  No PyTorch dependency is required at runtime.

        ROI layout (indices into :meth:`get_digit_rois`):
            0 = PH integer,  1 = decimal dot (skipped),  2 = PH tenth,
            3 = PH hundredth,  4 = separator (skipped),  5 = Redux hundreds,
            6 = Redux tens,  7 = Redux ones.

        Returns:
            ``(ph, redux)`` — e.g. ``(7.16, 796)``.

        Raises:
            RuntimeError: If the image is invalid or LCD unwarping fails.
            FileNotFoundError: If the ONNX model file is missing.
        """
        global _digit_session
        import onnxruntime as ort  # deferred: not available in all environments

        if _digit_session is None:
            model_path = Path(__file__).parent.parent.parent / "models" / "digit_cnn.onnx"
            if not model_path.exists():
                raise FileNotFoundError(f"ONNX digit model not found: {model_path}")
            _digit_session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            logger.info("Loaded ONNX digit model from %s", model_path)

        rois = self.get_digit_rois()  # raises RuntimeError if invalid

        digits: list[int] = []
        for i in (0, 2, 3, 5, 6, 7):  # skip dot (1) and separator (4)
            roi = rois[i].astype(np.float32) / 255.0
            roi = (roi - 0.5) / 0.5  # same normalisation as training
            inp = roi[np.newaxis, np.newaxis]  # (1, 1, H, W)
            logits = _digit_session.run(["logits"], {"image": inp})[0]
            digits.append(int(logits.argmax()))

        # digits: [ph_int, ph_tenth, ph_hundredth, rx_hundreds, rx_tens, rx_ones]
        ph = round(digits[0] + digits[1] * 0.1 + digits[2] * 0.01, 2)
        redux = digits[3] * 100 + digits[4] * 10 + digits[5]
        return ph, redux
