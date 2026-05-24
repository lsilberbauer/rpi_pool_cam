# Copyright (c) 2026 Lukas Silberbauer. All rights reserved.

"""Tests for the LCD perspective-unwarp and digit ROI extraction pipeline."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from src.cam_image_processor.lcd_detector import detect_and_unwarp, extract_digit_rois

_DATA_DIR = Path("data")
_CONFIG_PATH = Path("config.yaml")


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    with _CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


def _load_lcd_crop(jpg_path: Path, config: dict[str, Any]) -> np.ndarray:
    """Load a full frame, crop to the LCD region and convert to grayscale."""
    image = cv2.imread(str(jpg_path))
    assert image is not None, f"Could not read {jpg_path}"
    ox, oy = config["lcd"]["rectangle_origin"]
    w, h = config["lcd"]["rectangle_size"]
    crop = image[oy : oy + h, ox : ox + w]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def _real_jpgs() -> list[Path]:
    """Return only images that have non-empty ground truth YAMLs (i.e. real captures)."""
    result = []
    for jpg_path in sorted(_DATA_DIR.glob("*.jpg")):
        yaml_path = jpg_path.with_suffix(".yaml")
        if not yaml_path.is_file():
            continue
        with yaml_path.open() as fh:
            data = yaml.safe_load(fh)
        if data:  # skip empty calibration images
            result.append(jpg_path)
    return result


_REAL_JPGS = _real_jpgs()


class TestDetectAndUnwarp:
    """detect_and_unwarp() must succeed on all real data images."""

    @pytest.mark.parametrize("jpg_path", _REAL_JPGS, ids=[p.stem for p in _REAL_JPGS])
    def test_returns_non_empty_image(self, jpg_path: Path, config: dict[str, Any]) -> None:
        lcd_gray = _load_lcd_crop(jpg_path, config)
        unwarped, quad = detect_and_unwarp(lcd_gray)
        assert unwarped.size > 0, "Unwarped image must not be empty"
        assert unwarped.ndim == 2, "Unwarped image must be grayscale (2D)"

    @pytest.mark.parametrize("jpg_path", _REAL_JPGS, ids=[p.stem for p in _REAL_JPGS])
    def test_quad_has_four_points(self, jpg_path: Path, config: dict[str, Any]) -> None:
        lcd_gray = _load_lcd_crop(jpg_path, config)
        _, quad = detect_and_unwarp(lcd_gray)
        assert quad.shape == (4, 2), f"Quad must have shape (4, 2), got {quad.shape}"

    def test_raises_on_blank_image(self) -> None:
        blank = np.zeros((85, 140), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="LCD quadrilateral not detected"):
            detect_and_unwarp(blank)


class TestExtractDigitRois:
    """extract_digit_rois() must return correctly-sized ROIs for all real images."""

    @pytest.mark.parametrize("jpg_path", _REAL_JPGS, ids=[p.stem for p in _REAL_JPGS])
    def test_returns_eight_rois(self, jpg_path: Path, config: dict[str, Any]) -> None:
        lcd_gray = _load_lcd_crop(jpg_path, config)
        unwarped, _ = detect_and_unwarp(lcd_gray)
        rois = extract_digit_rois(unwarped, config["lcd"]["digits"])
        expected_count = config["lcd"]["digits"]["num_rows"] * config["lcd"]["digits"]["num_cols"]
        assert len(rois) == expected_count, f"Expected {expected_count} ROIs, got {len(rois)}"

    @pytest.mark.parametrize("jpg_path", _REAL_JPGS, ids=[p.stem for p in _REAL_JPGS])
    def test_roi_shape_matches_config(self, jpg_path: Path, config: dict[str, Any]) -> None:
        lcd_gray = _load_lcd_crop(jpg_path, config)
        unwarped, _ = detect_and_unwarp(lcd_gray)
        rois = extract_digit_rois(unwarped, config["lcd"]["digits"])
        expected_w, expected_h = config["lcd"]["digits"]["size"]
        for i, roi in enumerate(rois):
            assert roi.shape == (expected_h, expected_w), (
                f"ROI {i} has shape {roi.shape}, expected ({expected_h}, {expected_w})"
            )
