# Copyright (c) 2026 Lukas Silberbauer. All rights reserved.

"""Tests for CamImageProcessor LED detection against ground-truth data."""

import cv2
import pytest

from src.cam_image_processor import CamImageProcessor

_LED_NAMES = ["S1", "S2", "S3", "S4", "K1", "K2", "K3", "Status"]


def test_led_states_match_ground_truth(image_yaml_pair, config):
    """LED states returned by CamImageProcessor must match the ground-truth YAML.

    For images whose YAML is empty (e.g. black/white calibration frames),
    CamImageProcessor is expected to return an empty dict (invalid image).
    """
    yaml_path, jpg_path = image_yaml_pair
    ground_truth = yaml_path  # already loaded by fixture — see conftest.py

    image = cv2.imread(str(jpg_path))
    assert image is not None, f"Could not read image: {jpg_path}"

    cip = CamImageProcessor(config, image)
    led_states = cip.get_led_dict()

    if ground_truth is None:
        # Calibration image (empty YAML) — processor must detect it as invalid
        assert led_states == {}, f"Expected empty dict for calibration image {jpg_path}, got {led_states}"
        return

    for led in _LED_NAMES:
        assert led in led_states, f"LED '{led}' missing from detected states for {jpg_path}"
        assert led_states[led] == ground_truth[led], (
            f"LED '{led}' mismatch for {jpg_path}: "
            f"detected={led_states[led]}, expected={ground_truth[led]}"
        )

