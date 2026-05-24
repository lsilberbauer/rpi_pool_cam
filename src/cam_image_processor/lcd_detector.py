# Copyright (c) 2026 Lukas Silberbauer. All rights reserved.

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Digit grid defaults — can be overridden from config["lcd"]["digits"]
_DEFAULT_DIGIT_ORIGIN = (23, 3)
_DEFAULT_DIGIT_SIZE = (12, 18)
_DEFAULT_NUM_COLS = 4
_DEFAULT_NUM_ROWS = 2


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four corner points as top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: Array of shape (4, 2) containing the four corner coordinates.

    Returns:
        Array of shape (4, 2) with points in (TL, TR, BR, BL) order.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Perspective-warp a quadrilateral region to a straight rectangle.

    Args:
        image: Source image (grayscale or BGR).
        pts: Array of shape (4, 2) containing the four corner coordinates
            of the quadrilateral in arbitrary order.

    Returns:
        Perspective-corrected image cropped to the bounding rectangle.
    """
    rect = order_points(pts)
    tl, tr, br, bl = rect
    max_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype="float32",
    )
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, transform_matrix, (max_w, max_h))


def detect_and_unwarp(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect the LCD quadrilateral in a grayscale crop and return the perspective-corrected image.

    Uses Otsu binarisation followed by contour detection and polygon approximation
    at multiple epsilon values to reliably find the four LCD corners. Corner positions
    are then refined to sub-pixel accuracy via Canny edges + cornerSubPix.

    Args:
        image: Grayscale image already cropped to the approximate LCD region.

    Returns:
        Tuple of (unwarped_image, quad) where:
            - unwarped_image is the perspective-corrected grayscale LCD image.
            - quad is an (4, 2) float32 array of the detected corner positions
              in the input image (TL, TR, BR, BL order after ordering).

    Raises:
        RuntimeError: If no quadrilateral can be detected. Check image quality
            and config["lcd"]["rectangle_origin/size"].
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    min_area = image.shape[0] * image.shape[1] * 0.2
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    quad = None
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        for eps in [0.02, 0.03, 0.05, 0.08]:
            approx = cv2.approxPolyDP(contour, eps * perimeter, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype("float32")
                break
        if quad is not None:
            break

    if quad is None:
        raise RuntimeError(
            "LCD quadrilateral not detected. "
            "Check image quality and config['lcd']['rectangle_origin/size']."
        )

    # Refine corners to sub-pixel accuracy using the Canny edge image
    edges = cv2.Canny(blurred, 50, 150)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    quad = cv2.cornerSubPix(edges, quad, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)

    unwarped = four_point_transform(image, quad)
    ordered_quad = order_points(quad)
    logger.debug("LCD quad detected: %s", ordered_quad.tolist())
    return unwarped, ordered_quad


def extract_digit_rois(
    unwarped_image: np.ndarray,
    digit_config: dict[str, Any],
) -> list[np.ndarray]:
    """Slice all digit ROIs from an unwarped LCD image.

    The display has two rows of four digits each (eight digits total).
    Digit positions are read from digit_config, which must contain the
    keys ``origin``, ``size``, ``num_cols``, and ``num_rows``.

    Args:
        unwarped_image: Perspective-corrected LCD image as returned by
            :func:`detect_and_unwarp`.
        digit_config: Mapping with digit grid parameters:
            - ``origin``: [x, y] pixel offset of the first digit (top-left corner).
            - ``size``: [width, height] of each digit ROI in pixels.
            - ``num_cols``: Number of digit columns.
            - ``num_rows``: Number of digit rows.

    Returns:
        List of ROI images indexed left-to-right, top-to-bottom (length =
        num_rows * num_cols). Each ROI has the same dtype as the input image.
    """
    origin_x: int = digit_config["origin"][0]
    origin_y: int = digit_config["origin"][1]
    width: int = digit_config["size"][0]
    height: int = digit_config["size"][1]
    num_cols: int = digit_config["num_cols"]
    num_rows: int = digit_config["num_rows"]

    rois: list[np.ndarray] = []
    for row in range(num_rows):
        for col in range(num_cols):
            x = origin_x + col * width
            y = origin_y + row * height
            roi = unwarped_image[y : y + height, x : x + width]
            if roi.size == 0:
                raise RuntimeError(
                    f"Digit ROI at row={row}, col={col} is empty. "
                    "Check digit_config['origin'] and ['size'] against the unwarped image dimensions."
                )
            rois.append(roi)

    return rois
