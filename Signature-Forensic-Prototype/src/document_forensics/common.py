"""Shared image and bounding-box helpers for Branch 3."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_rgb_image(
    image: np.ndarray | Image.Image | str | Path,
) -> np.ndarray:
    """Return an RGB uint8 copy from an array, PIL image, or image path."""

    if isinstance(image, np.ndarray):
        array = image.copy()
        if array.ndim == 2:
            array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif array.ndim == 3 and array.shape[2] == 4:
            array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGBA2RGB)
        elif array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Image must be grayscale, RGB, or RGBA")
        return np.clip(array, 0, 255).astype(np.uint8)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB")).copy()

    path = Path(image).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def validate_bbox(
    bbox_xyxy: list[int] | tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Clip and validate an `(x1, y1, x2, y2)` pixel bounding box."""

    if len(bbox_xyxy) != 4:
        raise ValueError("Bounding box must contain four coordinates")
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [round(float(value)) for value in bbox_xyxy]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def crop_bbox(
    image: np.ndarray,
    bbox_xyxy: list[int] | tuple[int, int, int, int],
) -> np.ndarray:
    """Return a defensive crop from a validated bounding box."""

    x1, y1, x2, y2 = validate_bbox(bbox_xyxy, image.shape)
    return image[y1:y2, x1:x2].copy()
