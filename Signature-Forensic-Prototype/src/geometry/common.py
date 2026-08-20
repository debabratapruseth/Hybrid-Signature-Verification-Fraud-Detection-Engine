"""Shared image loading and normalization for Branch 2."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def load_rgb_image(
    image: np.ndarray | Image.Image | str | Path,
) -> np.ndarray:
    """Return a defensive RGB uint8 copy from an array, PIL image, or path."""

    if isinstance(image, np.ndarray):
        array = image.copy()
        if array.ndim == 2:
            array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif array.ndim == 3 and array.shape[2] == 4:
            array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGBA2RGB)
        elif array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Image array must be grayscale, RGB, or RGBA")
        return np.clip(array, 0, 255).astype(np.uint8)

    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB")).copy()

    path = Path(image).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Signature image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix}")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read signature image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def create_normalized_ink_mask(
    image: np.ndarray | Image.Image | str | Path,
    canvas_width: int = 512,
    canvas_height: int = 256,
    padding: int = 16,
) -> np.ndarray:
    """Create a centered boolean ink mask while preserving aspect ratio."""

    if canvas_width <= 2 * padding or canvas_height <= 2 * padding:
        raise ValueError("Canvas must be larger than twice the padding")

    rgb = load_rgb_image(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    ink = binary > 0
    if float(ink.mean()) > 0.50:
        ink = ~ink

    coordinates = np.argwhere(ink)
    if coordinates.size == 0:
        raise ValueError("No signature foreground was detected")
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0) + 1
    cropped = ink[y_min:y_max, x_min:x_max]

    usable_width = canvas_width - 2 * padding
    usable_height = canvas_height - 2 * padding
    scale = min(
        usable_width / cropped.shape[1],
        usable_height / cropped.shape[0],
    )
    target_width = max(1, round(cropped.shape[1] * scale))
    target_height = max(1, round(cropped.shape[0] * scale))
    resized = cv2.resize(
        cropped.astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    canvas = np.zeros((canvas_height, canvas_width), dtype=bool)
    x_start = (canvas_width - target_width) // 2
    y_start = (canvas_height - target_height) // 2
    canvas[
        y_start : y_start + target_height,
        x_start : x_start + target_width,
    ] = resized
    return canvas


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Render a boolean ink mask as black ink on a white RGB canvas."""

    boolean_mask = np.asarray(mask, dtype=bool)
    grayscale = np.where(boolean_mask, 0, 255).astype(np.uint8)
    return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)


def json_safe(value: object) -> object:
    """Recursively convert NumPy values into JSON-compatible Python values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
