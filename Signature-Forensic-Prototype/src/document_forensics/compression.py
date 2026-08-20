"""Compression-artifact screening using ELA and block-boundary statistics."""

from __future__ import annotations

import cv2
import numpy as np

from .common import load_rgb_image, validate_bbox


def calculate_error_level_map(
    image: object,
    jpeg_quality: int = 90,
) -> dict[str, object]:
    """Recompress in memory and visualize absolute reconstruction error."""

    rgb = load_rgb_image(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
    )
    if not success:
        raise RuntimeError("JPEG recompression failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    difference = cv2.absdiff(bgr, decoded)
    grayscale_error = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    scale = 255.0 / max(float(np.percentile(grayscale_error, 99)), 1.0)
    visual = np.clip(grayscale_error.astype(float) * scale, 0, 255).astype(
        np.uint8
    )
    visual = cv2.applyColorMap(visual, cv2.COLORMAP_INFERNO)
    return {
        "jpeg_quality": int(jpeg_quality),
        "mean_error": round(float(grayscale_error.mean()), 6),
        "maximum_error": int(grayscale_error.max()),
        "error_map": grayscale_error,
        "visualization": cv2.cvtColor(visual, cv2.COLOR_BGR2RGB),
    }


def calculate_blockiness(image: object, block_size: int = 8) -> dict[str, float]:
    """Compare intensity jumps at JPEG block boundaries and other positions."""

    gray = cv2.cvtColor(load_rgb_image(image), cv2.COLOR_RGB2GRAY).astype(float)
    vertical_difference = np.abs(np.diff(gray, axis=1))
    horizontal_difference = np.abs(np.diff(gray, axis=0))
    vertical_indices = np.arange(1, gray.shape[1]) % block_size == 0
    horizontal_indices = np.arange(1, gray.shape[0]) % block_size == 0
    boundary_values = np.concatenate(
        [
            vertical_difference[:, vertical_indices].ravel(),
            horizontal_difference[horizontal_indices, :].ravel(),
        ]
    )
    nonboundary_values = np.concatenate(
        [
            vertical_difference[:, ~vertical_indices].ravel(),
            horizontal_difference[~horizontal_indices, :].ravel(),
        ]
    )
    boundary_mean = float(boundary_values.mean()) if boundary_values.size else 0.0
    nonboundary_mean = (
        float(nonboundary_values.mean()) if nonboundary_values.size else 0.0
    )
    return {
        "block_boundary_difference": round(boundary_mean, 6),
        "nonboundary_difference": round(nonboundary_mean, 6),
        "blockiness_ratio": round(
            boundary_mean / max(nonboundary_mean, 1e-6),
            6,
        ),
    }


def analyze_local_compression(
    page_image: object,
    bbox_xyxy: list[int] | tuple[int, int, int, int],
    jpeg_quality: int = 90,
) -> dict[str, object]:
    """Compare ELA intensity inside the signature box with the remaining page."""

    page = load_rgb_image(page_image)
    x1, y1, x2, y2 = validate_bbox(bbox_xyxy, page.shape)
    ela = calculate_error_level_map(page, jpeg_quality=jpeg_quality)
    error_map = np.asarray(ela["error_map"], dtype=float)
    inside_mask = np.zeros(error_map.shape, dtype=bool)
    inside_mask[y1:y2, x1:x2] = True
    inside_values = error_map[inside_mask]
    outside_values = error_map[~inside_mask]
    inside_mean = float(inside_values.mean())
    outside_mean = float(outside_values.mean())
    outside_std = float(outside_values.std())
    z_score = (inside_mean - outside_mean) / max(outside_std, 1e-6)
    return {
        "inside_signature_ela_mean": round(inside_mean, 6),
        "outside_signature_ela_mean": round(outside_mean, 6),
        "outside_signature_ela_std": round(outside_std, 6),
        "local_ela_z_score": round(float(z_score), 6),
        "page_blockiness": calculate_blockiness(page),
        "jpeg_quality": int(jpeg_quality),
        "visualization": ela["visualization"],
        "limitation": (
            "ELA is a screening indicator. PDF rendering, scanning, image "
            "resizing, and repeated compression can produce similar patterns."
        ),
    }
