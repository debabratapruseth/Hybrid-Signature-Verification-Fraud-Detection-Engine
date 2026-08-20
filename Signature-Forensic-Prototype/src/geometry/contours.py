"""Contour extraction and consistent contour resampling."""

from __future__ import annotations

import cv2
import numpy as np


def extract_contours(
    ink_mask: np.ndarray,
    minimum_area: float = 2.0,
) -> list[np.ndarray]:
    """Extract external contours sorted from largest to smallest."""

    binary = np.asarray(ink_mask, dtype=np.uint8) * 255
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    filtered = [
        contour.reshape(-1, 2).astype(np.float64)
        for contour in contours
        if cv2.contourArea(contour) >= minimum_area
        and len(contour) >= 3
    ]
    return sorted(
        filtered,
        key=lambda contour: abs(cv2.contourArea(contour.astype(np.float32))),
        reverse=True,
    )


def contour_measurements(
    contours: list[np.ndarray],
) -> dict[str, object]:
    """Summarize contour count, perimeter, and area."""

    areas = [
        float(abs(cv2.contourArea(contour.astype(np.float32))))
        for contour in contours
    ]
    perimeters = [
        float(
            cv2.arcLength(
                contour.astype(np.float32).reshape(-1, 1, 2),
                True,
            )
        )
        for contour in contours
    ]
    return {
        "contour_count": len(contours),
        "total_contour_area": round(sum(areas), 6),
        "total_contour_perimeter": round(sum(perimeters), 6),
        "largest_contour_area": round(max(areas, default=0.0), 6),
        "areas": [round(value, 6) for value in areas],
        "perimeters": [round(value, 6) for value in perimeters],
    }


def resample_closed_contour(
    contour: np.ndarray,
    point_count: int = 256,
) -> np.ndarray:
    """Resample a closed contour at uniform arc-length intervals."""

    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        raise ValueError("A contour needs at least three points")
    if point_count < 8:
        raise ValueError("point_count must be at least 8")

    closed = np.vstack([points, points[0]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 0:
        raise ValueError("Contour has zero length")

    sample_positions = np.linspace(
        0.0,
        total_length,
        point_count,
        endpoint=False,
    )
    x = np.interp(sample_positions, cumulative, closed[:, 0])
    y = np.interp(sample_positions, cumulative, closed[:, 1])
    return np.column_stack([x, y])
