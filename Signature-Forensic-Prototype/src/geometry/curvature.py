"""Discrete curvature measurements for resampled signature contours."""

from __future__ import annotations

import numpy as np


def compute_contour_curvature(
    contour: np.ndarray,
    neighbourhood: int = 4,
) -> np.ndarray:
    """Return signed turning angle per local arc length."""

    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2 * neighbourhood + 1:
        raise ValueError("Contour is too short for the requested neighbourhood")
    previous = np.roll(points, neighbourhood, axis=0)
    following = np.roll(points, -neighbourhood, axis=0)
    incoming = points - previous
    outgoing = following - points
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    dot = np.sum(incoming * outgoing, axis=1)
    angle = np.arctan2(cross, dot)
    local_length = (
        np.linalg.norm(incoming, axis=1)
        + np.linalg.norm(outgoing, axis=1)
    ) / 2.0
    return angle / np.maximum(local_length, 1e-6)


def summarize_curvature(curvature: np.ndarray) -> dict[str, float]:
    """Return robust absolute-curvature statistics."""

    values = np.asarray(curvature, dtype=np.float64)
    absolute = np.abs(values)
    return {
        "mean_absolute_curvature": round(float(absolute.mean()), 8),
        "median_absolute_curvature": round(float(np.median(absolute)), 8),
        "maximum_absolute_curvature": round(float(absolute.max()), 8),
        "curvature_standard_deviation": round(float(values.std()), 8),
    }
