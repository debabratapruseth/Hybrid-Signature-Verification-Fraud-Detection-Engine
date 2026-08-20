"""Critical-point detection from skeleton topology and contour curvature."""

from __future__ import annotations

import numpy as np


def detect_curvature_extrema(
    contour: np.ndarray,
    curvature: np.ndarray,
    minimum_absolute_curvature: float | None = None,
    minimum_index_separation: int = 8,
    maximum_points: int = 40,
) -> np.ndarray:
    """Select separated local maxima of absolute contour curvature."""

    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    values = np.abs(np.asarray(curvature, dtype=np.float64))
    if len(points) != len(values):
        raise ValueError("Contour and curvature lengths must match")
    if not len(points):
        return np.empty((0, 2), dtype=np.float64)

    threshold = (
        float(minimum_absolute_curvature)
        if minimum_absolute_curvature is not None
        else float(np.quantile(values, 0.85))
    )
    candidates = [
        index
        for index in range(len(values))
        if values[index] >= threshold
        and values[index] >= values[index - 1]
        and values[index] >= values[(index + 1) % len(values)]
    ]
    selected: list[int] = []
    for index in sorted(candidates, key=lambda item: values[item], reverse=True):
        circular_distances = [
            min(
                abs(index - previous),
                len(values) - abs(index - previous),
            )
            for previous in selected
        ]
        if not circular_distances or min(circular_distances) >= minimum_index_separation:
            selected.append(index)
        if len(selected) >= maximum_points:
            break
    return points[sorted(selected)]


def combine_critical_points(
    endpoint_yx: np.ndarray,
    junction_yx: np.ndarray,
    curvature_xy: np.ndarray,
) -> dict[str, object]:
    """Return named critical-point groups and total count."""

    endpoints_xy = np.asarray(endpoint_yx)[:, ::-1].reshape(-1, 2)
    junctions_xy = np.asarray(junction_yx)[:, ::-1].reshape(-1, 2)
    curvature_points = np.asarray(curvature_xy).reshape(-1, 2)
    return {
        "endpoints_xy": endpoints_xy,
        "junctions_xy": junctions_xy,
        "curvature_extrema_xy": curvature_points,
        "endpoint_count": int(len(endpoints_xy)),
        "junction_count": int(len(junctions_xy)),
        "curvature_extrema_count": int(len(curvature_points)),
        "total_critical_points": int(
            len(endpoints_xy) + len(junctions_xy) + len(curvature_points)
        ),
    }
