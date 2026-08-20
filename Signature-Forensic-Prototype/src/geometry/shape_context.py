"""Lightweight shape-context descriptors for sampled skeleton points."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def sample_shape_points(
    skeleton: np.ndarray,
    point_count: int = 100,
) -> np.ndarray:
    """Select approximately uniform points along a skeleton."""

    coordinates_yx = np.argwhere(np.asarray(skeleton, dtype=bool))
    if len(coordinates_yx) < 5:
        raise ValueError("Too few skeleton pixels for shape context")
    coordinates_xy = coordinates_yx[:, ::-1].astype(np.float64)
    if len(coordinates_xy) <= point_count:
        return coordinates_xy
    indices = np.linspace(
        0,
        len(coordinates_xy) - 1,
        point_count,
    ).astype(int)
    return coordinates_xy[indices]


def compute_shape_context(
    points_xy: np.ndarray,
    radial_bins: int = 5,
    angular_bins: int = 12,
) -> np.ndarray:
    """Calculate a log-polar neighbour histogram for every point."""

    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(points) < 5:
        raise ValueError("Shape context needs at least five points")
    differences = points[None, :, :] - points[:, None, :]
    distances = np.linalg.norm(differences, axis=2)
    nonzero = distances[distances > 0]
    mean_distance = float(nonzero.mean())
    normalized_distance = distances / max(mean_distance, 1e-9)
    angles = np.mod(
        np.arctan2(differences[:, :, 1], differences[:, :, 0]),
        2 * np.pi,
    )
    radial_edges = np.logspace(
        np.log10(0.125),
        np.log10(2.0),
        radial_bins + 1,
    )
    angular_edges = np.linspace(
        0.0,
        2 * np.pi,
        angular_bins + 1,
    )
    descriptors = np.zeros(
        (len(points), radial_bins * angular_bins),
        dtype=np.float64,
    )
    for index in range(len(points)):
        valid = np.arange(len(points)) != index
        histogram, _, _ = np.histogram2d(
            normalized_distance[index, valid],
            angles[index, valid],
            bins=(radial_edges, angular_edges),
        )
        flattened = histogram.flatten()
        descriptors[index] = flattened / max(flattened.sum(), 1.0)
    return descriptors


def shape_context_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Match point descriptors and return mean chi-square assignment cost."""

    first_descriptors = np.asarray(first, dtype=np.float64)
    second_descriptors = np.asarray(second, dtype=np.float64)
    numerator = (
        first_descriptors[:, None, :]
        - second_descriptors[None, :, :]
    ) ** 2
    denominator = (
        first_descriptors[:, None, :]
        + second_descriptors[None, :, :]
        + 1e-12
    )
    costs = 0.5 * np.sum(numerator / denominator, axis=2)
    row_indices, column_indices = linear_sum_assignment(costs)
    return round(
        float(costs[row_indices, column_indices].mean()),
        8,
    )
