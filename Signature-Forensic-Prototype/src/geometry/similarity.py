"""Transparent distances and combined shape-similarity scoring."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


DEFAULT_METRIC_WEIGHTS = {
    "hu": 0.15,
    "fourier": 0.20,
    "shape_context": 0.25,
    "contour": 0.15,
    "graph": 0.15,
    "critical_points": 0.10,
}


def contour_shape_distance(
    first_contour: np.ndarray,
    second_contour: np.ndarray,
) -> float:
    """Return OpenCV I1 contour distance; zero means identical contours."""

    first = np.asarray(first_contour, dtype=np.float32).reshape(-1, 1, 2)
    second = np.asarray(second_contour, dtype=np.float32).reshape(-1, 1, 2)
    return round(
        float(cv2.matchShapes(first, second, cv2.CONTOURS_MATCH_I1, 0.0)),
        8,
    )


def critical_point_distance(
    first: dict[str, Any],
    second: dict[str, Any],
) -> float:
    """Compare endpoint, junction, and curvature-extrema counts."""

    fields = (
        "endpoint_count",
        "junction_count",
        "curvature_extrema_count",
    )
    differences = []
    for field in fields:
        first_value = float(first[field])
        second_value = float(second[field])
        differences.append(
            abs(first_value - second_value)
            / max(first_value, second_value, 1.0)
        )
    return round(float(np.mean(differences)), 8)


def distance_to_similarity(distance: float) -> float:
    """Convert a non-negative distance to a bounded, monotonic similarity."""

    return float(np.exp(-max(float(distance), 0.0)))


def combine_metric_distances(
    distances: dict[str, float],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Combine named distances while exposing every transform and weight."""

    selected_weights = dict(weights or DEFAULT_METRIC_WEIGHTS)
    missing = set(selected_weights) - set(distances)
    if missing:
        raise ValueError(f"Missing geometry distances: {sorted(missing)}")
    weight_sum = sum(float(value) for value in selected_weights.values())
    if weight_sum <= 0:
        raise ValueError("Geometry metric weights must sum to a positive value")
    normalized_weights = {
        name: float(value) / weight_sum
        for name, value in selected_weights.items()
    }
    similarities = {
        name: distance_to_similarity(distances[name])
        for name in normalized_weights
    }
    combined = sum(
        normalized_weights[name] * similarities[name]
        for name in normalized_weights
    )
    return {
        "distances": {
            name: round(float(value), 8)
            for name, value in distances.items()
        },
        "metric_similarities": {
            name: round(float(value), 8)
            for name, value in similarities.items()
        },
        "normalized_weights": {
            name: round(float(value), 8)
            for name, value in normalized_weights.items()
        },
        "combined_shape_similarity": round(float(combined), 8),
    }


def compare_to_reference_variation(
    questioned_scores: list[float],
    reference_pair_scores: list[float],
) -> dict[str, Any]:
    """Contextualize questioned scores against genuine-reference variation."""

    if not questioned_scores:
        raise ValueError("At least one questioned-reference score is required")
    questioned = np.asarray(questioned_scores, dtype=np.float64)
    references = np.asarray(reference_pair_scores, dtype=np.float64)
    result: dict[str, Any] = {
        "questioned_mean": round(float(questioned.mean()), 8),
        "questioned_median": round(float(np.median(questioned)), 8),
        "questioned_minimum": round(float(questioned.min()), 8),
        "questioned_maximum": round(float(questioned.max()), 8),
        "reference_pair_count": int(len(references)),
    }
    if len(references) < 3:
        result.update(
            {
                "reference_similarity_median": None,
                "reference_similarity_10th_percentile": None,
                "relative_status": "insufficient_reference_pairs",
            }
        )
        return result

    lower_reference_bound = float(np.percentile(references, 10))
    questioned_median = float(np.median(questioned))
    result.update(
        {
            "reference_similarity_median": round(
                float(np.median(references)),
                8,
            ),
            "reference_similarity_10th_percentile": round(
                lower_reference_bound,
                8,
            ),
            "relative_status": (
                "within_observed_reference_variation"
                if questioned_median >= lower_reference_bound
                else "below_observed_reference_variation"
            ),
        }
    )
    return result
