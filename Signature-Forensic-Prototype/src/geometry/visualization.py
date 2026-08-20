"""Visual explanations for Branch 2 matched and mismatched geometry."""

from __future__ import annotations

import cv2
import numpy as np


def create_geometry_overlay(
    questioned_ink: np.ndarray,
    reference_ink: np.ndarray,
    tolerance_pixels: float = 3.0,
) -> dict[str, np.ndarray]:
    """Render approximate matched, questioned-only, and reference-only ink."""

    questioned = np.asarray(questioned_ink, dtype=bool)
    reference = np.asarray(reference_ink, dtype=bool)
    if questioned.shape != reference.shape:
        raise ValueError("Questioned and reference masks must have equal shape")

    distance_to_reference = cv2.distanceTransform(
        (~reference).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    distance_to_questioned = cv2.distanceTransform(
        (~questioned).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    questioned_matched = questioned & (
        distance_to_reference <= tolerance_pixels
    )
    reference_matched = reference & (
        distance_to_questioned <= tolerance_pixels
    )
    questioned_only = questioned & ~questioned_matched
    reference_only = reference & ~reference_matched
    matched = questioned_matched | reference_matched

    overlay = np.full((*questioned.shape, 3), 255, dtype=np.uint8)
    overlay[matched] = [30, 150, 60]
    overlay[questioned_only] = [220, 40, 40]
    overlay[reference_only] = [40, 80, 220]

    mismatch_strength = np.maximum(
        np.where(questioned, distance_to_reference, 0.0),
        np.where(reference, distance_to_questioned, 0.0),
    )
    mismatch_strength = np.clip(
        mismatch_strength / max(tolerance_pixels * 4.0, 1.0),
        0.0,
        1.0,
    )
    heatmap = cv2.applyColorMap(
        np.round(mismatch_strength * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    white_background = ~(questioned | reference)
    heatmap[white_background] = [255, 255, 255]
    return {
        "overlay": overlay,
        "mismatch_heatmap": heatmap,
        "matched_mask": matched,
        "questioned_only_mask": questioned_only,
        "reference_only_mask": reference_only,
    }


def draw_skeleton_points(
    ink_mask: np.ndarray,
    skeleton: np.ndarray,
    endpoint_yx: np.ndarray,
    junction_yx: np.ndarray,
) -> np.ndarray:
    """Draw skeleton, endpoints, and clustered junctions on one canvas."""

    visualization = np.full((*ink_mask.shape, 3), 255, dtype=np.uint8)
    visualization[np.asarray(ink_mask, dtype=bool)] = [190, 190, 190]
    visualization[np.asarray(skeleton, dtype=bool)] = [20, 20, 20]
    for y, x in np.asarray(endpoint_yx).reshape(-1, 2):
        cv2.circle(
            visualization,
            (int(x), int(y)),
            4,
            (220, 40, 40),
            -1,
        )
    for y, x in np.asarray(junction_yx).reshape(-1, 2):
        cv2.circle(
            visualization,
            (int(x), int(y)),
            5,
            (40, 80, 220),
            1,
        )
    return visualization


def draw_contour_critical_points(
    ink_mask: np.ndarray,
    contour_xy: np.ndarray,
    curvature_points_xy: np.ndarray,
) -> np.ndarray:
    """Draw the main contour and its selected high-curvature points."""

    visualization = np.full((*ink_mask.shape, 3), 255, dtype=np.uint8)
    visualization[np.asarray(ink_mask, dtype=bool)] = [220, 220, 220]
    contour = np.round(contour_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        visualization,
        [contour],
        True,
        (30, 130, 30),
        1,
    )
    for x, y in np.asarray(curvature_points_xy).reshape(-1, 2):
        cv2.circle(
            visualization,
            (round(float(x)), round(float(y))),
            3,
            (180, 40, 180),
            -1,
        )
    return visualization
