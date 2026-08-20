"""Skeleton extraction and stable endpoint/junction detection."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize


def extract_skeleton(ink_mask: np.ndarray) -> np.ndarray:
    """Reduce signature strokes to a one-pixel-wide boolean skeleton."""

    mask = np.asarray(ink_mask, dtype=bool)
    if not mask.any():
        raise ValueError("Cannot skeletonize an empty mask")
    return skeletonize(mask)


def count_skeleton_neighbours(skeleton: np.ndarray) -> np.ndarray:
    """Count eight-connected skeleton neighbours for every pixel."""

    binary = np.asarray(skeleton, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    count_with_centre = cv2.filter2D(
        binary,
        cv2.CV_16U,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    return count_with_centre - binary.astype(np.uint16)


def find_skeleton_points(
    skeleton: np.ndarray,
) -> dict[str, object]:
    """Return endpoints and clustered junction centres.

    Adjacent branch pixels are clustered into one junction. This avoids the
    inflated branch-point count seen in Branch 1.
    """

    boolean_skeleton = np.asarray(skeleton, dtype=bool)
    neighbours = count_skeleton_neighbours(boolean_skeleton)
    endpoint_mask = boolean_skeleton & (neighbours == 1)
    branch_pixel_mask = boolean_skeleton & (neighbours >= 3)

    endpoint_yx = np.argwhere(endpoint_mask)
    junction_yx = _component_centres(branch_pixel_mask)
    return {
        "endpoint_yx": endpoint_yx,
        "junction_yx": junction_yx,
        "endpoint_count": int(len(endpoint_yx)),
        "junction_count": int(len(junction_yx)),
        "raw_branch_pixel_count": int(branch_pixel_mask.sum()),
        "endpoint_mask": endpoint_mask,
        "branch_pixel_mask": branch_pixel_mask,
    }


def skeleton_length(skeleton: np.ndarray) -> dict[str, float]:
    """Estimate eight-connected path length in pixels."""

    binary = np.asarray(skeleton, dtype=bool)
    horizontal = int(np.sum(binary[:, 1:] & binary[:, :-1]))
    vertical = int(np.sum(binary[1:, :] & binary[:-1, :]))
    diagonal_one = int(np.sum(binary[1:, 1:] & binary[:-1, :-1]))
    diagonal_two = int(np.sum(binary[1:, :-1] & binary[:-1, 1:]))
    length = horizontal + vertical + np.sqrt(2.0) * (
        diagonal_one + diagonal_two
    )
    return {
        "skeleton_pixel_count": int(binary.sum()),
        "eight_connected_length": round(float(length), 6),
    }


def _component_centres(mask: np.ndarray) -> np.ndarray:
    """Return rounded `(y, x)` centres for connected true regions."""

    component_count, labels, _, centroids = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8),
        connectivity=8,
    )
    centres = []
    for component_index in range(1, component_count):
        x, y = centroids[component_index]
        centres.append([round(y), round(x)])
    return np.asarray(centres, dtype=np.int32).reshape(-1, 2)
