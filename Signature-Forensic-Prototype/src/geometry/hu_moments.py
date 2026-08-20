"""Translation-, scale-, and rotation-aware Hu moment descriptors."""

from __future__ import annotations

import cv2
import numpy as np


def compute_hu_moments(ink_mask: np.ndarray) -> dict[str, list[float]]:
    """Calculate raw and signed-log Hu moments from a boolean mask."""

    binary = np.asarray(ink_mask, dtype=np.uint8)
    if not binary.any():
        raise ValueError("Cannot calculate Hu moments for an empty mask")
    raw = cv2.HuMoments(cv2.moments(binary)).flatten()
    signed_log = np.asarray(
        [
            -np.sign(value) * np.log10(abs(value) + 1e-30)
            for value in raw
        ],
        dtype=np.float64,
    )
    return {
        "raw": [float(value) for value in raw],
        "signed_log": [round(float(value), 8) for value in signed_log],
    }


def hu_moment_distance(
    first_signed_log: list[float] | np.ndarray,
    second_signed_log: list[float] | np.ndarray,
) -> float:
    """Return mean absolute distance between signed-log Hu descriptors."""

    first = np.asarray(first_signed_log, dtype=np.float64)
    second = np.asarray(second_signed_log, dtype=np.float64)
    if first.shape != (7,) or second.shape != (7,):
        raise ValueError("Each Hu descriptor must contain seven values")
    return round(float(np.mean(np.abs(first - second))), 8)
