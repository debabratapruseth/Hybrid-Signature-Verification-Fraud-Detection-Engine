"""Fourier contour descriptors normalized for translation, scale, and start."""

from __future__ import annotations

import numpy as np


def compute_fourier_descriptor(
    contour: np.ndarray,
    coefficient_count: int = 32,
) -> np.ndarray:
    """Describe a resampled contour using normalized Fourier magnitudes."""

    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if coefficient_count < 4:
        raise ValueError("coefficient_count must be at least 4")
    if len(points) < coefficient_count:
        raise ValueError("Contour has fewer points than requested coefficients")

    complex_contour = points[:, 0] + 1j * points[:, 1]
    complex_contour -= complex_contour.mean()
    coefficients = np.fft.fft(complex_contour)
    non_dc = coefficients[1 : coefficient_count + 1]
    scale = abs(non_dc[0])
    if scale <= 1e-12:
        scale = float(np.linalg.norm(non_dc))
    if scale <= 1e-12:
        raise ValueError("Contour has no usable Fourier variation")
    return np.abs(non_dc / scale).astype(np.float64)


def fourier_distance(
    first_descriptor: np.ndarray,
    second_descriptor: np.ndarray,
) -> float:
    """Return normalized Euclidean distance between Fourier descriptors."""

    first = np.asarray(first_descriptor, dtype=np.float64)
    second = np.asarray(second_descriptor, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("Fourier descriptor shapes must match")
    return round(
        float(
            np.linalg.norm(first - second)
            / np.sqrt(max(first.size, 1))
        ),
        8,
    )
