"""Local high-frequency residual comparison around the signature region."""

from __future__ import annotations

import cv2
import numpy as np

from .common import load_rgb_image, validate_bbox


def calculate_noise_residual(
    image: object,
    blur_sigma: float = 1.2,
) -> dict[str, object]:
    """Subtract a Gaussian estimate to expose local high-frequency residuals."""

    gray = cv2.cvtColor(load_rgb_image(image), cv2.COLOR_RGB2GRAY).astype(
        np.float32
    )
    smooth = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=float(blur_sigma),
        sigmaY=float(blur_sigma),
    )
    residual = gray - smooth
    magnitude = np.abs(residual)
    display_scale = 255.0 / max(float(np.percentile(magnitude, 99)), 1.0)
    visual = np.clip(magnitude * display_scale, 0, 255).astype(np.uint8)
    visual = cv2.applyColorMap(visual, cv2.COLORMAP_VIRIDIS)
    return {
        "residual": residual,
        "mean_absolute_residual": round(float(magnitude.mean()), 6),
        "residual_standard_deviation": round(float(residual.std()), 6),
        "visualization": cv2.cvtColor(visual, cv2.COLOR_BGR2RGB),
    }


def compare_signature_noise(
    page_image: object,
    bbox_xyxy: list[int] | tuple[int, int, int, int],
    *,
    surrounding_margin: int = 40,
    blur_sigma: float = 1.2,
    flat_gradient_threshold: float = 12.0,
) -> dict[str, object]:
    """Compare residuals in locally flat pixels inside and around the box.

    Strong ink and printed edges are excluded because edge contrast is not
    camera/scanner noise.
    """

    page = load_rgb_image(page_image)
    x1, y1, x2, y2 = validate_bbox(bbox_xyxy, page.shape)
    noise = calculate_noise_residual(page, blur_sigma=blur_sigma)
    residual = np.asarray(noise["residual"], dtype=float)
    gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    flat_mask = gradient <= float(flat_gradient_threshold)
    height, width = residual.shape
    outer_x1 = max(0, x1 - surrounding_margin)
    outer_y1 = max(0, y1 - surrounding_margin)
    outer_x2 = min(width, x2 + surrounding_margin)
    outer_y2 = min(height, y2 + surrounding_margin)
    inside_mask = np.zeros(residual.shape, dtype=bool)
    inside_mask[y1:y2, x1:x2] = True
    ring_mask = np.zeros(residual.shape, dtype=bool)
    ring_mask[outer_y1:outer_y2, outer_x1:outer_x2] = True
    ring_mask &= ~inside_mask
    inside = residual[inside_mask & flat_mask]
    surrounding = residual[ring_mask & flat_mask]
    if inside.size < 20 or surrounding.size < 20:
        raise ValueError(
            "Too few locally flat pixels for a reliable noise comparison"
        )
    inside_std = float(inside.std())
    surrounding_std = float(surrounding.std())
    std_ratio = inside_std / max(surrounding_std, 1e-6)
    mean_difference = abs(float(inside.mean()) - float(surrounding.mean()))
    return {
        "inside_residual_std": round(inside_std, 6),
        "surrounding_residual_std": round(surrounding_std, 6),
        "residual_std_ratio": round(std_ratio, 6),
        "absolute_residual_mean_difference": round(mean_difference, 6),
        "inside_flat_pixel_count": int(inside.size),
        "surrounding_flat_pixel_count": int(surrounding.size),
        "flat_gradient_threshold": float(flat_gradient_threshold),
        "surrounding_bbox_xyxy": [
            outer_x1,
            outer_y1,
            outer_x2,
            outer_y2,
        ],
        "visualization": noise["visualization"],
        "limitation": (
            "Noise inconsistency is not specific to editing. Ink density, paper "
            "texture, shadows, scanning, and compression can change residuals."
        ),
    }
