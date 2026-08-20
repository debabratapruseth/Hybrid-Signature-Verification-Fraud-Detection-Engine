"""Branch 3 evidence overlays and compact review panels."""

from __future__ import annotations

import cv2
import numpy as np

from .common import load_rgb_image, validate_bbox


def draw_analysis_regions(
    page_image: object,
    source_bbox_xyxy: list[int] | tuple[int, int, int, int],
    surrounding_bbox_xyxy: list[int] | tuple[int, int, int, int],
) -> np.ndarray:
    """Draw the signature box and surrounding noise-comparison region."""

    page = load_rgb_image(page_image)
    x1, y1, x2, y2 = validate_bbox(source_bbox_xyxy, page.shape)
    sx1, sy1, sx2, sy2 = validate_bbox(surrounding_bbox_xyxy, page.shape)
    visual = page.copy()
    cv2.rectangle(visual, (sx1, sy1), (sx2, sy2), (40, 100, 220), 2)
    cv2.rectangle(visual, (x1, y1), (x2, y2), (220, 50, 40), 2)
    return visual


def create_three_panel(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
) -> np.ndarray:
    """Resize three RGB evidence images to a common height and concatenate."""

    images = [load_rgb_image(image) for image in (first, second, third)]
    target_height = min(image.shape[0] for image in images)
    resized = []
    for image in images:
        scale = target_height / image.shape[0]
        width = max(1, round(image.shape[1] * scale))
        resized.append(
            cv2.resize(
                image,
                (width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        )
    return np.hstack(resized)
