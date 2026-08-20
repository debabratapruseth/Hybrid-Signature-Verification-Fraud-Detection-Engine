"""Load branch images and create a compact fusion dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_rgb(path: str | Path) -> np.ndarray:
    """Load one evidence image as RGB."""

    file_path = Path(path).expanduser().resolve()
    bgr = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read evidence image: {file_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def create_fusion_dashboard(
    images: dict[str, list[str]],
    fusion_result: dict[str, Any],
    panel_width: int = 420,
    panel_height: int = 260,
) -> np.ndarray:
    """Create a labelled input/branch-output dashboard."""

    selected = []
    for group in (
        "inputs",
        "branch1_outputs",
        "branch2_outputs",
        "branch3_outputs",
    ):
        if images.get(group):
            selected.append((group, images[group][0]))
    tiles = [
        _make_tile(
            load_rgb(path),
            group.replace("_", " ").title(),
            panel_width,
            panel_height,
        )
        for group, path in selected
    ]
    while len(tiles) < 4:
        tiles.append(
            np.full((panel_height + 45, panel_width, 3), 245, dtype=np.uint8)
        )
    first_row = np.hstack(tiles[:2])
    second_row = np.hstack(tiles[2:4])
    image_grid = np.vstack([first_row, second_row])

    summary_height = 180
    summary = np.full(
        (summary_height, image_grid.shape[1], 3),
        255,
        dtype=np.uint8,
    )
    lines = [
        "Three-Branch Fusion",
        f"Conclusion: {fusion_result['conclusion']}",
        f"Fused review risk: {fusion_result['fused_review_risk']:.3f}",
        "Branch risks: "
        + ", ".join(
            f"{name}={value:.3f}"
            for name, value in fusion_result["branch_risks"].items()
        ),
        fusion_result["cross_branch_agreement"]["summary"],
        "Review-priority synthesis only; not an authenticity determination.",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            summary,
            line[:115],
            (18, 28 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62 if index else 0.78,
            (25, 25, 25),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
    return np.vstack([image_grid, summary])


def _make_tile(
    rgb: np.ndarray,
    label: str,
    width: int,
    height: int,
) -> np.ndarray:
    """Fit one image on a white labelled tile."""

    scale = min(width / rgb.shape[1], height / rgb.shape[0])
    resized = cv2.resize(
        rgb,
        (
            max(1, round(rgb.shape[1] * scale)),
            max(1, round(rgb.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    tile = np.full((height + 45, width, 3), 255, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.putText(
        tile,
        label,
        (12, height + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return tile
