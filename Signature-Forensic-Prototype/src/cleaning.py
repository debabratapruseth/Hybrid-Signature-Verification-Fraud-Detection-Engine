"""Conservative OpenCV cleaning for segmented signature crops.

The cleaning stage removes only well-supported interference. It preserves the
original YOLO crop and never edits the SAM result in place. Small dots,
disconnected strokes, short endpoints, and flourishes are intentionally treated
as meaningful unless they are extremely small.

Cleaning output is supporting analysis material, not proof of authenticity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np
import yaml
from skimage.morphology import skeletonize


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class CleaningQuality(TypedDict):
    """Interpretable comparisons before and after cleaning."""

    passed: bool
    quality_score: float
    foreground_removal_ratio: float
    original_components: int
    cleaned_components: int
    original_endpoints: int
    cleaned_endpoints: int
    endpoint_change_ratio: float
    fallback_used: bool
    fallback_reason: str | None
    warnings: list[str]


class CleaningResult(TypedDict):
    """Data contract returned for one segmented signature."""

    original_crop: np.ndarray
    sam_segmented_signature: np.ndarray
    binary_signature: np.ndarray
    removed_line_mask: np.ndarray
    detected_lines_image: np.ndarray
    cleaned_signature: np.ndarray
    connected_components: int
    quality: CleaningQuality


def load_cleaning_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Read the ``cleaning`` section from config.yaml."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a dictionary")
    if not isinstance(config.get("cleaning"), dict):
        raise ValueError("config.yaml must contain a 'cleaning' section")
    return config["cleaning"]


def clean_segmented_signature(
    segmentation_result: dict[str, Any],
    cleaning_config: dict[str, Any],
) -> CleaningResult:
    """Conservatively clean one accepted SAM segmentation.

    Args:
        segmentation_result: Output from ``segment_with_box``.
        cleaning_config: The ``cleaning`` section of config.yaml.

    Returns:
        Preserved inputs, detected-line evidence, cleaned binary signature,
        padded model input, and cleaning-quality results.
    """

    required_keys = {
        "original_crop",
        "segmented_signature",
        "mask",
    }
    missing_keys = required_keys - segmentation_result.keys()
    if missing_keys:
        raise ValueError(
            f"Segmentation result is missing keys: {sorted(missing_keys)}"
        )

    original_crop = segmentation_result["original_crop"].copy()
    sam_segmented = segmentation_result["segmented_signature"].copy()
    sam_mask = segmentation_result["mask"].astype(bool).copy()
    _validate_rgb_image(original_crop)
    _validate_rgb_image(sam_segmented)
    if sam_mask.shape != sam_segmented.shape[:2]:
        raise ValueError("SAM mask dimensions do not match segmented image")

    original_foreground = create_binary_ink_mask(
        sam_segmented,
        sam_mask,
        cleaning_config,
    )
    line_mask, detected_lines_image = detect_straight_lines(
        sam_segmented,
        original_foreground,
        cleaning_config,
    )

    without_lines = np.logical_and(
        original_foreground,
        np.logical_not(line_mask),
    )
    removed_line_pixels = np.logical_and(
        original_foreground,
        line_mask,
    )
    original_count = int(original_foreground.sum())
    provisional_removed_ratio = (
        int(removed_line_pixels.sum()) / original_count
        if original_count
        else 0.0
    )

    # Roll back line removal when it would destroy too much foreground.
    maximum_removal = float(
        cleaning_config["maximum_foreground_removal_ratio"]
    )
    if provisional_removed_ratio > maximum_removal:
        without_lines = original_foreground.copy()
        removed_line_pixels = np.zeros_like(original_foreground)

    component_filtered = remove_tiny_components(
        without_lines,
        minimum_area=int(cleaning_config["minimum_component_area"]),
        maximum_area_ratio=float(
            cleaning_config["maximum_component_area_ratio"]
        ),
    )
    closing_kernel_size = int(
        cleaning_config["closing_kernel_size"]
    )
    if closing_kernel_size > 1 and component_filtered.any():
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (closing_kernel_size, closing_kernel_size),
        )
        closed = cv2.morphologyEx(
            component_filtered.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )
        cleaned_foreground = closed > 0
    else:
        cleaned_foreground = component_filtered

    # Component filtering and morphology can also remove or fragment ink after
    # the earlier line-removal check. Evaluate the complete proposed change and
    # roll everything back when it is too destructive.
    fallback_reason = determine_fallback_reason(
        original_foreground,
        cleaned_foreground,
        cleaning_config,
    )
    if fallback_reason is not None:
        cleaned_foreground = original_foreground.copy()
        removed_line_pixels = np.zeros_like(original_foreground)

    binary_signature = foreground_to_binary_image(cleaned_foreground)
    cleaned_signature = place_on_fixed_canvas(
        cleaned_foreground,
        canvas_width=int(cleaning_config["canvas_width"]),
        canvas_height=int(cleaning_config["canvas_height"]),
        padding=int(cleaning_config["canvas_padding"]),
    )
    quality = assess_cleaning_quality(
        original_foreground,
        cleaned_foreground,
        cleaning_config,
        fallback_reason=fallback_reason,
    )

    return {
        "original_crop": original_crop,
        "sam_segmented_signature": sam_segmented,
        "binary_signature": binary_signature,
        "removed_line_mask": (
            removed_line_pixels.astype(np.uint8) * 255
        ),
        "detected_lines_image": detected_lines_image,
        "cleaned_signature": cleaned_signature,
        "connected_components": quality["cleaned_components"],
        "quality": quality,
    }


def clean_segmentation_results(
    segmentation_results: list[dict[str, Any]],
    cleaning_config: dict[str, Any],
) -> list[CleaningResult]:
    """Clean every accepted SAM segmentation."""

    if not segmentation_results:
        raise ValueError("At least one segmentation result is required")
    return [
        clean_segmented_signature(result, cleaning_config)
        for result in segmentation_results
    ]


def create_binary_ink_mask(
    segmented_signature: np.ndarray,
    sam_mask: np.ndarray,
    cleaning_config: dict[str, Any],
) -> np.ndarray:
    """Estimate dark ink and restrict it to SAM-selected pixels."""

    grayscale = cv2.cvtColor(
        segmented_signature,
        cv2.COLOR_RGB2GRAY,
    )
    block_size = int(
        cleaning_config["adaptive_threshold_block_size"]
    )
    if block_size < 3:
        raise ValueError("adaptive_threshold_block_size must be at least 3")
    if block_size % 2 == 0:
        block_size += 1

    thresholded = cv2.adaptiveThreshold(
        grayscale,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        int(cleaning_config["adaptive_threshold_constant"]),
    )
    return np.logical_and(
        thresholded > 0,
        sam_mask.astype(bool),
    )


def detect_straight_lines(
    segmented_signature: np.ndarray,
    foreground_mask: np.ndarray,
    cleaning_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Detect long, nearly horizontal or vertical line interference."""

    height, width = foreground_mask.shape
    line_mask = np.zeros((height, width), dtype=np.uint8)
    visualization = segmented_signature.copy()
    minimum_configured_length = int(
        cleaning_config["minimum_line_length"]
    )
    minimum_line_length = max(
        minimum_configured_length,
        int(min(width, height) * 0.35),
    )
    lines = cv2.HoughLinesP(
        foreground_mask.astype(np.uint8) * 255,
        rho=1,
        theta=np.pi / 180,
        threshold=int(cleaning_config["line_detection_threshold"]),
        minLineLength=minimum_line_length,
        maxLineGap=int(cleaning_config["maximum_line_gap"]),
    )
    if lines is None:
        return line_mask.astype(bool), visualization

    removal_width = max(
        1,
        int(cleaning_config["line_removal_width"]),
    )
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = abs(
            float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        )
        angle = min(angle, 180.0 - angle)
        near_horizontal = angle <= 5.0
        near_vertical = abs(angle - 90.0) <= 5.0
        if not (near_horizontal or near_vertical):
            continue

        cv2.line(
            line_mask,
            (x1, y1),
            (x2, y2),
            255,
            removal_width,
            cv2.LINE_8,
        )
        cv2.line(
            visualization,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            max(2, removal_width),
            cv2.LINE_AA,
        )

    return line_mask > 0, visualization


def remove_tiny_components(
    foreground_mask: np.ndarray,
    minimum_area: int,
    maximum_area_ratio: float,
) -> np.ndarray:
    """Remove only extremely small or implausibly dominant components."""

    if minimum_area < 1:
        raise ValueError("minimum_area must be at least one")
    if not 0.0 < maximum_area_ratio <= 1.0:
        raise ValueError("maximum_area_ratio must be between 0 and 1")

    component_count, labels, statistics, _ = (
        cv2.connectedComponentsWithStats(
            foreground_mask.astype(np.uint8),
            connectivity=8,
        )
    )
    output = np.zeros_like(foreground_mask, dtype=bool)
    image_area = foreground_mask.size

    for component_id in range(1, component_count):
        area = int(
            statistics[component_id, cv2.CC_STAT_AREA]
        )
        if area < minimum_area:
            continue
        if area / image_area > maximum_area_ratio:
            continue
        output[labels == component_id] = True
    return output


def place_on_fixed_canvas(
    foreground_mask: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    padding: int,
) -> np.ndarray:
    """Tightly crop foreground, resize without stretching, and add padding."""

    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("Canvas dimensions must be positive")
    if padding < 0:
        raise ValueError("Canvas padding cannot be negative")
    available_width = canvas_width - 2 * padding
    available_height = canvas_height - 2 * padding
    if available_width <= 0 or available_height <= 0:
        raise ValueError("Canvas padding leaves no usable image area")

    canvas = np.full(
        (canvas_height, canvas_width),
        255,
        dtype=np.uint8,
    )
    coordinates = cv2.findNonZero(
        foreground_mask.astype(np.uint8)
    )
    if coordinates is None:
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    x, y, width, height = cv2.boundingRect(coordinates)
    tight_mask = foreground_mask[
        y : y + height,
        x : x + width,
    ]
    scale = min(
        available_width / width,
        available_height / height,
    )
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized_mask = cv2.resize(
        tight_mask.astype(np.uint8),
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    start_x = (canvas_width - resized_width) // 2
    start_y = (canvas_height - resized_height) // 2
    canvas_region = canvas[
        start_y : start_y + resized_height,
        start_x : start_x + resized_width,
    ]
    canvas_region[resized_mask > 0] = 0
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)


def assess_cleaning_quality(
    original_foreground: np.ndarray,
    cleaned_foreground: np.ndarray,
    cleaning_config: dict[str, Any],
    fallback_reason: str | None = None,
) -> CleaningQuality:
    """Compare foreground, components, and endpoints before and after cleaning."""

    original_pixels = int(original_foreground.sum())
    cleaned_pixels = int(cleaned_foreground.sum())
    removed_pixels = max(0, original_pixels - cleaned_pixels)
    removal_ratio = (
        removed_pixels / original_pixels if original_pixels else 1.0
    )
    original_components = count_components(original_foreground)
    cleaned_components = count_components(cleaned_foreground)
    original_endpoints = count_skeleton_endpoints(original_foreground)
    cleaned_endpoints = count_skeleton_endpoints(cleaned_foreground)
    endpoint_change_ratio = (
        abs(cleaned_endpoints - original_endpoints)
        / max(1, original_endpoints)
    )

    maximum_removal = float(
        cleaning_config["maximum_foreground_removal_ratio"]
    )
    maximum_endpoint_change = float(
        cleaning_config.get(
            "maximum_endpoint_change_ratio",
            0.50,
        )
    )
    checks = {
        "original_not_empty": original_pixels > 0,
        "cleaned_not_empty": cleaned_pixels > 0,
        "removal_acceptable": removal_ratio <= maximum_removal,
        "components_remain": cleaned_components > 0,
        "endpoints_preserved": (
            endpoint_change_ratio <= maximum_endpoint_change
        ),
        "not_reduced_to_line": not is_reduced_to_line(
            cleaned_foreground
        ),
    }
    warnings = []
    if not checks["original_not_empty"]:
        warnings.append("SAM segmentation contains no measurable ink")
    if not checks["cleaned_not_empty"]:
        warnings.append("Cleaning removed all foreground pixels")
    if not checks["removal_acceptable"]:
        warnings.append("Cleaning removed excessive foreground")
    if not checks["components_remain"]:
        warnings.append("No meaningful connected components remain")
    if not checks["endpoints_preserved"]:
        warnings.append("Skeleton endpoint count changed substantially")
    if not checks["not_reduced_to_line"]:
        warnings.append("Cleaned result resembles a single straight line")
    if fallback_reason is not None:
        warnings.append(
            "Conservative fallback used: "
            f"{fallback_reason}. The original SAM-derived ink was preserved."
        )

    return {
        # A fallback is safe for evidence preservation but still requires human
        # review because the detected interference remains in the image.
        "passed": not warnings and fallback_reason is None,
        "quality_score": round(
            sum(checks.values()) / len(checks),
            3,
        ),
        "foreground_removal_ratio": round(removal_ratio, 6),
        "original_components": original_components,
        "cleaned_components": cleaned_components,
        "original_endpoints": original_endpoints,
        "cleaned_endpoints": cleaned_endpoints,
        "endpoint_change_ratio": round(endpoint_change_ratio, 6),
        "fallback_used": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "warnings": warnings,
    }


def determine_fallback_reason(
    original_foreground: np.ndarray,
    proposed_foreground: np.ndarray,
    cleaning_config: dict[str, Any],
) -> str | None:
    """Explain why a proposed cleaning result must be rolled back."""

    original_pixels = int(original_foreground.sum())
    proposed_pixels = int(proposed_foreground.sum())
    removal_ratio = (
        max(0, original_pixels - proposed_pixels) / original_pixels
        if original_pixels
        else 1.0
    )
    maximum_removal = float(
        cleaning_config["maximum_foreground_removal_ratio"]
    )
    if removal_ratio > maximum_removal:
        return (
            f"proposed cleaning would remove {removal_ratio:.1%} "
            f"of foreground, above the {maximum_removal:.1%} limit"
        )

    original_components = count_components(original_foreground)
    proposed_components = count_components(proposed_foreground)
    fragmentation_limit = max(
        original_components + 3,
        original_components * 2,
    )
    if proposed_components > fragmentation_limit:
        return (
            f"connected components would increase from "
            f"{original_components} to {proposed_components}"
        )

    original_endpoints = count_skeleton_endpoints(original_foreground)
    proposed_endpoints = count_skeleton_endpoints(proposed_foreground)
    endpoint_change_ratio = (
        abs(proposed_endpoints - original_endpoints)
        / max(1, original_endpoints)
    )
    maximum_endpoint_change = float(
        cleaning_config.get(
            "maximum_endpoint_change_ratio",
            0.50,
        )
    )
    if endpoint_change_ratio > maximum_endpoint_change:
        return (
            f"skeleton endpoints would change by "
            f"{endpoint_change_ratio:.1%}"
        )

    return None


def count_components(foreground_mask: np.ndarray) -> int:
    """Count non-background connected components."""

    component_count, _ = cv2.connectedComponents(
        foreground_mask.astype(np.uint8),
        connectivity=8,
    )
    return max(0, int(component_count) - 1)


def count_skeleton_endpoints(foreground_mask: np.ndarray) -> int:
    """Skeletonize foreground and count pixels with one neighbour."""

    if not foreground_mask.any():
        return 0
    skeleton = skeletonize(foreground_mask.astype(bool))
    neighbour_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbour_kernel[1, 1] = 0
    neighbour_count = cv2.filter2D(
        skeleton.astype(np.uint8),
        cv2.CV_16S,
        neighbour_kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    return int(
        np.logical_and(
            skeleton,
            neighbour_count == 1,
        ).sum()
    )


def is_reduced_to_line(foreground_mask: np.ndarray) -> bool:
    """Flag foreground whose tight box is only a few pixels thick."""

    coordinates = cv2.findNonZero(
        foreground_mask.astype(np.uint8)
    )
    if coordinates is None:
        return True
    _, _, width, height = cv2.boundingRect(coordinates)
    return min(width, height) <= 2


def foreground_to_binary_image(
    foreground_mask: np.ndarray,
) -> np.ndarray:
    """Convert Boolean foreground to conventional black ink on white."""

    binary_image = np.full(
        foreground_mask.shape,
        255,
        dtype=np.uint8,
    )
    binary_image[foreground_mask] = 0
    return binary_image


def save_cleaning_outputs(
    cleaning_results: list[CleaningResult],
    output_directory: str | Path,
) -> list[dict[str, str]]:
    """Save line evidence, binary images, cleaned canvases, and quality JSON."""

    import json

    output_path = Path(output_directory).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    saved_results = []

    for index, result in enumerate(cleaning_results, start=1):
        suffix = f"_{index:02d}" if len(cleaning_results) > 1 else ""
        lines_path = output_path / f"07_detected_lines{suffix}.png"
        binary_path = output_path / f"07_binary_signature{suffix}.png"
        removed_path = output_path / f"07_removed_line_mask{suffix}.png"
        cleaned_path = output_path / f"08_cleaned_signature{suffix}.png"
        quality_path = output_path / f"cleaning_quality{suffix}.json"

        _save_rgb_image(lines_path, result["detected_lines_image"])
        if not cv2.imwrite(str(binary_path), result["binary_signature"]):
            raise OSError(f"Could not save binary image: {binary_path}")
        if not cv2.imwrite(str(removed_path), result["removed_line_mask"]):
            raise OSError(f"Could not save line mask: {removed_path}")
        _save_rgb_image(cleaned_path, result["cleaned_signature"])
        quality_path.write_text(
            json.dumps(result["quality"], indent=2),
            encoding="utf-8",
        )
        saved_results.append(
            {
                "detected_lines_path": str(lines_path),
                "binary_signature_path": str(binary_path),
                "removed_line_mask_path": str(removed_path),
                "cleaned_signature_path": str(cleaned_path),
                "quality_path": str(quality_path),
            }
        )
    return saved_results


def _save_rgb_image(path: Path, image: np.ndarray) -> None:
    """Save an RGB image through OpenCV."""

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr_image):
        raise OSError(f"Could not save image: {path}")


def _validate_rgb_image(image: np.ndarray) -> None:
    """Reject empty or incorrectly shaped RGB input."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.size == 0:
        raise ValueError("image is empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected RGB shape (height, width, 3), received {image.shape}"
        )
