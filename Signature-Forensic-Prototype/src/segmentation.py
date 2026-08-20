"""Box-prompted SAM segmentation for detected signature regions.

YOLO tells SAM where to look. SAM estimates which pixels inside that region
belong together. OpenCV provides an independent dark-ink estimate that helps
select and quality-check SAM's candidate masks.

SAM is a general segmentation model. Its mask is not proof that the selected
pixels are a signature, and poor detection boxes produce poor segmentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np
import yaml


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class SegmentationQuality(TypedDict):
    """Interpretable checks for a selected signature mask."""

    passed: bool
    quality_score: float
    sam_predicted_iou: float
    mask_area_ratio: float
    ink_agreement: float
    boundary_sides_touched: int
    rectangularity: float
    warnings: list[str]


class SegmentationResult(TypedDict):
    """Data contract returned for one accepted detection."""

    bbox_xyxy: list[int]
    original_crop: np.ndarray
    mask: np.ndarray
    full_page_mask: np.ndarray
    segmented_signature: np.ndarray
    mask_area_ratio: float
    touches_boundary: bool
    quality: SegmentationQuality


def load_segmentation_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Read the ``segmentation`` section from config.yaml."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a dictionary")
    if not isinstance(config.get("segmentation"), dict):
        raise ValueError("config.yaml must contain a 'segmentation' section")
    return config["segmentation"]


def load_sam_model(
    model_name: str,
    device: str,
) -> tuple[Any, Any]:
    """Download or load a Hugging Face SAM model and processor.

    Args:
        model_name: Hugging Face model identifier, for example
            ``facebook/sam-vit-base``.
        device: PyTorch device string: ``cuda`` or ``cpu``.

    Returns:
        ``(model, processor)`` ready for box-prompted inference.
    """

    try:
        from transformers import SamModel, SamProcessor
    except ImportError as error:
        raise ImportError(
            "Transformers is not installed. Install transformers and "
            "safetensors before loading SAM."
        ) from error

    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be 'cuda' or 'cpu'")

    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, processor


def segment_with_box(
    page_image: np.ndarray,
    bbox_xyxy: list[int],
    model: Any,
    processor: Any,
    segmentation_config: dict[str, Any],
    device: str,
) -> SegmentationResult:
    """Segment one accepted YOLO box using SAM candidate masks.

    The candidate with the strongest combination of SAM-predicted IoU,
    OpenCV-ink agreement, and plausible foreground area is selected.
    """

    _validate_rgb_image(page_image)
    x1, y1, x2, y2 = _validate_and_clamp_box(
        bbox_xyxy,
        page_width=page_image.shape[1],
        page_height=page_image.shape[0],
    )
    original_crop = page_image[y1:y2, x1:x2].copy()
    ink_mask = estimate_ink_mask(original_crop)

    try:
        import torch
    except ImportError as error:
        raise ImportError("PyTorch is required for SAM inference") from error

    model_inputs = processor(
        images=page_image,
        input_boxes=[[[x1, y1, x2, y2]]],
        return_tensors="pt",
    )
    tensor_inputs = {
        key: value.to(device)
        if hasattr(value, "to")
        else value
        for key, value in model_inputs.items()
    }

    with torch.no_grad():
        outputs = model(**tensor_inputs)

    processed_masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        model_inputs["original_sizes"].cpu(),
        model_inputs["reshaped_input_sizes"].cpu(),
    )
    candidate_masks = _extract_candidate_masks(
        processed_masks[0],
        expected_height=page_image.shape[0],
        expected_width=page_image.shape[1],
    )
    predicted_iou_scores = (
        outputs.iou_scores.detach().cpu().numpy().reshape(-1)
    )

    if not candidate_masks:
        raise RuntimeError("SAM returned no candidate masks")

    evaluated_candidates: list[dict[str, Any]] = []
    for candidate_index, full_page_mask in enumerate(candidate_masks):
        crop_mask = full_page_mask[y1:y2, x1:x2]
        sam_score = (
            float(predicted_iou_scores[candidate_index])
            if candidate_index < len(predicted_iou_scores)
            else 0.0
        )
        measurements = measure_mask(
            crop_mask,
            ink_mask,
        )
        selection_score = calculate_candidate_score(
            sam_predicted_iou=sam_score,
            measurements=measurements,
            segmentation_config=segmentation_config,
        )
        evaluated_candidates.append(
            {
                "full_page_mask": full_page_mask,
                "crop_mask": crop_mask,
                "sam_predicted_iou": sam_score,
                "measurements": measurements,
                "selection_score": selection_score,
            }
        )

    selected = max(
        evaluated_candidates,
        key=lambda candidate: candidate["selection_score"],
    )
    selected_mask = selected["crop_mask"].astype(bool)
    selected_full_page_mask = selected["full_page_mask"].astype(bool)
    quality = assess_segmentation_quality(
        mask=selected_mask,
        ink_mask=ink_mask,
        sam_predicted_iou=selected["sam_predicted_iou"],
        segmentation_config=segmentation_config,
    )

    # Keep selected pixels and replace the remaining background with white.
    segmented_signature = np.full_like(original_crop, 255)
    segmented_signature[selected_mask] = original_crop[selected_mask]

    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "original_crop": original_crop,
        "mask": selected_mask,
        "full_page_mask": selected_full_page_mask,
        "segmented_signature": segmented_signature,
        "mask_area_ratio": quality["mask_area_ratio"],
        "touches_boundary": quality["boundary_sides_touched"] > 0,
        "quality": quality,
    }


def segment_accepted_detections(
    page_image: np.ndarray,
    accepted_detections: list[dict[str, Any]],
    model: Any,
    processor: Any,
    segmentation_config: dict[str, Any],
    device: str,
) -> list[SegmentationResult]:
    """Run SAM once for every human-accepted YOLO detection."""

    if not accepted_detections:
        raise ValueError("At least one accepted detection is required")

    results = []
    for detection in accepted_detections:
        if "bbox_xyxy" not in detection:
            raise ValueError("Accepted detection is missing bbox_xyxy")
        results.append(
            segment_with_box(
                page_image=page_image,
                bbox_xyxy=detection["bbox_xyxy"],
                model=model,
                processor=processor,
                segmentation_config=segmentation_config,
                device=device,
            )
        )
    return results


def estimate_ink_mask(signature_crop: np.ndarray) -> np.ndarray:
    """Estimate dark ink independently using OpenCV Otsu thresholding."""

    _validate_rgb_image(signature_crop)
    grayscale = cv2.cvtColor(signature_crop, cv2.COLOR_RGB2GRAY)
    _, ink_mask = cv2.threshold(
        grayscale,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return ink_mask > 0


def measure_mask(
    mask: np.ndarray,
    ink_mask: np.ndarray,
) -> dict[str, float | int]:
    """Measure area, ink agreement, boundary contact, and rectangularity."""

    boolean_mask = mask.astype(bool)
    boolean_ink = ink_mask.astype(bool)
    if boolean_mask.shape != boolean_ink.shape:
        raise ValueError("SAM mask and ink mask must have the same shape")

    total_pixels = boolean_mask.size
    foreground_pixels = int(boolean_mask.sum())
    mask_area_ratio = (
        foreground_pixels / total_pixels if total_pixels else 0.0
    )

    intersection = int(np.logical_and(boolean_mask, boolean_ink).sum())
    union = int(np.logical_or(boolean_mask, boolean_ink).sum())
    ink_agreement = intersection / union if union else 0.0

    boundary_sides_touched = sum(
        (
            bool(boolean_mask[0, :].any()),
            bool(boolean_mask[-1, :].any()),
            bool(boolean_mask[:, 0].any()),
            bool(boolean_mask[:, -1].any()),
        )
    )

    foreground_coordinates = cv2.findNonZero(
        boolean_mask.astype(np.uint8)
    )
    if foreground_coordinates is None:
        rectangularity = 0.0
    else:
        _, _, box_width, box_height = cv2.boundingRect(
            foreground_coordinates
        )
        bounding_area = box_width * box_height
        rectangularity = (
            foreground_pixels / bounding_area
            if bounding_area
            else 0.0
        )

    return {
        "mask_area_ratio": float(mask_area_ratio),
        "ink_agreement": float(ink_agreement),
        "boundary_sides_touched": int(boundary_sides_touched),
        "rectangularity": float(rectangularity),
    }


def calculate_candidate_score(
    sam_predicted_iou: float,
    measurements: dict[str, float | int],
    segmentation_config: dict[str, Any],
) -> float:
    """Rank candidate masks using simple, visible rules."""

    area_ratio = float(measurements["mask_area_ratio"])
    minimum_area = float(
        segmentation_config["minimum_mask_area_ratio"]
    )
    maximum_area = float(
        segmentation_config["maximum_mask_area_ratio"]
    )
    plausible_area = float(minimum_area <= area_ratio <= maximum_area)
    ink_agreement = float(measurements["ink_agreement"])

    # SAM confidence matters, but independent ink agreement receives equal
    # weight because thin handwriting is not SAM's only training objective.
    return (
        0.4 * max(0.0, min(1.0, sam_predicted_iou))
        + 0.4 * max(0.0, min(1.0, ink_agreement))
        + 0.2 * plausible_area
    )


def assess_segmentation_quality(
    mask: np.ndarray,
    ink_mask: np.ndarray,
    sam_predicted_iou: float,
    segmentation_config: dict[str, Any],
) -> SegmentationQuality:
    """Apply interpretable quality checks to the selected SAM mask."""

    measurements = measure_mask(mask, ink_mask)
    area_ratio = float(measurements["mask_area_ratio"])
    ink_agreement = float(measurements["ink_agreement"])
    boundary_sides = int(measurements["boundary_sides_touched"])
    rectangularity = float(measurements["rectangularity"])
    minimum_area = float(
        segmentation_config["minimum_mask_area_ratio"]
    )
    maximum_area = float(
        segmentation_config["maximum_mask_area_ratio"]
    )
    minimum_ink_agreement = float(
        segmentation_config["minimum_ink_agreement"]
    )
    maximum_boundary_contact = float(
        segmentation_config["maximum_boundary_contact_ratio"]
    )
    maximum_boundary_sides = int(round(4 * maximum_boundary_contact))
    rectangularity_limit = float(
        segmentation_config["rectangularity_warning_threshold"]
    )

    checks = {
        "mask_not_empty": bool(mask.any()),
        "area_plausible": minimum_area <= area_ratio <= maximum_area,
        "boundary_contact_acceptable": boundary_sides <= maximum_boundary_sides,
        "ink_agreement_acceptable": ink_agreement >= minimum_ink_agreement,
        "not_large_rectangle": rectangularity < rectangularity_limit,
    }
    warnings: list[str] = []
    if not checks["mask_not_empty"]:
        warnings.append("SAM mask is empty")
    if area_ratio < minimum_area:
        warnings.append("SAM mask area is unusually small")
    if area_ratio > maximum_area:
        warnings.append("SAM mask area is unusually large")
    if not checks["boundary_contact_acceptable"]:
        warnings.append("SAM mask touches too many crop boundaries")
    if not checks["ink_agreement_acceptable"]:
        warnings.append("SAM mask disagrees with the OpenCV ink estimate")
    if not checks["not_large_rectangle"]:
        warnings.append("SAM mask resembles a large rectangular region")

    quality_score = sum(checks.values()) / len(checks)
    return {
        "passed": not warnings,
        "quality_score": round(quality_score, 3),
        "sam_predicted_iou": round(float(sam_predicted_iou), 6),
        "mask_area_ratio": round(area_ratio, 6),
        "ink_agreement": round(ink_agreement, 6),
        "boundary_sides_touched": boundary_sides,
        "rectangularity": round(rectangularity, 6),
        "warnings": warnings,
    }


def create_mask_overlay(
    page_image: np.ndarray,
    full_page_mask: np.ndarray,
    colour: tuple[int, int, int] = (0, 180, 255),
    opacity: float = 0.45,
) -> np.ndarray:
    """Overlay a translucent RGB colour on selected SAM pixels."""

    _validate_rgb_image(page_image)
    boolean_mask = full_page_mask.astype(bool)
    if boolean_mask.shape != page_image.shape[:2]:
        raise ValueError("full_page_mask dimensions must match page_image")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between 0 and 1")

    overlay = page_image.copy()
    colour_array = np.asarray(colour, dtype=np.float32)
    blended_pixels = (
        (1.0 - opacity) * overlay[boolean_mask].astype(np.float32)
        + opacity * colour_array
    )
    overlay[boolean_mask] = np.clip(
        blended_pixels,
        0,
        255,
    ).astype(np.uint8)
    return overlay


def save_segmentation_outputs(
    segmentation_results: list[SegmentationResult],
    page_image: np.ndarray,
    output_directory: str | Path,
) -> list[dict[str, str]]:
    """Save mask overlays, masks, segmented images, and quality JSON."""

    import json

    output_path = Path(output_directory).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    saved_results = []

    for index, result in enumerate(segmentation_results, start=1):
        suffix = f"_{index:02d}" if len(segmentation_results) > 1 else ""
        overlay_path = output_path / f"05_sam_mask_overlay{suffix}.png"
        mask_path = output_path / f"05_sam_mask{suffix}.png"
        segmented_path = (
            output_path / f"06_segmented_signature{suffix}.png"
        )
        quality_path = (
            output_path / f"segmentation_quality{suffix}.json"
        )

        overlay = create_mask_overlay(
            page_image,
            result["full_page_mask"],
        )
        _save_rgb_image(overlay_path, overlay)
        if not cv2.imwrite(
            str(mask_path),
            result["mask"].astype(np.uint8) * 255,
        ):
            raise OSError(f"Could not save mask: {mask_path}")
        _save_rgb_image(segmented_path, result["segmented_signature"])
        quality_path.write_text(
            json.dumps(result["quality"], indent=2),
            encoding="utf-8",
        )
        saved_results.append(
            {
                "overlay_path": str(overlay_path),
                "mask_path": str(mask_path),
                "segmented_signature_path": str(segmented_path),
                "quality_path": str(quality_path),
            }
        )
    return saved_results


def _extract_candidate_masks(
    processed_masks: Any,
    expected_height: int,
    expected_width: int,
) -> list[np.ndarray]:
    """Convert varying SAM tensor layouts into page-sized Boolean masks."""

    if hasattr(processed_masks, "detach"):
        mask_array = processed_masks.detach().cpu().numpy()
    else:
        mask_array = np.asarray(processed_masks)

    mask_array = np.squeeze(mask_array)
    if mask_array.ndim == 2:
        mask_array = mask_array[np.newaxis, ...]
    if mask_array.ndim != 3:
        raise RuntimeError(
            f"Unexpected processed SAM mask shape: {mask_array.shape}"
        )

    candidates = []
    for mask in mask_array:
        if mask.shape != (expected_height, expected_width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (expected_width, expected_height),
                interpolation=cv2.INTER_NEAREST,
            )
        candidates.append(mask.astype(bool))
    return candidates


def _validate_and_clamp_box(
    bbox_xyxy: list[int],
    page_width: int,
    page_height: int,
) -> tuple[int, int, int, int]:
    """Validate and restrict a box to the page dimensions."""

    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain four values")
    x1, y1, x2, y2 = (int(round(value)) for value in bbox_xyxy)
    x1 = max(0, min(x1, page_width))
    x2 = max(0, min(x2, page_width))
    y1 = max(0, min(y1, page_height))
    y2 = max(0, min(y2, page_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box is empty after page clamping")
    return x1, y1, x2, y2


def _save_rgb_image(path: Path, image: np.ndarray) -> None:
    """Save an RGB image with OpenCV's required BGR conversion."""

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr_image):
        raise OSError(f"Could not save image: {path}")


def _validate_rgb_image(image: np.ndarray) -> None:
    """Reject empty or incorrectly shaped RGB arrays."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.size == 0:
        raise ValueError("image is empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected RGB shape (height, width, 3), received {image.shape}"
        )

