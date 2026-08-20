"""YOLO signature detection, training, validation, and visualization.

YOLO answers: "Where is a possible signature on this page?"

It does not verify who wrote a signature and does not prove authenticity. A
high confidence score only means that the detected region resembles patterns
learned from the detector's training data.

All input and output colour images in this project use RGB channel order.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np
import yaml


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class DetectionQuality(TypedDict):
    """Understandable quality information for one YOLO box."""

    passed: bool
    box_area_ratio: float
    touches_boundary: bool
    warnings: list[str]


class DetectionResult(TypedDict):
    """Data contract for one possible signature detection."""

    class_name: str
    class_id: int
    confidence: float
    bbox_xyxy: list[int]
    crop: np.ndarray
    quality: DetectionQuality


def load_detection_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Read the ``detection`` section from config.yaml."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a dictionary")
    if not isinstance(config.get("detection"), dict):
        raise ValueError("config.yaml must contain a 'detection' section")
    return config["detection"]


def load_yolo_model(
    model_path: str | Path,
    require_signature_class: bool = True,
) -> Any:
    """Load an Ultralytics YOLO model.

    Args:
        model_path: Custom signature checkpoint or an Ultralytics model name.
        require_signature_class: When true, reject checkpoints whose class list
            does not include ``signature``. This prevents accidental use of an
            ordinary COCO detector as though it were a signature detector.

    Returns:
        An Ultralytics ``YOLO`` model.

    Raises:
        ImportError: If Ultralytics is not installed.
        FileNotFoundError: If a local checkpoint path is missing.
        ValueError: If the model has no signature class.
    """

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError(
            "Ultralytics is not installed. Run "
            "'pip install ultralytics' in the active environment."
        ) from error

    model_value = str(model_path)
    looks_like_local_path = (
        Path(model_value).suffix.lower() in {".pt", ".onnx"}
        and ("/" in model_value or "\\" in model_value)
    )
    if looks_like_local_path and not Path(model_value).expanduser().is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {model_value}")

    model = YOLO(model_value)
    available_names = _normalise_class_names(model.names)

    if require_signature_class:
        lower_names = {
            name.lower().strip()
            for name in available_names.values()
        }
        if "signature" not in lower_names:
            raise ValueError(
                "This checkpoint does not contain a 'signature' class. "
                "Train YOLO on signature bounding boxes or load a compatible "
                "signature-detection checkpoint."
            )

    return model


def train_detector(
    data_yaml_path: str | Path,
    base_model: str = "yolo11n.pt",
    epochs: int = 50,
    image_size: int = 1280,
    batch_size: int = 8,
    project_directory: str | Path = "runs/detection",
    run_name: str = "signature_detector",
    device: str | int | None = None,
    random_seed: int = 42,
) -> dict[str, str]:
    """Train a lightweight YOLO detector on document-level signature boxes.

    The dataset YAML must point to training and validation images and define:

    ``names: {0: signature}``

    Returns:
        Paths to the training directory and best checkpoint.
    """

    data_path = Path(data_yaml_path).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"YOLO dataset YAML not found: {data_path}")
    if epochs <= 0 or image_size <= 0 or batch_size <= 0:
        raise ValueError("epochs, image_size, and batch_size must be positive")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError("Install ultralytics before training YOLO") from error

    model = YOLO(base_model)
    training_result = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        project=str(Path(project_directory).expanduser()),
        name=run_name,
        device=device,
        seed=random_seed,
        deterministic=True,
        plots=True,
    )

    save_directory = Path(training_result.save_dir).resolve()
    best_checkpoint = save_directory / "weights" / "best.pt"
    return {
        "save_directory": str(save_directory),
        "best_checkpoint": str(best_checkpoint),
    }


def validate_detector(
    model: Any,
    data_yaml_path: str | Path,
    image_size: int = 1280,
    confidence_threshold: float = 0.25,
    device: str | int | None = None,
) -> dict[str, float | int | None]:
    """Validate a trained detector and return common detection metrics.

    ``missed_signature_rate`` is reported as ``1 - recall``. Ultralytics does
    not expose a stable total false-detection count in every version, so that
    field is ``None`` when it cannot be calculated safely.
    """

    data_path = Path(data_yaml_path).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"YOLO dataset YAML not found: {data_path}")

    metrics = model.val(
        data=str(data_path),
        imgsz=image_size,
        conf=confidence_threshold,
        device=device,
        plots=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "map50": round(float(metrics.box.map50), 6),
        "map50_95": round(float(metrics.box.map), 6),
        "missed_signature_rate": round(1.0 - recall, 6),
        "false_detection_count": None,
    }


def detect_signatures(
    page_image: np.ndarray,
    model: Any,
    detection_config: dict[str, Any],
    device: str | int | None = None,
) -> list[DetectionResult]:
    """Find all accepted signature regions in one RGB page image.

    Each returned crop is copied from the supplied page before annotation. The
    crop therefore remains unchanged when visualization boxes are later drawn.
    """

    _validate_rgb_image(page_image)
    confidence_threshold = float(
        detection_config["confidence_threshold"]
    )
    iou_threshold = float(detection_config["iou_threshold"])
    image_size = int(detection_config["image_size"])

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")

    # Ultralytics treats NumPy input as OpenCV BGR, whereas this project stores
    # colour images as RGB.
    page_bgr = cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR)
    predictions = model.predict(
        source=page_bgr,
        conf=confidence_threshold,
        iou=iou_threshold,
        imgsz=image_size,
        device=device,
        verbose=False,
    )
    if not predictions:
        return []

    prediction = predictions[0]
    class_names = _normalise_class_names(prediction.names)
    detections: list[DetectionResult] = []
    boxes = prediction.boxes

    if boxes is None:
        return detections

    page_height, page_width = page_image.shape[:2]
    for coordinates, confidence, class_id_value in zip(
        boxes.xyxy.cpu().numpy(),
        boxes.conf.cpu().numpy(),
        boxes.cls.cpu().numpy(),
    ):
        class_id = int(class_id_value)
        class_name = class_names.get(class_id, f"class_{class_id}")

        # The first prototype processes only the signature class.
        if class_name.lower().strip() != "signature":
            continue

        x1, y1, x2, y2 = _clamp_box(
            coordinates,
            page_width,
            page_height,
        )
        if x2 <= x1 or y2 <= y1:
            LOGGER.warning("Ignoring an empty YOLO box: %s", coordinates)
            continue

        crop = page_image[y1:y2, x1:x2].copy()
        quality = assess_detection_quality(
            bbox_xyxy=[x1, y1, x2, y2],
            confidence=float(confidence),
            page_width=page_width,
            page_height=page_height,
            detection_config=detection_config,
        )
        detections.append(
            {
                "class_name": class_name,
                "class_id": class_id,
                "confidence": round(float(confidence), 6),
                "bbox_xyxy": [x1, y1, x2, y2],
                "crop": crop,
                "quality": quality,
            }
        )

    _add_overlap_warnings(detections, iou_threshold)
    return detections


def assess_detection_quality(
    bbox_xyxy: list[int],
    confidence: float,
    page_width: int,
    page_height: int,
    detection_config: dict[str, Any],
) -> DetectionQuality:
    """Check confidence, size, and page-boundary contact for one box."""

    x1, y1, x2, y2 = bbox_xyxy
    page_area = page_width * page_height
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    box_area_ratio = box_area / page_area if page_area else 0.0
    boundary_margin = int(detection_config["boundary_margin_pixels"])
    touches_boundary = (
        x1 <= boundary_margin
        or y1 <= boundary_margin
        or x2 >= page_width - boundary_margin
        or y2 >= page_height - boundary_margin
    )

    warnings: list[str] = []
    if confidence < float(detection_config["confidence_threshold"]):
        warnings.append("Detection confidence is below the configured threshold")
    if box_area_ratio < float(detection_config["minimum_box_area_ratio"]):
        warnings.append("Detection box is unusually small")
    if box_area_ratio > float(detection_config["maximum_box_area_ratio"]):
        warnings.append("Detection box is unusually large")
    if touches_boundary:
        warnings.append("Detection box touches the page boundary")

    return {
        "passed": not warnings,
        "box_area_ratio": round(box_area_ratio, 6),
        "touches_boundary": touches_boundary,
        "warnings": warnings,
    }


def summarize_detections(
    detections: list[DetectionResult],
) -> dict[str, Any]:
    """Create page-level messages for zero, one, or multiple signatures."""

    count = len(detections)
    warnings: list[str] = []
    if count == 0:
        warnings.append("No signature was detected")
    elif count > 1:
        warnings.append(f"Multiple possible signatures were detected: {count}")

    low_quality_count = sum(
        not detection["quality"]["passed"]
        for detection in detections
    )
    if low_quality_count:
        warnings.append(
            f"{low_quality_count} detection(s) have quality warnings"
        )

    return {
        "signature_detected": count > 0,
        "detection_count": count,
        "passed": count > 0 and low_quality_count == 0,
        "warnings": warnings,
    }


def draw_detections(
    page_image: np.ndarray,
    detections: list[DetectionResult],
) -> np.ndarray:
    """Draw labeled detection boxes on a copy of an RGB page."""

    _validate_rgb_image(page_image)
    visualization = page_image.copy()
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        confidence = detection["confidence"]
        colour = (30, 180, 30) if detection["quality"]["passed"] else (230, 140, 20)
        cv2.rectangle(visualization, (x1, y1), (x2, y2), colour, 3)
        label = f"signature {index}: {confidence:.2f}"
        label_y = max(25, y1 - 10)
        cv2.putText(
            visualization,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2,
            cv2.LINE_AA,
        )
    return visualization


def save_detection_outputs(
    detections: list[DetectionResult],
    visualization: np.ndarray,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Save visualization, unchanged crops, and JSON-safe metadata."""

    output_path = Path(output_directory).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    visualization_path = output_path / "03_yolo_detection.png"
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(visualization_path), visualization_bgr):
        raise OSError(f"Could not save visualization: {visualization_path}")

    metadata: list[dict[str, Any]] = []
    crop_paths: list[str] = []
    for index, detection in enumerate(detections, start=1):
        crop_path = output_path / f"04_original_signature_crop_{index:02d}.png"
        crop_bgr = cv2.cvtColor(detection["crop"], cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(crop_path), crop_bgr):
            raise OSError(f"Could not save crop: {crop_path}")
        crop_paths.append(str(crop_path))
        metadata.append(
            {
                key: value
                for key, value in detection.items()
                if key != "crop"
            }
        )

    metadata_path = output_path / "detection_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "summary": summarize_detections(detections),
                "detections": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "visualization_path": str(visualization_path),
        "crop_paths": crop_paths,
        "metadata_path": str(metadata_path),
    }


def xyxy_to_yolo(
    bbox_xyxy: list[float | int],
    image_width: int,
    image_height: int,
    class_id: int = 0,
) -> str:
    """Convert a pixel ``[x1, y1, x2, y2]`` box to one YOLO label row."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain four values")

    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
        raise ValueError("Bounding box must lie inside the image")

    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return (
        f"{class_id} {x_center:.6f} {y_center:.6f} "
        f"{width:.6f} {height:.6f}"
    )


def _normalise_class_names(names: Any) -> dict[int, str]:
    """Convert Ultralytics list/dictionary names to one dictionary form."""

    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    raise ValueError(f"Unsupported YOLO class-name structure: {type(names)}")


def _clamp_box(
    coordinates: np.ndarray,
    page_width: int,
    page_height: int,
) -> tuple[int, int, int, int]:
    """Round and restrict a YOLO box to valid page pixels."""

    x1, y1, x2, y2 = (int(round(float(value))) for value in coordinates)
    return (
        max(0, min(x1, page_width)),
        max(0, min(y1, page_height)),
        max(0, min(x2, page_width)),
        max(0, min(y2, page_height)),
    )


def _intersection_over_union(
    first_box: list[int],
    second_box: list[int],
) -> float:
    """Calculate intersection-over-union for two XYXY boxes."""

    left = max(first_box[0], second_box[0])
    top = max(first_box[1], second_box[1])
    right = min(first_box[2], second_box[2])
    bottom = min(first_box[3], second_box[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first_box[2] - first_box[0]) * max(
        0, first_box[3] - first_box[1]
    )
    second_area = max(0, second_box[2] - second_box[0]) * max(
        0, second_box[3] - second_box[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _add_overlap_warnings(
    detections: list[DetectionResult],
    overlap_threshold: float,
) -> None:
    """Add warnings when accepted signature boxes overlap strongly."""

    for first_index in range(len(detections)):
        for second_index in range(first_index + 1, len(detections)):
            overlap = _intersection_over_union(
                detections[first_index]["bbox_xyxy"],
                detections[second_index]["bbox_xyxy"],
            )
            if overlap >= overlap_threshold:
                message = (
                    f"Detection overlaps another box (IoU {overlap:.2f})"
                )
                for index in (first_index, second_index):
                    detections[index]["quality"]["warnings"].append(message)
                    detections[index]["quality"]["passed"] = False


def _validate_rgb_image(image: np.ndarray) -> None:
    """Reject empty or incorrectly shaped RGB input."""

    if not isinstance(image, np.ndarray):
        raise TypeError("page_image must be a NumPy array")
    if image.size == 0:
        raise ValueError("page_image is empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"page_image must have RGB shape (height, width, 3), got {image.shape}"
        )

