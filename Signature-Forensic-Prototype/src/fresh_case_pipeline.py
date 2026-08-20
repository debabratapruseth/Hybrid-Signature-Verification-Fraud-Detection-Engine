"""Reusable inference-only runners for Branches 1, 2, and 3.

This module deliberately contains no training or dataset preparation. It uses
the saved YOLO and Siamese checkpoints plus the configured SAM model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .cleaning import (
    clean_segmentation_results,
    load_cleaning_config,
    save_cleaning_outputs,
)
from .detection import (
    assess_detection_quality,
    detect_signatures,
    draw_detections,
    load_detection_config,
    load_yolo_model,
    save_detection_outputs,
)
from .document import load_document
from .forensics import analyze_signature_forensics, load_forensics_config
from .geometry.pipeline import run_structural_ai_branch, save_structural_ai_result
from .document_forensics.pipeline import (
    run_document_forensics_branch,
    save_document_forensics_result,
)
from .risk import fuse_signature_evidence, load_risk_config, save_risk_result
from .segmentation import (
    load_sam_model,
    load_segmentation_config,
    save_segmentation_outputs,
    segment_accepted_detections,
)
from .verification import (
    compare_with_references,
    load_calibration,
    load_siamese_checkpoint,
    load_verification_config,
    validate_reference_signatures,
)


# ---------------------------------------------------------------------------
# Case-level validation
# ---------------------------------------------------------------------------
# Keep this validation close to the orchestration boundary. Lower-level image
# modules accept many input types for reuse, but a complete case has a stricter
# contract: one questioned document and exactly three references.

def validate_fresh_case(
    questioned_document: str | Path,
    reference_paths: list[str | Path],
) -> tuple[Path, list[Path]]:
    """Resolve and validate the minimum file-level contract for one run.

    The consolidated experiment deliberately requires exactly three reference
    paths. This check is about file availability only; writer identity and
    suitability still require human confirmation in the Colab runner.

    Returns:
        A resolved questioned-document path and three resolved reference paths.
    """
    document = Path(questioned_document).expanduser().resolve()
    references = [Path(path).expanduser().resolve() for path in reference_paths]
    if not document.is_file():
        raise FileNotFoundError(f"Questioned document not found: {document}")
    if len(references) != 3:
        raise ValueError(f"Exactly three references are required; found {len(references)}")
    missing = [str(path) for path in references if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Reference images not found: {missing}")
    return document, references


# ---------------------------------------------------------------------------
# Phase 1: document preparation and detection preview
# ---------------------------------------------------------------------------
# This phase deliberately stops before segmentation. A human must see every
# candidate because model confidence cannot identify which of several genuine
# signature regions is relevant to the business question.

def preview_branch1_detections(
    *,
    questioned_document: str | Path,
    reference_paths: list[str | Path],
    run_directory: str | Path,
    config_path: str | Path,
    page_number: int = 0,
    device: str | None = None,
    detection_confidence_override: float | None = None,
) -> dict[str, Any]:
    """Run only document preparation and YOLO so a human can choose a box.

    This function is intentionally separated from ``run_branch1_fresh``.
    Detection confidence answers "how signature-like is this box?" and cannot
    answer "which signature is relevant to this case?" The caller must display
    the returned candidates and obtain a human selection.

    The returned dictionary is also a reusable in-memory cache. Passing it to
    ``run_branch1_fresh`` avoids decoding the document and running YOLO twice.
    """
    document_path, references = validate_fresh_case(
        questioned_document, reference_paths
    )
    run = Path(run_directory).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    config_file = Path(config_path).expanduser().resolve()
    project = config_file.parent
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # The original and preprocessed arrays are independent. Save both now so a
    # later failure cannot erase the evidence of what entered detection.
    document_result = load_document(document_path, page_number, config_file)
    _save_rgb(run / "01_original_page.png", document_result["original_image"])
    _save_rgb(run / "02_preprocessed_page.png", document_result["preprocessed_image"])
    (run / "document_quality.json").write_text(
        json.dumps(document_result["quality"], indent=2), encoding="utf-8"
    )

    # A confidence override is runtime-only. Copy the dictionary before
    # changing it so the caller's config and Drive YAML remain unchanged.
    detection_config = load_detection_config(config_file)
    if detection_confidence_override is not None:
        if not 0.0 <= detection_confidence_override <= 1.0:
            raise ValueError(
                "detection_confidence_override must be between 0 and 1."
            )
        detection_config = dict(detection_config)
        detection_config["confidence_threshold"] = float(
            detection_confidence_override
        )
    detector = load_yolo_model(
        _project_path(project, detection_config["model_path"])
    )
    detections = detect_signatures(
        document_result["preprocessed_image"],
        detector,
        detection_config,
        device=0 if device == "cuda" else "cpu",
    )
    detection_outputs = save_detection_outputs(
        detections,
        draw_detections(document_result["preprocessed_image"], detections),
        run,
    )
    return {
        "document_path": document_path,
        "reference_paths": references,
        "run_directory": run,
        "config_file": config_file,
        "device": device,
        "document_result": document_result,
        "detection_config": detection_config,
        "detections": detections,
        "detection_outputs": detection_outputs,
    }


# ---------------------------------------------------------------------------
# Phase 2: accepted-box processing and Branch 1
# ---------------------------------------------------------------------------

def run_branch1_fresh(
    *,
    questioned_document: str | Path,
    reference_paths: list[str | Path],
    run_directory: str | Path,
    config_path: str | Path,
    accepted_detection_number: int,
    page_number: int = 0,
    device: str | None = None,
    detection_confidence_override: float | None = None,
    manual_signature_bbox_xyxy: list[int] | None = None,
    detection_preview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Branch 1 inference and save its canonical fusion JSON.

    Processing order:

    1. reuse or create document/YOLO preview evidence;
    2. accept one numbered detection or a validated human box;
    3. use the accepted box to prompt SAM;
    4. conservatively clean and normalize the selected ink;
    5. compare the cleaned signature with exactly three references;
    6. calculate explainable structural/duplicate observations;
    7. apply Branch 1 quality gates and visible risk weights; and
    8. save ``risk_fusion_result.json`` for final three-branch fusion.

    A manual box receives confidence ``0.0`` and an explicit selection method.
    This prevents human input from being misrepresented as model confidence.
    """
    if detection_preview is None:
        detection_preview = preview_branch1_detections(
            questioned_document=questioned_document,
            reference_paths=reference_paths,
            run_directory=run_directory,
            config_path=config_path,
            page_number=page_number,
            device=device,
            detection_confidence_override=detection_confidence_override,
        )
    document_path = detection_preview["document_path"]
    references = detection_preview["reference_paths"]
    run = detection_preview["run_directory"]
    config_file = detection_preview["config_file"]
    project = config_file.parent
    device = detection_preview["device"]
    document_result = detection_preview["document_result"]
    detection_config = detection_preview["detection_config"]
    detections = detection_preview["detections"]
    if not detections and manual_signature_bbox_xyxy is None:
        height, width = document_result["preprocessed_image"].shape[:2]
        raise ValueError(
            "YOLO found no signature candidates. The saved preprocessed page "
            f"is {width} × {height} pixels. Retry with a lower confidence "
            "override or provide manual_signature_bbox_xyxy=[x1, y1, x2, y2]."
        )

    if manual_signature_bbox_xyxy is not None:
        manual_box = _validate_manual_box(
            manual_signature_bbox_xyxy,
            document_result["preprocessed_image"],
        )
        x1, y1, x2, y2 = manual_box
        # Use the same data shape as a YOLO result so downstream modules need no
        # special branch. Confidence remains zero and the explicit method field
        # preserves provenance.
        manual_detection = {
            "class_name": "signature",
            "class_id": 0,
            "confidence": 0.0,
            "bbox_xyxy": manual_box,
            "crop": document_result["preprocessed_image"][y1:y2, x1:x2].copy(),
            "quality": assess_detection_quality(
                manual_box,
                0.0,
                document_result["preprocessed_image"].shape[1],
                document_result["preprocessed_image"].shape[0],
                detection_config,
            ),
            "selection_method": "human_supplied_bounding_box",
        }
        detections_for_saving = detections + [manual_detection]
        accepted = [manual_detection]
        accepted_saved_index = len(detections_for_saving)
    else:
        accepted_index = accepted_detection_number - 1
        if not 0 <= accepted_index < len(detections):
            raise IndexError(
                f"Accepted detection must be between 1 and {len(detections)}."
            )
        detections_for_saving = detections
        accepted = [detections[accepted_index]]
        accepted_saved_index = accepted_index + 1

    detection_outputs = save_detection_outputs(
        detections_for_saving,
        draw_detections(
            document_result["preprocessed_image"],
            detections_for_saving,
        ),
        run,
    )
    (run / "accepted_detection.json").write_text(
        json.dumps(
            {key: value for key, value in accepted[0].items() if key != "crop"},
            indent=2,
        ),
        encoding="utf-8",
    )

    # SAM receives only the accepted box. Pixels outside it cannot be recovered,
    # which is why the Colab padding preview occurs before this point.
    segmentation_config = load_segmentation_config(config_file)
    sam, processor = load_sam_model(segmentation_config["model_name"], device)
    segmentation_results = segment_accepted_detections(
        document_result["preprocessed_image"],
        accepted,
        sam,
        processor,
        segmentation_config,
        device,
    )
    save_segmentation_outputs(
        segmentation_results,
        document_result["preprocessed_image"],
        run,
    )

    # Cleaning is conservative and may return the unmodified SAM-derived ink
    # when proposed removal would be destructive.
    cleaning_config = load_cleaning_config(config_file)
    cleaning_results = clean_segmentation_results(
        segmentation_results, cleaning_config
    )
    save_cleaning_outputs(cleaning_results, run)

    # Validate the exact same references that will later enter Branches 2 and 3.
    # Exact duplicates would falsely reduce observed reference variation.
    verification_config = load_verification_config(config_file)
    reference_quality = validate_reference_signatures(
        references,
        minimum_reference_count=int(
            verification_config["minimum_reference_count"]
        ),
    )
    if not reference_quality["passed"]:
        raise ValueError(f"Reference checks failed: {reference_quality['warnings']}")
    model, _ = load_siamese_checkpoint(
        _project_path(project, verification_config["model_path"]), device
    )
    calibration = load_calibration(
        _project_path(project, verification_config["calibration_path"])
    )
    verification = compare_with_references(
        model,
        cleaning_results[0]["cleaned_signature"],
        references,
        verification_config,
        calibration,
        device,
    )
    verification["input_quality"] = cleaning_results[0]["quality"]
    (run / "verification_result.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )

    # These explainable observations complement the learned score. They do not
    # replace Branch 2, which calculates a richer descriptor set separately.
    forensics_config, duplicate_config = load_forensics_config(config_file)
    forensics_config, duplicate_config = _normalise_branch1_forensics_config(
        forensics_config,
        duplicate_config,
    )
    forensic_result = analyze_signature_forensics(
        cleaning_results[0]["cleaned_signature"],
        references,
        forensics_config,
        duplicate_config,
    )
    foreground = int(np.sum(cleaning_results[0]["binary_signature"] < 128))
    removed = int(np.sum(cleaning_results[0]["removed_line_mask"] > 0))
    line_ratio = removed / max(foreground, 1)
    forensic_result["line_overlap"] = {
        "removed_line_ratio": round(line_ratio, 6),
        "possible_line_overlap": line_ratio
        > float(forensics_config["line_overlap_warning_ratio"]),
        "assessment_limited_by_cleaning_fallback": cleaning_results[0][
            "quality"
        ]["fallback_used"],
    }
    _save_branch1_forensics(forensic_result, run / "forensics")

    # CEDAR is not a case reference. Its saved aggregate report only estimates
    # how much trust to place in the BHSig-trained verifier on unseen data.
    cedar_path = project / "models/cedar_external_test_report.json"
    cedar_report = (
        json.loads(cedar_path.read_text(encoding="utf-8"))
        if cedar_path.is_file()
        else None
    )
    risk_result = fuse_signature_evidence(
        document_quality=document_result["quality"],
        detection=accepted[0],
        segmentation_quality=segmentation_results[0]["quality"],
        cleaning_quality=cleaning_results[0]["quality"],
        verification_result=verification,
        calibration=calibration,
        forensic_result=forensic_result,
        external_validation_report=cedar_report,
        risk_config=load_risk_config(config_file),
    )
    save_risk_result(risk_result, run / "risk_fusion_result.json")
    return {
        "document_result": document_result,
        "accepted_detection": accepted[0],
        "cleaning_result": cleaning_results[0],
        "verification_result": verification,
        "forensic_result": forensic_result,
        "risk_result": risk_result,
        "accepted_crop_path": detection_outputs["crop_paths"][
            accepted_saved_index - 1
        ],
    }


# ---------------------------------------------------------------------------
# Branch 2 and Branch 3 adapters
# ---------------------------------------------------------------------------
# These small wrappers guarantee canonical output directories and keep the
# notebook independent of each package's internal save function names.

def run_branch2_fresh(
    *,
    cleaned_questioned_signature: Any,
    reference_paths: list[str | Path],
    run_directory: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Run and save the reusable Structural AI branch.

    Args:
        cleaned_questioned_signature: Branch 1's normalized RGB signature.
        reference_paths: The same three references used by Branch 1.
        run_directory: Case-specific output directory shared by all branches.
        config_path: Project configuration containing the ``geometry`` section.

    Returns:
        ``result`` holds in-memory measurements and visuals; ``saved`` maps
        artifact names to persistent paths.
    """
    result = run_structural_ai_branch(
        cleaned_questioned_signature,
        list(reference_paths),
        config_path,
    )
    saved = save_structural_ai_result(
        result, Path(run_directory) / "branch2_geometry"
    )
    return {"result": result, "saved": saved}


def run_branch3_fresh(
    *,
    original_page: Any,
    signature_bbox_xyxy: list[int],
    reference_paths: list[str | Path],
    run_directory: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Run and save the reusable Document Forensics branch.

    Branch 3 intentionally receives the preserved original page rather than the
    contrast-enhanced working page. Compression and residual-noise screening
    would be distorted if performed on the preprocessed image.
    """
    result = run_document_forensics_branch(
        original_page=original_page,
        signature_bbox_xyxy=signature_bbox_xyxy,
        reference_signatures=list(reference_paths),
        config_path=config_path,
    )
    saved = save_document_forensics_result(
        result, Path(run_directory) / "branch3_document_forensics"
    )
    return {"result": result, "saved": saved}


def _project_path(project: Path, configured_path: str | Path) -> Path:
    """Resolve a configuration path relative to the project when necessary."""

    path = Path(configured_path)
    return path if path.is_absolute() else project / path


def _normalise_branch1_forensics_config(
    forensics_config: dict[str, Any],
    duplicate_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Support both the original and current visible configuration names.

    The function copies both dictionaries before filling values. Compatibility
    must never silently rewrite the user's Drive configuration.
    """
    forensic = dict(forensics_config)
    duplicate = dict(duplicate_config)

    forensic.setdefault("minimum_foreground_ratio", 0.01)
    forensic.setdefault("maximum_foreground_ratio", 0.50)
    forensic.setdefault(
        "orb_feature_count",
        int(forensic.get("orb_max_keypoints", 1000)),
    )
    forensic.setdefault(
        "orb_ratio_test",
        float(forensic.get("orb_match_threshold", 0.75)),
    )
    forensic.setdefault("minimum_alignment_matches", 8)
    forensic.setdefault(
        "ransac_reprojection_threshold",
        float(forensic.get("ransac_threshold_pixels", 5.0)),
    )
    forensic.setdefault("line_overlap_warning_ratio", 0.10)

    duplicate.setdefault("perceptual_hash_size", 16)
    duplicate.setdefault(
        "perceptual_hash_distance_warning",
        int(duplicate.get("phash_hamming_distance_threshold", 8)),
    )
    duplicate.setdefault("minimum_good_matches", 12)
    duplicate.setdefault("good_match_ratio_warning", 0.65)
    duplicate.setdefault("minimum_ransac_inliers", 8)
    duplicate.setdefault("ransac_inlier_ratio_warning", 0.60)
    return forensic, duplicate


def _save_rgb(path: Path, image: np.ndarray) -> None:
    """Persist one internal RGB array through OpenCV's BGR writer."""

    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save image: {path}")


def _validate_manual_box(values: list[int], page: np.ndarray) -> list[int]:
    """Validate and normalize a human `[x1, y1, x2, y2]` selection."""

    if len(values) != 4:
        raise ValueError("Manual bounding box must contain [x1, y1, x2, y2].")
    x1, y1, x2, y2 = [int(value) for value in values]
    height, width = page.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"Manual box {[x1, y1, x2, y2]} is outside the "
            f"{width} × {height} page."
        )
    return [x1, y1, x2, y2]


def _save_branch1_forensics(result: dict[str, Any], directory: Path) -> None:
    """Separate large Branch 1 images from the readable JSON measurements."""

    directory.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in result.items() if key != "visuals"}
    (directory / "structural_and_duplicate_report.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    for visual in result["visuals"]:
        index = int(visual["reference_index"])
        _save_rgb(directory / f"reference_{index:02d}_overlay.png", visual["overlay"])
        cv2.imwrite(
            str(directory / f"reference_{index:02d}_xor.png"), visual["xor_map"]
        )
        _save_rgb(
            directory / f"reference_{index:02d}_orb.png", visual["orb_matches"]
        )
