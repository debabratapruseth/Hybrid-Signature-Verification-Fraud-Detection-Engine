"""Orchestration for Branch 3 document-forensics screening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .common import crop_bbox, load_rgb_image
from .compression import analyze_local_compression
from .copy_paste import locate_signature_reuse
from .duplicate_detection import compare_possible_duplicate
from .noise_analysis import compare_signature_noise
from .risk import calculate_document_forensics_risk
from .visualization import create_three_panel, draw_analysis_regions


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_document_forensics_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load Branch 3 settings or return documented prototype defaults."""

    defaults = {
        "orb_feature_count": 2000,
        "orb_ratio_test": 0.75,
        "ransac_reprojection_threshold": 5.0,
        "minimum_good_matches": 12,
        "minimum_ransac_inliers": 10,
        "ransac_inlier_ratio_warning": 0.75,
        "good_match_ratio_warning": 0.65,
        "perceptual_hash_size": 16,
        "perceptual_hash_distance_warning": 4,
        "jpeg_recompression_quality": 90,
        "ela_z_score_warning": 2.5,
        "noise_blur_sigma": 1.2,
        "noise_surrounding_margin": 40,
        "noise_flat_gradient_threshold": 12.0,
        "noise_log_ratio_warning": 0.50,
        "minimum_copy_paste_matches": 6,
        "minimum_copy_paste_inliers": 6,
        "copy_paste_inlier_ratio_warning": 0.70,
        "template_minimum_scale": 0.75,
        "template_maximum_scale": 1.30,
        "template_scale_steps": 12,
        "template_match_score_warning": 0.72,
        "template_phase_response_warning": 0.05,
        "risk_weights": {
            "copy_paste": 0.35,
            "compression": 0.20,
            "noise": 0.20,
            "duplicate": 0.25,
        },
        "manual_review_risk": 0.30,
        "elevated_screening_risk": 0.60,
    }
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        return defaults
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a dictionary")
    section = config.get("document_forensics")
    if section is None:
        return defaults
    if not isinstance(section, dict):
        raise ValueError("document_forensics must be a dictionary")
    merged = dict(defaults)
    merged.update(section)
    if isinstance(section.get("risk_weights"), dict):
        merged["risk_weights"] = section["risk_weights"]
    return merged


def run_document_forensics_branch(
    *,
    original_page: object,
    signature_bbox_xyxy: list[int] | tuple[int, int, int, int],
    original_signature_crop: object | None = None,
    reference_signatures: list[object] | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Run Branch 3 using preserved Branch 1 evidence."""

    config = load_document_forensics_config(config_path)
    page = load_rgb_image(original_page)
    crop = (
        load_rgb_image(original_signature_crop)
        if original_signature_crop is not None
        else crop_bbox(page, signature_bbox_xyxy)
    )
    copy_paste = locate_signature_reuse(
        page,
        crop,
        signature_bbox_xyxy,
        config,
    )
    compression = analyze_local_compression(
        page,
        signature_bbox_xyxy,
        jpeg_quality=int(config["jpeg_recompression_quality"]),
    )
    noise = compare_signature_noise(
        page,
        signature_bbox_xyxy,
        surrounding_margin=int(config["noise_surrounding_margin"]),
        blur_sigma=float(config["noise_blur_sigma"]),
        flat_gradient_threshold=float(
            config["noise_flat_gradient_threshold"]
        ),
    )
    duplicate_results = []
    duplicate_visuals = []
    for index, reference in enumerate(reference_signatures or [], start=1):
        comparison = compare_possible_duplicate(
            crop,
            reference,
            config,
        )
        duplicate_results.append(
            {
                "reference_index": index,
                **{
                    key: value
                    for key, value in comparison.items()
                    if key != "visualization"
                },
            }
        )
        duplicate_visuals.append(
            {
                "reference_index": index,
                "visualization": comparison["visualization"],
            }
        )
    risk = calculate_document_forensics_risk(
        copy_paste,
        compression,
        noise,
        duplicate_results,
        config,
    )
    analysis_regions = draw_analysis_regions(
        page,
        signature_bbox_xyxy,
        noise["surrounding_bbox_xyxy"],
    )
    evidence_panel = create_three_panel(
        analysis_regions,
        compression["visualization"],
        noise["visualization"],
    )
    return {
        "branch": "Branch 3 - Document Forensics",
        "inputs": {
            "page_shape": list(page.shape),
            "signature_bbox_xyxy": [
                int(value) for value in signature_bbox_xyxy
            ],
            "reference_count": len(reference_signatures or []),
            "used_preserved_original_page": True,
        },
        "copy_paste": _without_images(copy_paste),
        "compression": _without_images(compression),
        "noise_analysis": _without_images(noise),
        "duplicate_comparisons": duplicate_results,
        "risk": risk,
        "limitations": [
            "Rendered PDFs no longer preserve the original PDF object history.",
            "ELA and noise differences are not specific to editing.",
            "Feature matching can fail on thin or low-resolution signatures.",
            "No warning means no configured indicator fired; editing is not ruled out.",
        ],
        "visuals": {
            "copy_paste": copy_paste["visualization"],
            "ela": compression["visualization"],
            "noise": noise["visualization"],
            "analysis_regions": analysis_regions,
            "evidence_panel": evidence_panel,
            "duplicate_matches": duplicate_visuals,
        },
    }


def save_document_forensics_result(
    result: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, str]:
    """Save Branch 3 JSON and every review visualization."""

    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "branch3_document_forensics.json"
    serializable = {
        key: value for key, value in result.items() if key != "visuals"
    }
    report_path.write_text(
        json.dumps(serializable, indent=2),
        encoding="utf-8",
    )
    saved = {"report": str(report_path)}
    for name in (
        "copy_paste",
        "ela",
        "noise",
        "analysis_regions",
        "evidence_panel",
    ):
        path = directory / f"{name}.png"
        _save_rgb(path, result["visuals"][name])
        saved[name] = str(path)
    for item in result["visuals"]["duplicate_matches"]:
        index = int(item["reference_index"])
        path = directory / f"reference_{index:02d}_orb_matches.png"
        _save_rgb(path, item["visualization"])
        saved[f"reference_{index:02d}_orb_matches"] = str(path)
    return saved


def _without_images(result: dict[str, Any]) -> dict[str, Any]:
    """Remove working arrays and visualizations from a result dictionary."""

    return {
        key: value
        for key, value in result.items()
        if key not in {"visualization", "error_map", "residual"}
        and not isinstance(value, np.ndarray)
    }


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    """Save an RGB image and raise if OpenCV cannot write it."""

    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save image: {path}")
