"""Load and validate saved evidence from Branches 1, 2, and 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one required JSON object with a clear error message."""

    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {file_path}")
    value = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {file_path}")
    return value


def load_three_branch_evidence(
    run_directory: str | Path,
) -> dict[str, Any]:
    """Load the canonical saved result for each completed branch."""

    run = Path(run_directory).expanduser().resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run}")
    paths = {
        "branch1": run / "risk_fusion_result.json",
        "branch2": (
            run
            / "branch2_geometry"
            / "branch2_geometry_result.json"
        ),
        "branch3": (
            run
            / "branch3_document_forensics"
            / "branch3_document_forensics.json"
        ),
    }
    evidence = {
        name: load_json(path)
        for name, path in paths.items()
    }
    _validate_branch_shapes(evidence)
    return {
        "run_id": run.name,
        "run_directory": str(run),
        "source_paths": {
            name: str(path) for name, path in paths.items()
        },
        **evidence,
    }


def discover_evidence_images(
    run_directory: str | Path,
) -> dict[str, list[str]]:
    """Return existing input and branch-output images in display order."""

    run = Path(run_directory).expanduser().resolve()
    groups = {
        "inputs": [
            run / "01_original_page.png",
            run / "04_original_signature_crop_01.png",
            run / "08_cleaned_signature.png",
        ],
        "branch1_outputs": [
            run / "03_yolo_detection.png",
            run / "05_sam_mask_overlay.png",
            run / "08_cleaned_signature.png",
        ],
        "branch2_outputs": [
            (
                run
                / "branch2_geometry"
                / "questioned_skeleton_points.png"
            ),
            (
                run
                / "branch2_geometry"
                / "questioned_contour_critical_points.png"
            ),
            (
                run
                / "branch2_geometry"
                / "reference_01_overlay.png"
            ),
        ],
        "branch3_outputs": [
            (
                run
                / "branch3_document_forensics"
                / "analysis_regions.png"
            ),
            (
                run
                / "branch3_document_forensics"
                / "ela.png"
            ),
            (
                run
                / "branch3_document_forensics"
                / "noise.png"
            ),
            (
                run
                / "branch3_document_forensics"
                / "copy_paste.png"
            ),
        ],
    }
    return {
        group: [
            str(path)
            for path in paths
            if path.is_file()
        ]
        for group, paths in groups.items()
    }


def _validate_branch_shapes(evidence: dict[str, Any]) -> None:
    """Fail early when a saved file is from an incompatible pipeline stage."""

    required = {
        "branch1": {"overall_risk", "decision", "component_risks"},
        "branch2": {"comparisons", "reference_variation"},
        "branch3": {"risk", "copy_paste", "compression", "noise_analysis"},
    }
    for branch, keys in required.items():
        missing = keys - set(evidence[branch])
        if missing:
            raise ValueError(
                f"{branch} result is missing fields: {sorted(missing)}"
            )
