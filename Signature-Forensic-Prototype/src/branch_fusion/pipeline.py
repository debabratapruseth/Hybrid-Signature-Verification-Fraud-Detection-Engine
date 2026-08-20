"""Orchestration for deterministic branch fusion and required LLM reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import yaml

from .evidence import discover_evidence_images, load_three_branch_evidence
from .fusion import calculate_branch_fusion
from .reports import generate_openai_reports
from .visualization import create_fusion_dashboard


def load_branch_fusion_config(config_path: Path) -> Dict[str, Any]:
    """Load the final-fusion section and reject a missing/non-mapping value."""

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    section = config.get("branch_fusion")
    if not isinstance(section, dict):
        raise ValueError("config.yaml needs a 'branch_fusion' section.")
    return section


def prepare_branch_fusion(run_directory: Path, config_path: Path) -> Dict[str, Any]:
    """Load canonical branch evidence and calculate deterministic fusion.

    No OpenAI call occurs here. The returned bundle freezes the evidence,
    numerical result, image paths, and configuration used for later reporting.
    """

    run_directory = Path(run_directory)
    config = load_branch_fusion_config(config_path)
    evidence = load_three_branch_evidence(run_directory)
    fusion = calculate_branch_fusion(evidence, config)
    return {
        "run_directory": str(run_directory),
        "config": config,
        "evidence": evidence,
        "fusion": fusion,
        "images": discover_evidence_images(run_directory),
    }


def generate_required_reports(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the senior and technical reports from the frozen bundle."""

    config = bundle["config"]
    return generate_openai_reports(
        bundle,
        model=config.get("openai_model", "gpt-5.6-terra"),
        api_key_environment_variable=config.get(
            "api_key_environment_variable", "OPENAI_API_KEY"
        ),
    )


def save_branch_fusion_outputs(
    bundle: Dict[str, Any],
    reports: Dict[str, Any],
) -> Dict[str, str]:
    """Persist all final machine-readable, human-readable, and visual outputs.

    JSON and Markdown are written independently so report generation never
    changes the deterministic result. The dashboard is converted from the
    project's RGB convention to OpenCV BGR only at write time.
    """

    output_directory = Path(bundle["run_directory"]) / "branch_fusion"
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "fusion_result": output_directory / "branch_fusion_result.json",
        "report_evidence": output_directory / "report_evidence.json",
        "senior_report": output_directory / "senior_stakeholder_report.md",
        "technical_report": output_directory / "technical_fusion_report.md",
        "report_metadata": output_directory / "report_metadata.json",
        "dashboard": output_directory / "branch_fusion_dashboard.png",
    }
    paths["fusion_result"].write_text(
        json.dumps(bundle["fusion"], indent=2), encoding="utf-8"
    )
    paths["report_evidence"].write_text(
        json.dumps(reports["report_evidence"], indent=2), encoding="utf-8"
    )
    paths["senior_report"].write_text(reports["senior_report"], encoding="utf-8")
    paths["technical_report"].write_text(reports["technical_report"], encoding="utf-8")
    paths["report_metadata"].write_text(
        json.dumps(reports["metadata"], indent=2), encoding="utf-8"
    )
    dashboard_rgb = create_fusion_dashboard(bundle["images"], bundle["fusion"])
    cv2.imwrite(
        str(paths["dashboard"]),
        cv2.cvtColor(dashboard_rgb, cv2.COLOR_RGB2BGR),
    )
    return {name: str(path) for name, path in paths.items()}


def run_branch_fusion(
    run_directory: Path,
    config_path: Path,
) -> Dict[str, Any]:
    """Convenience entry point that requires OpenAI and saves all artifacts."""
    bundle = prepare_branch_fusion(run_directory, config_path)
    reports = generate_required_reports(bundle)
    saved_paths = save_branch_fusion_outputs(bundle, reports)
    return {"bundle": bundle, "reports": reports, "saved_paths": saved_paths}
