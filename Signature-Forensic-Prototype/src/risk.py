"""Transparent quality validation and rule-based evidence fusion.

The output is a review-priority summary, not an authenticity verdict. Every
component score and decision rule is returned so a reviewer can inspect it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_risk_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, dict[str, Any]]:
    """Load quality, risk-weight, and decision settings."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = ("quality", "risk_weights", "risk_decisions")
    missing = [name for name in required if not isinstance(config.get(name), dict)]
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    return {name: config[name] for name in required}


def _bounded(value: float) -> float:
    """Clamp a prototype risk or quality value to the closed interval [0, 1]."""

    return float(np.clip(value, 0.0, 1.0))


def calculate_verification_reliability(
    external_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Estimate transfer reliability from untouched external-test metrics."""

    if not external_report:
        return {
            "score": 0.0,
            "passed": False,
            "reason": "No untouched external-validation report was provided",
        }
    auc = float(external_report.get("roc_auc", 0.5))
    false_acceptance = float(
        external_report.get(
            "false_acceptance_rate_at_bhsig_threshold",
            1.0,
        )
    )
    false_rejection = float(
        external_report.get(
            "false_rejection_rate_at_bhsig_threshold",
            1.0,
        )
    )
    discrimination = _bounded((auc - 0.5) / 0.5)
    operating_quality = float(
        np.sqrt(
            max(0.0, 1.0 - false_acceptance)
            * max(0.0, 1.0 - false_rejection)
        )
    )
    reliability = _bounded(discrimination * operating_quality)
    return {
        "score": round(reliability, 6),
        "passed": reliability >= 0.50,
        "roc_auc": auc,
        "false_acceptance_rate": false_acceptance,
        "false_rejection_rate": false_rejection,
        "reason": (
            "External generalization is adequate for prototype weighting"
            if reliability >= 0.50
            else "Verification contribution was down-weighted because "
            "external generalization is weak"
        ),
    }


def calculate_verification_risk(
    verification_result: dict[str, Any],
    calibration: dict[str, Any],
) -> float:
    """Map median similarity to a continuous risk using calibrated boundaries."""

    score = float(verification_result["median_similarity"])
    lower = float(calibration["elevated_inconsistency_below"])
    upper = float(calibration["provisionally_consistent_at_or_above"])
    if upper <= lower:
        raise ValueError("Verification calibration boundaries are invalid")
    if score >= upper:
        risk = 0.30 * (1.0 - score) / max(1.0 - upper, 1e-6)
    elif score >= lower:
        risk = 0.30 + 0.40 * (upper - score) / (upper - lower)
    else:
        risk = 0.70 + 0.30 * (lower - score) / max(lower + 1.0, 1e-6)
    spread_penalty = _bounded(
        float(verification_result.get("score_spread", 0.0)) / 0.40
    )
    return _bounded(0.85 * risk + 0.15 * spread_penalty)


def validate_pipeline_quality(
    document_quality: dict[str, Any] | None,
    detection: dict[str, Any] | None,
    segmentation_quality: dict[str, Any] | None,
    cleaning_quality: dict[str, Any] | None,
    verification_result: dict[str, Any] | None,
    forensic_result: dict[str, Any] | None,
    quality_config: dict[str, Any],
) -> dict[str, Any]:
    """Collect plain-language quality checks from every completed stage."""

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, message: str, score: float) -> None:
        """Append one consistently shaped, bounded quality-gate record."""

        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "score": round(_bounded(score), 6),
                "message": message,
            }
        )

    add(
        "document_quality",
        bool(document_quality and document_quality.get("passed")),
        "Document quality checks passed"
        if document_quality and document_quality.get("passed")
        else "Document has one or more quality warnings",
        float((document_quality or {}).get("quality_score", 0.0)),
    )
    detection_passed = bool(
        detection
        and detection.get("quality", {}).get("passed")
    )
    detection_confidence = float((detection or {}).get("confidence", 0.0))
    add(
        "accepted_detection",
        detection_passed,
        f"Accepted detection confidence is {detection_confidence:.3f}",
        detection_confidence if detection_passed else detection_confidence * 0.6,
    )
    segmentation_score = float(
        (segmentation_quality or {}).get("quality_score", 0.0)
    )
    add(
        "segmentation_quality",
        bool(
            segmentation_quality
            and segmentation_quality.get("passed")
            and segmentation_score
            >= float(quality_config["minimum_segmentation_score"])
        ),
        f"Segmentation quality score is {segmentation_score:.3f}",
        segmentation_score,
    )
    cleaning_score = float(
        (cleaning_quality or {}).get("quality_score", 0.0)
    )
    cleaning_fallback = bool(
        (cleaning_quality or {}).get("fallback_used")
    )
    add(
        "cleaning_quality",
        bool(
            cleaning_quality
            and cleaning_score
            >= float(quality_config["minimum_cleaning_score"])
            and not cleaning_fallback
        ),
        (
            "Conservative cleaning fallback preserved the original ink"
            if cleaning_fallback
            else f"Cleaning quality score is {cleaning_score:.3f}"
        ),
        cleaning_score * (0.80 if cleaning_fallback else 1.0),
    )
    reference_count = len(
        (verification_result or {}).get("reference_scores", [])
    )
    add(
        "verification_references",
        reference_count >= 3,
        f"{reference_count} reference comparisons are available",
        min(reference_count / 3.0, 1.0),
    )
    forensic_quality = (forensic_result or {}).get("quality", {})
    reference_total = int(forensic_quality.get("reference_count", 0))
    alignment_count = int(
        forensic_quality.get("alignment_success_count", 0)
    )
    alignment_score = (
        alignment_count / reference_total
        if reference_total
        else 0.0
    )
    add(
        "forensic_alignment",
        alignment_count > 0,
        f"{alignment_count} of {reference_total} references aligned",
        alignment_score,
    )

    missing_count = sum(
        item is None
        for item in (
            document_quality,
            detection,
            segmentation_quality,
            cleaning_quality,
            verification_result,
            forensic_result,
        )
    )
    missing_fraction = missing_count / 6
    scores = [check["score"] for check in checks]
    overall_score = float(np.mean(scores)) if scores else 0.0
    warnings = [
        check["message"]
        for check in checks
        if not check["passed"]
    ]
    if missing_fraction > float(
        quality_config["maximum_missing_stage_fraction"]
    ):
        warnings.append("Too many pipeline stages are missing")
    return {
        "passed": (
            overall_score
            >= float(quality_config["minimum_overall_score"])
            and missing_fraction
            <= float(quality_config["maximum_missing_stage_fraction"])
        ),
        "quality_score": round(overall_score, 6),
        "missing_stage_fraction": round(missing_fraction, 6),
        "checks": checks,
        "warnings": warnings,
    }


def calculate_forensic_risk(
    forensic_result: dict[str, Any],
) -> float:
    """Summarize structural differences without hiding their components."""

    comparisons = forensic_result.get("reference_comparisons", [])
    if not comparisons:
        return 1.0
    individual = []
    for comparison in comparisons:
        differences = comparison["structural_differences"]
        shape_difference = _bounded(
            float(differences["mean_relative_difference"])
        )
        slant_difference = _bounded(
            float(differences["slant_difference_degrees"]) / 45.0
        )
        xor_difference = _bounded(
            float(comparison["xor_difference_ratio"]) / 0.35
        )
        individual.append(
            0.50 * shape_difference
            + 0.20 * slant_difference
            + 0.30 * xor_difference
        )
    risk = float(np.median(individual))
    if forensic_result.get("line_overlap", {}).get(
        "possible_line_overlap"
    ):
        risk = min(1.0, risk + 0.10)
    return _bounded(risk)


def calculate_duplicate_risk(
    forensic_result: dict[str, Any],
) -> tuple[float, bool]:
    """Summarize multi-method reuse evidence across references."""

    risks = []
    confirmed_warning = False
    for comparison in forensic_result.get("reference_comparisons", []):
        evidence = comparison["duplicate_evidence"]
        if evidence["exact_normalized_duplicate"]:
            risks.append(1.0)
            confirmed_warning = True
            continue
        if evidence["possible_duplicate"]:
            risks.append(0.90)
            confirmed_warning = True
            continue
        hash_risk = _bounded(
            1.0 - float(evidence["perceptual_hash_distance"]) / 64.0
        )
        risks.append(
            0.30 * hash_risk
            + 0.30 * _bounded(float(evidence["orb_good_match_ratio"]))
            + 0.40 * _bounded(float(evidence["ransac_inlier_ratio"]))
        )
    return (
        _bounded(max(risks, default=0.0)),
        confirmed_warning,
    )


def fuse_signature_evidence(
    *,
    document_quality: dict[str, Any] | None,
    detection: dict[str, Any] | None,
    segmentation_quality: dict[str, Any] | None,
    cleaning_quality: dict[str, Any] | None,
    verification_result: dict[str, Any] | None,
    calibration: dict[str, Any] | None,
    forensic_result: dict[str, Any] | None,
    external_validation_report: dict[str, Any] | None,
    risk_config: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Fuse evidence with visible weights, gates, reasons, and warnings."""

    quality_result = validate_pipeline_quality(
        document_quality,
        detection,
        segmentation_quality,
        cleaning_quality,
        verification_result,
        forensic_result,
        risk_config["quality"],
    )
    verification_risk = (
        calculate_verification_risk(verification_result, calibration)
        if verification_result and calibration
        else 1.0
    )
    quality_risk = 1.0 - float(quality_result["quality_score"])
    forensic_risk = (
        calculate_forensic_risk(forensic_result)
        if forensic_result
        else 1.0
    )
    duplicate_risk, duplicate_warning = (
        calculate_duplicate_risk(forensic_result)
        if forensic_result
        else (1.0, False)
    )
    component_risks = {
        "verification": round(verification_risk, 6),
        "quality": round(quality_risk, 6),
        "forensic": round(forensic_risk, 6),
        "duplicate": round(duplicate_risk, 6),
    }

    reliability = calculate_verification_reliability(
        external_validation_report
    )
    configured_weights = {
        name: float(value)
        for name, value in risk_config["risk_weights"].items()
    }
    effective_weights = configured_weights.copy()
    effective_weights["verification"] *= float(reliability["score"])
    weight_total = sum(effective_weights.values())
    if weight_total <= 0:
        raise ValueError("Effective risk weights sum to zero")
    effective_weights = {
        name: value / weight_total
        for name, value in effective_weights.items()
    }
    overall_risk = sum(
        component_risks[name] * effective_weights[name]
        for name in component_risks
    )

    decisions = risk_config["risk_decisions"]
    reasons: list[str] = []
    warnings = list(quality_result["warnings"])
    warnings.append(reliability["reason"])
    if float(quality_result["quality_score"]) <= float(
        decisions["insufficient_quality_maximum_quality_score"]
    ):
        decision = "insufficient_quality"
        reasons.append("Pipeline quality is too low for evidence fusion")
    elif duplicate_warning and duplicate_risk >= float(
        decisions["possible_digital_reuse_minimum"]
    ):
        decision = "possible_digital_reuse"
        reasons.append(
            "Multiple duplicate-screening methods found possible normalized reuse"
        )
    elif (
        overall_risk
        <= float(decisions["provisionally_consistent_maximum"])
        and reliability["passed"]
        and quality_result["passed"]
    ):
        decision = "provisionally_consistent"
        reasons.append("Available indicators fall in the lower-risk region")
    elif overall_risk <= float(
        decisions["manual_review_maximum"]
    ):
        decision = "manual_review"
        reasons.append("Available indicators require human comparison")
    else:
        decision = "elevated_inconsistency"
        reasons.append("Several supporting indicators show elevated differences")

    if not reliability["passed"]:
        reasons.append(
            "The Siamese score cannot support a provisional-consistency "
            "outcome because untouched CEDAR generalization was weak"
        )
        if decision == "provisionally_consistent":
            decision = "manual_review"
    if verification_result:
        reasons.append(
            "Siamese status: "
            + str(verification_result.get("verification_status"))
        )
    return {
        "component_risks": component_risks,
        "configured_weights": configured_weights,
        "effective_weights": {
            name: round(value, 6)
            for name, value in effective_weights.items()
        },
        "verification_reliability": reliability,
        "quality_validation": quality_result,
        "overall_risk": round(_bounded(overall_risk), 6),
        "decision": decision,
        "reasons": reasons,
        "warnings": list(dict.fromkeys(warnings)),
        "disclaimer": (
            "This is a prototype review-priority result, not a finding of "
            "authenticity, forgery, or legal validity."
        ),
    }


def save_risk_result(
    result: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save the fully explainable fusion result."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
