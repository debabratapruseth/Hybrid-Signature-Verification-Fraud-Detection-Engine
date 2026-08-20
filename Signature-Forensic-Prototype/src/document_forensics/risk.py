"""Transparent Branch 3 document-forensics screening fusion."""

from __future__ import annotations

from typing import Any

import numpy as np


def calculate_document_forensics_risk(
    copy_paste_result: dict[str, Any],
    compression_result: dict[str, Any],
    noise_result: dict[str, Any],
    duplicate_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Combine screening indicators without claiming document manipulation."""

    if copy_paste_result["possible_second_location"]:
        copy_risk = 1.0
    else:
        geometric_partial = min(
            float(copy_paste_result["ransac_inliers"])
            / max(float(config["minimum_copy_paste_inliers"]), 1.0),
            1.0,
        ) * 0.5
        template_partial = min(
            float(copy_paste_result["template_match_score"])
            / max(float(config["template_match_score_warning"]), 1e-6),
            1.0,
        ) * 0.25
        copy_risk = max(geometric_partial, template_partial)
    ela_z = abs(float(compression_result["local_ela_z_score"]))
    compression_risk = float(
        np.clip(
            ela_z / max(float(config["ela_z_score_warning"]), 1e-6),
            0.0,
            1.0,
        )
    )
    noise_ratio = float(noise_result["residual_std_ratio"])
    noise_deviation = abs(np.log(max(noise_ratio, 1e-6)))
    noise_risk = float(
        np.clip(
            noise_deviation
            / max(float(config["noise_log_ratio_warning"]), 1e-6),
            0.0,
            1.0,
        )
    )
    duplicate_risk = max(
        (
            1.0
            if result["possible_duplicate"]
            else 0.5
            if result["near_hash_warning"]
            else 0.0
        )
        for result in duplicate_results
    ) if duplicate_results else 0.0

    component_risks = {
        "copy_paste": round(copy_risk, 6),
        "compression": round(compression_risk, 6),
        "noise": round(noise_risk, 6),
        "duplicate": round(duplicate_risk, 6),
    }
    weights = {
        name: float(value)
        for name, value in config["risk_weights"].items()
    }
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Branch 3 risk weights must sum to a positive value")
    weights = {name: value / total for name, value in weights.items()}
    overall = sum(component_risks[name] * weights[name] for name in weights)

    feature_matching_available = (
        copy_paste_result["feature_method_status"] == "available"
    )
    template_matching_available = (
        copy_paste_result["template_method_status"] == "available"
    )
    duplicate_geometric_available_count = sum(
        int(result["orb_comparison"]["first_keypoint_count"]) > 0
        and int(result["orb_comparison"]["second_keypoint_count"]) > 0
        for result in duplicate_results
    )
    evidence_availability = {
        "orb_crop_to_page": feature_matching_available,
        "multiscale_template": template_matching_available,
        "compression": True,
        "noise": True,
        "reference_hashing": bool(duplicate_results),
        "reference_orb_available_count": int(
            duplicate_geometric_available_count
        ),
        "reference_comparison_count": len(duplicate_results),
    }
    core_available = [
        template_matching_available or feature_matching_available,
        True,
        True,
        bool(duplicate_results),
    ]
    evidence_coverage = sum(core_available) / len(core_available)

    reasons = []
    warnings = [
        compression_result["limitation"],
        noise_result["limitation"],
    ]
    if copy_paste_result["possible_second_location"]:
        decision = "possible_copy_paste"
        if copy_paste_result.get("detection_method") == "orb_ransac":
            reasons.append(
                "Crop-to-page ORB/RANSAC found a possible second placement"
            )
        else:
            reasons.append(
                "Multiscale edge-template matching with phase confirmation "
                "found a possible second placement"
            )
    elif any(result["possible_duplicate"] for result in duplicate_results):
        decision = "possible_duplicate_reuse"
        reasons.append(
            "Hash and geometric checks jointly triggered a duplicate warning"
        )
    elif overall >= float(config["elevated_screening_risk"]):
        decision = "elevated_document_forensics_indicators"
        reasons.append("Several screening indicators require manual inspection")
    elif overall >= float(config["manual_review_risk"]):
        decision = "manual_review"
        reasons.append("At least one screening indicator merits review")
    else:
        if not feature_matching_available:
            decision = "no_elevated_indicators_limited_feature_evidence"
            reasons.append(
                "No configured screening threshold was exceeded, but ORB "
                "crop-to-page analysis lacked sufficient keypoints"
            )
        else:
            decision = "no_elevated_indicators"
            reasons.append(
                "No configured screening threshold was exceeded; editing is not ruled out"
            )
    if not feature_matching_available:
        warnings.append(
            "ORB crop-to-page evidence is inconclusive because the signature "
            "crop has too few stable keypoints; the template fallback was "
            "reported separately."
        )
    return {
        "component_risks": component_risks,
        "normalized_weights": {
            name: round(value, 6) for name, value in weights.items()
        },
        "overall_screening_risk": round(float(overall), 6),
        "evidence_availability": evidence_availability,
        "evidence_coverage": round(float(evidence_coverage), 6),
        "decision": decision,
        "reasons": reasons,
        "warnings": warnings,
        "disclaimer": (
            "Branch 3 reports document-forensics screening indicators. It does "
            "not prove or exclude editing, copying, authenticity, or intent."
        ),
    }
