"""Transparent deterministic fusion across the three analysis branches."""

from __future__ import annotations

from typing import Any

import numpy as np


def calculate_branch_fusion(
    evidence: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Combine branch-level screening risks with visible reliability factors.

    This is the authoritative final numerical calculation. The later OpenAI
    step receives its result as locked evidence and is never asked to choose
    weights, thresholds, or a replacement conclusion.

    The configured weights express intended importance. Reliability factors
    express how much usable evidence was available in this run. Multiplying and
    renormalizing them prevents a weak or unavailable method from retaining its
    full configured influence.
    """

    branch1 = evidence["branch1"]
    branch2 = evidence["branch2"]
    branch3 = evidence["branch3"]
    branch1_risk = float(branch1["overall_risk"])
    branch2_risk = _branch2_relative_risk(
        branch2["reference_variation"]
    )
    branch3_risk = float(
        branch3["risk"]["overall_screening_risk"]
    )

    branch1_quality = float(
        branch1.get("quality_validation", {}).get(
            "quality_score",
            0.50,
        )
    )
    branch2_pair_count = int(
        branch2["reference_variation"].get(
            "reference_pair_count",
            0,
        )
    )
    branch2_reliability = min(
        float(config["branch2_maximum_reliability"]),
        branch2_pair_count
        / max(float(config["branch2_pairs_for_full_reliability"]), 1.0),
    )
    branch3_coverage = float(
        branch3["risk"].get("evidence_coverage", 0.50)
    )
    if not branch3["risk"].get(
        "evidence_availability",
        {},
    ).get("orb_crop_to_page", False):
        branch3_coverage *= float(
            config["branch3_limited_feature_factor"]
        )

    reliabilities = {
        "branch1": float(np.clip(branch1_quality, 0.0, 1.0)),
        "branch2": float(np.clip(branch2_reliability, 0.0, 1.0)),
        "branch3": float(np.clip(branch3_coverage, 0.0, 1.0)),
    }
    configured_weights = {
        name: float(value)
        for name, value in config["branch_weights"].items()
    }
    # Reliability adjustment happens before normalization. The effective
    # weights therefore still sum to one and remain directly inspectable.
    effective_unnormalized = {
        name: configured_weights[name] * reliabilities[name]
        for name in configured_weights
    }
    total = sum(effective_unnormalized.values())
    if total <= 0:
        raise ValueError("All effective branch weights are zero")
    effective_weights = {
        name: value / total
        for name, value in effective_unnormalized.items()
    }
    branch_risks = {
        "branch1": branch1_risk,
        "branch2": branch2_risk,
        "branch3": branch3_risk,
    }
    fused_risk = sum(
        branch_risks[name] * effective_weights[name]
        for name in branch_risks
    )

    reasons = [
        f"Branch 1 decision: {branch1['decision']}",
        "Branch 2 relative status: "
        + str(branch2["reference_variation"]["relative_status"]),
        "Branch 3 decision: "
        + str(branch3["risk"]["decision"]),
    ]
    # A low numerical risk is not enough for a lower-priority result when the
    # pipeline lacks sufficient evidence to support that interpretation.
    quality_limited = (
        not bool(
            branch1.get("quality_validation", {}).get(
                "passed",
                False,
            )
        )
        or reliabilities["branch2"]
        < float(config["minimum_reliability_for_lower_priority"])
        or not branch3["risk"].get(
            "evidence_availability",
            {},
        ).get("orb_crop_to_page", False)
    )
    branch3_decision = str(branch3["risk"]["decision"])
    if branch3_decision in {
        "possible_copy_paste",
        "possible_duplicate_reuse",
    }:
        conclusion = "possible_digital_reuse_review"
        reasons.append(
            "Branch 3 triggered a multi-method digital-reuse warning"
        )
    elif fused_risk > float(config["elevated_review_boundary"]):
        conclusion = "elevated_inconsistency_review"
    elif (
        fused_risk <= float(config["lower_review_boundary"])
        and not quality_limited
    ):
        conclusion = "lower_review_priority"
    else:
        conclusion = "manual_review"
        if quality_limited:
            reasons.append(
                "Lower-priority output was blocked by evidence-quality or "
                "validation limitations"
            )

    agreements = _branch_agreement_summary(
        branch1,
        branch2,
        branch3,
    )
    return {
        "case_id": evidence.get("run_id", "unknown"),
        "fusion_type": "deterministic_three_branch_review_fusion",
        "branch_risks": {
            name: round(value, 6)
            for name, value in branch_risks.items()
        },
        "configured_weights": configured_weights,
        "branch_reliabilities": {
            name: round(value, 6)
            for name, value in reliabilities.items()
        },
        "effective_weights": {
            name: round(value, 6)
            for name, value in effective_weights.items()
        },
        "fused_review_risk": round(float(fused_risk), 6),
        "conclusion": conclusion,
        "quality_limited": quality_limited,
        "cross_branch_agreement": agreements,
        "reasons": reasons,
        "disclaimer": (
            "The fused result is a review-priority synthesis, not a probability "
            "or finding of authenticity, forgery, editing, intent, or legal validity."
        ),
    }


def _branch2_relative_risk(
    variation: dict[str, Any],
) -> float:
    """Map reference-relative geometry to risk without a universal threshold."""

    questioned = float(variation["questioned_median"])
    boundary_value = variation.get("reference_similarity_10th_percentile")
    if boundary_value is None:
        return 0.50
    boundary = float(boundary_value)
    if questioned < boundary:
        deficit = (boundary - questioned) / max(boundary, 1e-6)
        return float(np.clip(0.50 + 0.50 * deficit, 0.0, 1.0))
    margin = (questioned - boundary) / max(1.0 - boundary, 1e-6)
    return float(np.clip(0.50 * (1.0 - margin), 0.0, 1.0))


def _branch_agreement_summary(
    branch1: dict[str, Any],
    branch2: dict[str, Any],
    branch3: dict[str, Any],
) -> dict[str, Any]:
    """Describe agreement and disagreement without LLM judgment."""

    branch1_elevated = branch1["decision"] in {
        "manual_review",
        "elevated_inconsistency",
        "possible_digital_reuse",
    }
    branch2_elevated = (
        branch2["reference_variation"]["relative_status"]
        == "below_observed_reference_variation"
    )
    branch3_elevated = branch3["risk"]["decision"] not in {
        "no_elevated_indicators",
        "no_elevated_indicators_limited_feature_evidence",
    }
    elevated_count = sum(
        [branch1_elevated, branch2_elevated, branch3_elevated]
    )
    return {
        "branch1_review_indicator": branch1_elevated,
        "branch2_review_indicator": branch2_elevated,
        "branch3_review_indicator": branch3_elevated,
        "review_indicator_count": elevated_count,
        "summary": (
            "Branches 1 and 2 both indicate comparative inconsistency while "
            "Branch 3 did not trigger a digital-reuse warning."
            if branch1_elevated and branch2_elevated and not branch3_elevated
            else f"{elevated_count} of 3 branches raised a review indicator."
        ),
    }
