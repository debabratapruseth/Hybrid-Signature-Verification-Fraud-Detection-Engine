"""Multi-method duplicate screening across questioned and reference images."""

from __future__ import annotations

from typing import Any

from .orb_match import compare_orb_features, public_orb_result
from .perceptual_hash import calculate_image_hashes, compare_hashes


def compare_possible_duplicate(
    questioned_image: object,
    comparison_image: object,
    config: dict[str, Any],
) -> dict[str, object]:
    """Combine hashes, ORB matches, and RANSAC into an explainable warning."""

    questioned_hashes = calculate_image_hashes(
        questioned_image,
        hash_size=int(config["perceptual_hash_size"]),
    )
    comparison_hashes = calculate_image_hashes(
        comparison_image,
        hash_size=int(config["perceptual_hash_size"]),
    )
    hash_result = compare_hashes(questioned_hashes, comparison_hashes)
    orb_result = compare_orb_features(
        questioned_image,
        comparison_image,
        feature_count=int(config["orb_feature_count"]),
        ratio_test=float(config["orb_ratio_test"]),
        ransac_threshold=float(config["ransac_reprojection_threshold"]),
    )
    exact = bool(hash_result["exact_normalized_pixel_match"])
    near_hash = (
        int(hash_result["perceptual_distances"]["phash"])
        <= int(config["perceptual_hash_distance_warning"])
    )
    geometric = (
        int(orb_result["good_match_count"])
        >= int(config["minimum_good_matches"])
        and float(orb_result["good_match_ratio"])
        >= float(config["good_match_ratio_warning"])
        and int(orb_result["ransac_inlier_count"])
        >= int(config["minimum_ransac_inliers"])
        and float(orb_result["ransac_inlier_ratio"])
        >= float(config["ransac_inlier_ratio_warning"])
    )
    possible_duplicate = exact or (near_hash and geometric)
    return {
        "hash_comparison": hash_result,
        "orb_comparison": public_orb_result(orb_result),
        "exact_duplicate": exact,
        "near_hash_warning": near_hash,
        "geometric_consistency_warning": geometric,
        "possible_duplicate": possible_duplicate,
        "interpretation": (
            "Possible normalized reuse; inspect original evidence"
            if possible_duplicate
            else "No configured multi-method duplicate warning; reuse is not ruled out"
        ),
        "visualization": orb_result["visualization"],
    }
