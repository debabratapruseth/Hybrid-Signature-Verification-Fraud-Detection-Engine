"""ORB feature comparison with ratio testing and optional RANSAC."""

from __future__ import annotations

import cv2
import numpy as np

from .common import load_rgb_image


def compare_orb_features(
    first_image: object,
    second_image: object,
    *,
    feature_count: int = 1500,
    ratio_test: float = 0.75,
    ransac_threshold: float = 5.0,
) -> dict[str, object]:
    """Compare two images and retain geometrically consistent matches."""

    first_rgb = load_rgb_image(first_image)
    second_rgb = load_rgb_image(second_image)
    first_gray = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2GRAY)
    second_gray = cv2.cvtColor(second_rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=int(feature_count))
    first_keypoints, first_descriptors = orb.detectAndCompute(first_gray, None)
    second_keypoints, second_descriptors = orb.detectAndCompute(
        second_gray,
        None,
    )
    result: dict[str, object] = {
        "first_keypoint_count": len(first_keypoints),
        "second_keypoint_count": len(second_keypoints),
        "candidate_pair_count": 0,
        "good_match_count": 0,
        "good_match_ratio": 0.0,
        "ransac_inlier_count": 0,
        "ransac_inlier_ratio": 0.0,
        "homography_succeeded": False,
        "good_matches": [],
        "inlier_mask": [],
        "first_keypoints": first_keypoints,
        "second_keypoints": second_keypoints,
        "visualization": _blank_pair(first_rgb, second_rgb),
    }
    if first_descriptors is None or second_descriptors is None:
        return result

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        first_descriptors,
        second_descriptors,
        k=2,
    )
    valid_pairs = [pair for pair in pairs if len(pair) == 2]
    good = [
        first
        for first, second in valid_pairs
        if first.distance < float(ratio_test) * second.distance
    ]
    result["candidate_pair_count"] = len(valid_pairs)
    result["good_match_count"] = len(good)
    result["good_match_ratio"] = round(
        len(good) / max(len(valid_pairs), 1),
        6,
    )
    inlier_mask = np.zeros(len(good), dtype=np.uint8)
    if len(good) >= 4:
        source = np.float32(
            [first_keypoints[match.queryIdx].pt for match in good]
        ).reshape(-1, 1, 2)
        destination = np.float32(
            [second_keypoints[match.trainIdx].pt for match in good]
        ).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(
            source,
            destination,
            cv2.RANSAC,
            float(ransac_threshold),
        )
        if homography is not None and mask is not None:
            inlier_mask = mask.ravel().astype(np.uint8)
            result["homography_succeeded"] = True
            result["ransac_inlier_count"] = int(inlier_mask.sum())
            result["ransac_inlier_ratio"] = round(
                float(inlier_mask.mean()),
                6,
            )
    result["good_matches"] = good
    result["inlier_mask"] = inlier_mask.tolist()
    drawn_mask = inlier_mask.tolist() if len(inlier_mask) else None
    visualization_bgr = cv2.drawMatches(
        cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR),
        first_keypoints,
        cv2.cvtColor(second_rgb, cv2.COLOR_RGB2BGR),
        second_keypoints,
        good[:100],
        None,
        matchColor=(20, 190, 20),
        singlePointColor=(180, 180, 180),
        matchesMask=(
            drawn_mask[:100]
            if drawn_mask is not None
            else None
        ),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    result["visualization"] = cv2.cvtColor(
        visualization_bgr,
        cv2.COLOR_BGR2RGB,
    )
    return result


def public_orb_result(result: dict[str, object]) -> dict[str, object]:
    """Remove OpenCV objects and images for JSON serialization."""

    excluded = {
        "good_matches",
        "first_keypoints",
        "second_keypoints",
        "visualization",
    }
    return {
        key: value for key, value in result.items() if key not in excluded
    }


def _blank_pair(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Create a side-by-side fallback visualization."""

    height = max(first.shape[0], second.shape[0])
    first_canvas = np.full((height, first.shape[1], 3), 255, dtype=np.uint8)
    second_canvas = np.full((height, second.shape[1], 3), 255, dtype=np.uint8)
    first_canvas[: first.shape[0], : first.shape[1]] = first
    second_canvas[: second.shape[0], : second.shape[1]] = second
    return np.hstack([first_canvas, second_canvas])
