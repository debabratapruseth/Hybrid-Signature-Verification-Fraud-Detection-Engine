"""Explainable structural and duplicate-reuse indicators for signatures.

The measurements in this module are supporting observations. They do not
independently establish authenticity, forgery, or digital copying.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import imagehash
import numpy as np
import yaml
from PIL import Image
from skimage.morphology import skeletonize


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_forensics_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load structural-forensics and duplicate-detection settings."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config.get("forensics"), dict):
        raise ValueError("config.yaml needs a 'forensics' section")
    if not isinstance(config.get("duplicate_detection"), dict):
        raise ValueError("config.yaml needs a 'duplicate_detection' section")
    return config["forensics"], config["duplicate_detection"]


def load_signature_image(
    signature: np.ndarray | Image.Image | str | Path,
) -> np.ndarray:
    """Load a signature as an RGB uint8 array without changing the source."""

    if isinstance(signature, np.ndarray):
        image = signature.copy()
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("Signature array must be grayscale, RGB, or RGBA")
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image.astype(np.uint8)
    if isinstance(signature, Image.Image):
        return np.asarray(signature.convert("RGB")).copy()

    path = Path(signature).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Signature image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported signature format: {path.suffix}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable signature image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_signature_ink(
    signature: np.ndarray | Image.Image | str | Path,
    canvas_width: int = 512,
    canvas_height: int = 256,
    padding: int = 16,
) -> np.ndarray:
    """Convert dark ink to a comparable boolean mask on a fixed canvas."""

    rgb = load_signature_image(signature)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    ink = threshold > 0
    if float(ink.mean()) > 0.50:
        ink = ~ink
    coordinates = np.argwhere(ink)
    if coordinates.size == 0:
        raise ValueError("No foreground ink was detected")

    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0) + 1
    cropped = ink[y_min:y_max, x_min:x_max]
    available_width = canvas_width - 2 * padding
    available_height = canvas_height - 2 * padding
    scale = min(
        available_width / cropped.shape[1],
        available_height / cropped.shape[0],
    )
    resized_width = max(1, round(cropped.shape[1] * scale))
    resized_height = max(1, round(cropped.shape[0] * scale))
    resized = cv2.resize(
        cropped.astype(np.uint8),
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    canvas = np.zeros((canvas_height, canvas_width), dtype=bool)
    x_start = (canvas_width - resized_width) // 2
    y_start = (canvas_height - resized_height) // 2
    canvas[
        y_start : y_start + resized_height,
        x_start : x_start + resized_width,
    ] = resized
    return canvas


def extract_structural_features(ink: np.ndarray) -> dict[str, Any]:
    """Measure interpretable geometry from one normalized ink mask."""

    mask = np.asarray(ink, dtype=bool)
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("Cannot measure an empty signature mask")
    height, width = mask.shape
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0) + 1
    box_width = int(x_max - x_min)
    box_height = int(y_max - y_min)
    pixel_count = int(mask.sum())

    binary = mask.astype(np.uint8)
    component_count, _ = cv2.connectedComponents(binary, connectivity=8)
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    skeleton = skeletonize(mask)
    neighbour_count = cv2.filter2D(
        skeleton.astype(np.uint8),
        cv2.CV_16U,
        np.ones((3, 3), dtype=np.uint8),
    ) - skeleton.astype(np.uint16)
    endpoint_count = int(np.sum(skeleton & (neighbour_count == 1)))
    branch_count = int(np.sum(skeleton & (neighbour_count >= 3)))

    xy = np.column_stack((coordinates[:, 1], coordinates[:, 0])).astype(float)
    centred = xy - xy.mean(axis=0)
    covariance = np.cov(centred, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    slant_degrees = float(
        np.degrees(np.arctan2(major_axis[1], major_axis[0]))
    )
    if slant_degrees > 90:
        slant_degrees -= 180
    if slant_degrees < -90:
        slant_degrees += 180

    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    signed_log_hu = [
        float(-np.sign(value) * np.log10(abs(value) + 1e-30))
        for value in hu
    ]
    return {
        "aspect_ratio": round(box_width / max(box_height, 1), 6),
        "foreground_ratio": round(pixel_count / mask.size, 6),
        "connected_components": int(component_count - 1),
        "contour_count": len(contours),
        "bounding_box_occupancy": round(
            pixel_count / max(box_width * box_height, 1),
            6,
        ),
        "skeleton_length": int(skeleton.sum()),
        "skeleton_endpoints": endpoint_count,
        "skeleton_branch_points": branch_count,
        "slant_degrees": round(slant_degrees, 6),
        "centroid": [
            round(float(xy[:, 0].mean() / width), 6),
            round(float(xy[:, 1].mean() / height), 6),
        ],
        "hu_moments_log": [round(value, 6) for value in signed_log_hu],
    }


def compare_structural_features(
    questioned: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Return transparent normalized differences between two feature sets."""

    def relative_difference(name: str) -> float:
        """Scale an absolute difference by the larger observed magnitude."""

        first = float(questioned[name])
        second = float(reference[name])
        return abs(first - second) / max(abs(first), abs(second), 1e-6)

    fields = [
        "aspect_ratio",
        "foreground_ratio",
        "connected_components",
        "contour_count",
        "bounding_box_occupancy",
        "skeleton_length",
        "skeleton_endpoints",
        "skeleton_branch_points",
    ]
    differences = {
        field: round(relative_difference(field), 6)
        for field in fields
    }
    differences["slant_difference_degrees"] = round(
        abs(
            float(questioned["slant_degrees"])
            - float(reference["slant_degrees"])
        ),
        6,
    )
    differences["centroid_distance"] = round(
        float(
            np.linalg.norm(
                np.asarray(questioned["centroid"])
                - np.asarray(reference["centroid"])
            )
        ),
        6,
    )
    differences["mean_relative_difference"] = round(
        float(np.mean([differences[field] for field in fields])),
        6,
    )
    return differences


def align_and_compare(
    questioned_ink: np.ndarray,
    reference_ink: np.ndarray,
    forensics_config: dict[str, Any],
) -> dict[str, Any]:
    """Align with ORB/RANSAC and create XOR, overlay, and match visuals."""

    questioned = (questioned_ink.astype(np.uint8) * 255)
    reference = (reference_ink.astype(np.uint8) * 255)
    orb = cv2.ORB_create(
        nfeatures=int(forensics_config["orb_feature_count"])
    )
    questioned_keypoints, questioned_descriptors = orb.detectAndCompute(
        questioned,
        None,
    )
    reference_keypoints, reference_descriptors = orb.detectAndCompute(
        reference,
        None,
    )
    result: dict[str, Any] = {
        "questioned_keypoints": len(questioned_keypoints),
        "reference_keypoints": len(reference_keypoints),
        "good_matches": 0,
        "good_match_ratio": 0.0,
        "ransac_inliers": 0,
        "ransac_inlier_ratio": 0.0,
        "alignment_succeeded": False,
    }
    empty_visual = cv2.cvtColor(255 - questioned, cv2.COLOR_GRAY2RGB)
    result.update(
        {
            "aligned_reference_ink": reference_ink.copy(),
            "xor_map": np.logical_xor(questioned_ink, reference_ink),
            "overlay": _make_overlay(questioned_ink, reference_ink),
            "match_visualization": empty_visual,
        }
    )
    if questioned_descriptors is None or reference_descriptors is None:
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(
        questioned_descriptors,
        reference_descriptors,
        k=2,
    )
    valid_pairs = [pair for pair in raw_matches if len(pair) == 2]
    ratio_limit = float(forensics_config["orb_ratio_test"])
    good_matches = [
        first
        for first, second in valid_pairs
        if first.distance < ratio_limit * second.distance
    ]
    result["good_matches"] = len(good_matches)
    result["good_match_ratio"] = round(
        len(good_matches) / max(len(valid_pairs), 1),
        6,
    )
    result["match_visualization"] = cv2.cvtColor(
        cv2.drawMatches(
            questioned,
            questioned_keypoints,
            reference,
            reference_keypoints,
            good_matches[:80],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        ),
        cv2.COLOR_BGR2RGB,
    )
    minimum_matches = int(
        forensics_config["minimum_alignment_matches"]
    )
    if len(good_matches) < max(4, minimum_matches):
        return result

    source_points = np.float32(
        [reference_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    destination_points = np.float32(
        [questioned_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(
        source_points,
        destination_points,
        cv2.RANSAC,
        float(forensics_config["ransac_reprojection_threshold"]),
    )
    if homography is None or inlier_mask is None:
        return result

    aligned = cv2.warpPerspective(
        reference,
        homography,
        (questioned.shape[1], questioned.shape[0]),
        flags=cv2.INTER_NEAREST,
    ) > 0
    inliers = int(inlier_mask.sum())
    result.update(
        {
            "ransac_inliers": inliers,
            "ransac_inlier_ratio": round(
                inliers / len(good_matches),
                6,
            ),
            "alignment_succeeded": True,
            "aligned_reference_ink": aligned,
            "xor_map": np.logical_xor(questioned_ink, aligned),
            "overlay": _make_overlay(questioned_ink, aligned),
        }
    )
    return result


def assess_duplicate_evidence(
    questioned_ink: np.ndarray,
    reference_ink: np.ndarray,
    alignment: dict[str, Any],
    duplicate_config: dict[str, Any],
) -> dict[str, Any]:
    """Combine normalized hash, perceptual hash, ORB, and RANSAC evidence."""

    questioned_bytes = np.packbits(questioned_ink).tobytes()
    reference_bytes = np.packbits(reference_ink).tobytes()
    questioned_sha = hashlib.sha256(questioned_bytes).hexdigest()
    reference_sha = hashlib.sha256(reference_bytes).hexdigest()
    exact_duplicate = questioned_sha == reference_sha

    hash_size = int(duplicate_config["perceptual_hash_size"])
    questioned_pil = Image.fromarray(
        255 - questioned_ink.astype(np.uint8) * 255
    )
    reference_pil = Image.fromarray(
        255 - reference_ink.astype(np.uint8) * 255
    )
    hash_distance = int(
        imagehash.phash(questioned_pil, hash_size=hash_size)
        - imagehash.phash(reference_pil, hash_size=hash_size)
    )
    geometric_duplicate = (
        hash_distance
        <= int(duplicate_config["perceptual_hash_distance_warning"])
        and int(alignment["good_matches"])
        >= int(duplicate_config["minimum_good_matches"])
        and float(alignment["good_match_ratio"])
        >= float(duplicate_config["good_match_ratio_warning"])
        and int(alignment["ransac_inliers"])
        >= int(duplicate_config["minimum_ransac_inliers"])
        and float(alignment["ransac_inlier_ratio"])
        >= float(duplicate_config["ransac_inlier_ratio_warning"])
    )
    return {
        "normalized_sha256_questioned": questioned_sha,
        "normalized_sha256_reference": reference_sha,
        "exact_normalized_duplicate": exact_duplicate,
        "perceptual_hash_distance": hash_distance,
        "orb_good_matches": int(alignment["good_matches"]),
        "orb_good_match_ratio": float(alignment["good_match_ratio"]),
        "ransac_inliers": int(alignment["ransac_inliers"]),
        "ransac_inlier_ratio": float(alignment["ransac_inlier_ratio"]),
        "possible_duplicate": bool(exact_duplicate or geometric_duplicate),
        "warning": (
            "Possible normalized digital reuse; manual evidence review required"
            if exact_duplicate or geometric_duplicate
            else None
        ),
    }


def analyze_signature_forensics(
    questioned_signature: np.ndarray | Image.Image | str | Path,
    reference_signatures: list[np.ndarray | Image.Image | str | Path],
    forensics_config: dict[str, Any],
    duplicate_config: dict[str, Any],
) -> dict[str, Any]:
    """Analyze one questioned signature against three or more references."""

    if len(reference_signatures) < 3:
        raise ValueError("At least three genuine references are required")
    questioned_ink = normalize_signature_ink(questioned_signature)
    questioned_features = extract_structural_features(questioned_ink)
    comparisons = []
    visuals = []
    for index, reference in enumerate(reference_signatures, start=1):
        reference_ink = normalize_signature_ink(reference)
        reference_features = extract_structural_features(reference_ink)
        alignment = align_and_compare(
            questioned_ink,
            reference_ink,
            forensics_config,
        )
        duplicate = assess_duplicate_evidence(
            questioned_ink,
            reference_ink,
            alignment,
            duplicate_config,
        )
        xor_ratio = float(np.mean(alignment["xor_map"]))
        comparisons.append(
            {
                "reference_index": index,
                "reference_features": reference_features,
                "structural_differences": compare_structural_features(
                    questioned_features,
                    reference_features,
                ),
                "alignment": {
                    key: value
                    for key, value in alignment.items()
                    if not isinstance(value, np.ndarray)
                },
                "xor_difference_ratio": round(xor_ratio, 6),
                "duplicate_evidence": duplicate,
            }
        )
        visuals.append(
            {
                "reference_index": index,
                "overlay": alignment["overlay"],
                "xor_map": (
                    alignment["xor_map"].astype(np.uint8) * 255
                ),
                "orb_matches": alignment["match_visualization"],
            }
        )

    possible_duplicates = [
        item["reference_index"]
        for item in comparisons
        if item["duplicate_evidence"]["possible_duplicate"]
    ]
    alignment_successes = sum(
        item["alignment"]["alignment_succeeded"]
        for item in comparisons
    )
    foreground_ratio = float(questioned_features["foreground_ratio"])
    warnings = []
    if not (
        float(forensics_config["minimum_foreground_ratio"])
        <= foreground_ratio
        <= float(forensics_config["maximum_foreground_ratio"])
    ):
        warnings.append("Questioned foreground ratio is outside configured limits")
    if alignment_successes == 0:
        warnings.append("No reference could be aligned reliably")
    if possible_duplicates:
        warnings.append(
            "Possible normalized reuse with reference(s): "
            + ", ".join(map(str, possible_duplicates))
        )
    return {
        "questioned_features": questioned_features,
        "reference_comparisons": comparisons,
        "possible_duplicate_reference_indices": possible_duplicates,
        "quality": {
            "passed": not warnings,
            "alignment_success_count": alignment_successes,
            "reference_count": len(reference_signatures),
            "warnings": warnings,
        },
        "visuals": visuals,
    }


def _make_overlay(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Render first-only red, second-only blue, and overlap dark."""

    overlay = np.full((*first.shape, 3), 255, dtype=np.uint8)
    only_first = first & ~second
    only_second = second & ~first
    shared = first & second
    overlay[only_first] = [220, 30, 30]
    overlay[only_second] = [30, 80, 220]
    overlay[shared] = [30, 30, 30]
    return overlay
