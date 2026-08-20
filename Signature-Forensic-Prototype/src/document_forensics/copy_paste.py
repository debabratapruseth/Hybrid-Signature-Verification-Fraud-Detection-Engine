"""Locate a possible second placement of a signature crop on its source page."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .common import load_rgb_image, validate_bbox


def locate_signature_reuse(
    page_image: object,
    signature_crop: object,
    source_bbox_xyxy: list[int] | tuple[int, int, int, int],
    config: dict[str, Any],
) -> dict[str, object]:
    """Match crop features outside its accepted source box on the page."""

    page = load_rgb_image(page_image)
    crop = load_rgb_image(signature_crop)
    source_bbox = validate_bbox(source_bbox_xyxy, page.shape)
    page_gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    x1, y1, x2, y2 = source_bbox

    # Remove the known source placement before page feature extraction.
    # Otherwise the exact source is the nearest match and can suppress the
    # second placement during the nearest-neighbour ratio test.
    page_gray_without_source = page_gray.copy()
    page_gray_without_source[y1:y2, x1:x2] = 255

    orb = cv2.ORB_create(nfeatures=int(config["orb_feature_count"]))
    crop_keypoints, crop_descriptors = orb.detectAndCompute(crop_gray, None)
    page_keypoints, page_descriptors = orb.detectAndCompute(
        page_gray_without_source,
        None,
    )
    result: dict[str, object] = {
        "crop_keypoint_count": len(crop_keypoints),
        "page_keypoint_count": len(page_keypoints),
        "feature_method_status": (
            "available"
            if len(crop_keypoints)
            >= int(config["minimum_copy_paste_matches"])
            else "inconclusive_insufficient_keypoints"
        ),
        "outside_source_good_matches": 0,
        "ransac_inliers": 0,
        "ransac_inlier_ratio": 0.0,
        "template_match_score": 0.0,
        "template_scale": None,
        "template_phase_response": 0.0,
        "template_candidate_bbox_xyxy": None,
        "template_method_status": "not_run",
        "detection_method": None,
        "possible_second_location": False,
        "projected_polygon_xy": None,
        "source_bbox_xyxy": list(source_bbox),
        "visualization": page.copy(),
    }
    template_result = _multiscale_template_search(
        page_gray_without_source,
        crop_gray,
        source_bbox,
        config,
    )
    result.update(template_result)
    template_possible = (
        float(template_result["template_match_score"])
        >= float(config["template_match_score_warning"])
        and float(template_result["template_phase_response"])
        >= float(config["template_phase_response_warning"])
    )

    if crop_descriptors is None or page_descriptors is None:
        result["possible_second_location"] = template_possible
        result["detection_method"] = (
            "multiscale_template_and_phase"
            if template_possible
            else None
        )
        result["visualization"] = _draw_template_candidate(
            page,
            source_bbox,
            template_result,
            template_possible,
        )
        return result

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        crop_descriptors,
        page_descriptors,
        k=2,
    )
    good = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance >= float(config["orb_ratio_test"]) * second.distance:
            continue
        page_x, page_y = page_keypoints[first.trainIdx].pt
        if x1 <= page_x <= x2 and y1 <= page_y <= y2:
            continue
        good.append(first)
    result["outside_source_good_matches"] = len(good)
    visualization = page.copy()
    cv2.rectangle(
        visualization,
        (x1, y1),
        (x2, y2),
        (240, 150, 20),
        2,
    )
    if len(good) < max(
        4,
        int(config["minimum_copy_paste_matches"]),
    ):
        result["possible_second_location"] = template_possible
        result["detection_method"] = (
            "multiscale_template_and_phase"
            if template_possible
            else None
        )
        result["visualization"] = _draw_template_candidate(
            visualization,
            source_bbox,
            template_result,
            template_possible,
        )
        return result

    source_points = np.float32(
        [crop_keypoints[match.queryIdx].pt for match in good]
    ).reshape(-1, 1, 2)
    destination_points = np.float32(
        [page_keypoints[match.trainIdx].pt for match in good]
    ).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(
        source_points,
        destination_points,
        cv2.RANSAC,
        float(config["ransac_reprojection_threshold"]),
    )
    if homography is None or inlier_mask is None:
        result["possible_second_location"] = template_possible
        result["detection_method"] = (
            "multiscale_template_and_phase"
            if template_possible
            else None
        )
        result["visualization"] = _draw_template_candidate(
            visualization,
            source_bbox,
            template_result,
            template_possible,
        )
        return result

    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / len(good)
    crop_height, crop_width = crop.shape[:2]
    corners = np.float32(
        [[0, 0], [crop_width, 0], [crop_width, crop_height], [0, crop_height]]
    ).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    geometric_possible = (
        inliers >= int(config["minimum_copy_paste_inliers"])
        and inlier_ratio
        >= float(config["copy_paste_inlier_ratio_warning"])
    )
    possible = geometric_possible or template_possible
    colour = (220, 40, 40) if geometric_possible else (80, 80, 220)
    cv2.polylines(
        visualization,
        [np.round(projected).astype(np.int32)],
        True,
        colour,
        3,
    )
    result.update(
        {
            "ransac_inliers": inliers,
            "ransac_inlier_ratio": round(inlier_ratio, 6),
            "possible_second_location": possible,
            "projected_polygon_xy": projected.round(3).tolist(),
            "detection_method": (
                "orb_ransac"
                if geometric_possible
                else "multiscale_template_and_phase"
                if template_possible
                else None
            ),
            "visualization": visualization,
        }
    )
    return result


def _multiscale_template_search(
    page_gray: np.ndarray,
    crop_gray: np.ndarray,
    source_bbox: tuple[int, int, int, int],
    config: dict[str, Any],
) -> dict[str, object]:
    """Search edge structure at several scales and confirm the best patch."""

    page_edges = cv2.Canny(page_gray, 50, 150)
    crop_edges = cv2.Canny(crop_gray, 50, 150)
    best_score = -1.0
    best_scale: float | None = None
    best_bbox: list[int] | None = None
    minimum_scale = float(config["template_minimum_scale"])
    maximum_scale = float(config["template_maximum_scale"])
    scale_steps = int(config["template_scale_steps"])
    for scale in np.linspace(minimum_scale, maximum_scale, scale_steps):
        target_width = max(8, round(crop_gray.shape[1] * scale))
        target_height = max(8, round(crop_gray.shape[0] * scale))
        if (
            target_width >= page_gray.shape[1]
            or target_height >= page_gray.shape[0]
        ):
            continue
        resized_edges = cv2.resize(
            crop_edges,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        if not resized_edges.any():
            continue
        response = cv2.matchTemplate(
            page_edges,
            resized_edges,
            cv2.TM_CCOEFF_NORMED,
        )
        source_x1, source_y1, source_x2, source_y2 = source_bbox
        response_x1 = max(0, source_x1 - target_width + 1)
        response_y1 = max(0, source_y1 - target_height + 1)
        response_x2 = min(response.shape[1], source_x2)
        response_y2 = min(response.shape[0], source_y2)
        response[
            response_y1:response_y2,
            response_x1:response_x2,
        ] = -1.0
        _, maximum, _, location = cv2.minMaxLoc(response)
        if maximum > best_score:
            best_score = float(maximum)
            best_scale = float(scale)
            x, y = location
            best_bbox = [
                int(x),
                int(y),
                int(x + target_width),
                int(y + target_height),
            ]

    if best_bbox is None:
        return {
            "template_match_score": 0.0,
            "template_scale": None,
            "template_phase_response": 0.0,
            "template_candidate_bbox_xyxy": None,
            "template_method_status": "inconclusive_no_edge_template",
        }

    x1, y1, x2, y2 = best_bbox
    candidate = page_gray[y1:y2, x1:x2]
    candidate = cv2.resize(
        candidate,
        (crop_gray.shape[1], crop_gray.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    questioned = crop_gray.astype(np.float32)
    window = cv2.createHanningWindow(
        (crop_gray.shape[1], crop_gray.shape[0]),
        cv2.CV_32F,
    )
    _, phase_response = cv2.phaseCorrelate(
        questioned,
        candidate,
        window,
    )
    return {
        "template_match_score": round(best_score, 6),
        "template_scale": round(float(best_scale), 6),
        "template_phase_response": round(float(phase_response), 6),
        "template_candidate_bbox_xyxy": best_bbox,
        "template_method_status": "available",
    }


def _draw_template_candidate(
    page_rgb: np.ndarray,
    source_bbox: tuple[int, int, int, int],
    template_result: dict[str, object],
    warning: bool,
) -> np.ndarray:
    """Draw source and best low-resolution template candidate boxes."""

    visualization = page_rgb.copy()
    x1, y1, x2, y2 = source_bbox
    cv2.rectangle(
        visualization,
        (x1, y1),
        (x2, y2),
        (240, 150, 20),
        2,
    )
    candidate = template_result.get("template_candidate_bbox_xyxy")
    if candidate is not None:
        cx1, cy1, cx2, cy2 = [int(value) for value in candidate]
        colour = (220, 40, 40) if warning else (80, 80, 220)
        cv2.rectangle(
            visualization,
            (cx1, cy1),
            (cx2, cy2),
            colour,
            2,
        )
    return visualization
