"""Orchestration for Branch 2 explainable structural comparison."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .common import create_normalized_ink_mask, json_safe, mask_to_rgb
from .contours import (
    contour_measurements,
    extract_contours,
    resample_closed_contour,
)
from .critical_points import combine_critical_points, detect_curvature_extrema
from .curvature import compute_contour_curvature, summarize_curvature
from .fourier_descriptors import compute_fourier_descriptor, fourier_distance
from .graph import build_skeleton_graph, graph_distance
from .hu_moments import compute_hu_moments, hu_moment_distance
from .shape_context import (
    compute_shape_context,
    sample_shape_points,
    shape_context_distance,
)
from .similarity import (
    combine_metric_distances,
    compare_to_reference_variation,
    contour_shape_distance,
    critical_point_distance,
)
from .skeleton import extract_skeleton, find_skeleton_points, skeleton_length
from .visualization import (
    create_geometry_overlay,
    draw_contour_critical_points,
    draw_skeleton_points,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_geometry_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load Branch 2 settings, falling back to documented defaults."""

    defaults = {
        "canvas_width": 512,
        "canvas_height": 256,
        "canvas_padding": 16,
        "contour_sample_points": 256,
        "fourier_coefficients": 32,
        "shape_context_points": 100,
        "curvature_neighbourhood": 4,
        "overlay_tolerance_pixels": 3.0,
        "metric_weights": {
            "hu": 0.15,
            "fourier": 0.20,
            "shape_context": 0.25,
            "contour": 0.15,
            "graph": 0.15,
            "critical_points": 0.10,
        },
    }
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        return defaults
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a dictionary")
    geometry = config.get("geometry")
    if geometry is None:
        return defaults
    if not isinstance(geometry, dict):
        raise ValueError("The geometry configuration must be a dictionary")
    merged = dict(defaults)
    merged.update(geometry)
    if isinstance(geometry.get("metric_weights"), dict):
        merged["metric_weights"] = geometry["metric_weights"]
    return merged


def extract_geometry_descriptor(
    signature: Any,
    geometry_config: dict[str, Any],
) -> dict[str, Any]:
    """Extract all Branch 2 representations for one signature."""

    ink = create_normalized_ink_mask(
        signature,
        canvas_width=int(geometry_config["canvas_width"]),
        canvas_height=int(geometry_config["canvas_height"]),
        padding=int(geometry_config["canvas_padding"]),
    )
    skeleton = extract_skeleton(ink)
    skeleton_points = find_skeleton_points(skeleton)
    contours = extract_contours(ink)
    if not contours:
        raise ValueError("No usable signature contour was extracted")
    main_contour = resample_closed_contour(
        contours[0],
        point_count=int(geometry_config["contour_sample_points"]),
    )
    curvature = compute_contour_curvature(
        main_contour,
        neighbourhood=int(geometry_config["curvature_neighbourhood"]),
    )
    curvature_points = detect_curvature_extrema(
        main_contour,
        curvature,
    )
    critical_points = combine_critical_points(
        skeleton_points["endpoint_yx"],
        skeleton_points["junction_yx"],
        curvature_points,
    )
    shape_points = sample_shape_points(
        skeleton,
        point_count=int(geometry_config["shape_context_points"]),
    )
    descriptor = {
        "ink_mask": ink,
        "skeleton": skeleton,
        "skeleton_points": skeleton_points,
        "skeleton_measurements": skeleton_length(skeleton),
        "contours": contours,
        "contour_measurements": contour_measurements(contours),
        "main_contour": main_contour,
        "curvature": curvature,
        "curvature_summary": summarize_curvature(curvature),
        "critical_points": critical_points,
        "hu_moments": compute_hu_moments(ink),
        "fourier_descriptor": compute_fourier_descriptor(
            main_contour,
            coefficient_count=int(geometry_config["fourier_coefficients"]),
        ),
        "shape_context": compute_shape_context(shape_points),
        "skeleton_graph": build_skeleton_graph(skeleton),
    }
    descriptor["visuals"] = {
        "normalized_signature": mask_to_rgb(ink),
        "skeleton_points": draw_skeleton_points(
            ink,
            skeleton,
            skeleton_points["endpoint_yx"],
            skeleton_points["junction_yx"],
        ),
        "contour_critical_points": draw_contour_critical_points(
            ink,
            main_contour,
            curvature_points,
        ),
    }
    return descriptor


def compare_geometry_descriptors(
    first: dict[str, Any],
    second: dict[str, Any],
    geometry_config: dict[str, Any],
) -> dict[str, Any]:
    """Calculate named descriptor distances and a transparent combined score."""

    distances = {
        "hu": hu_moment_distance(
            first["hu_moments"]["signed_log"],
            second["hu_moments"]["signed_log"],
        ),
        "fourier": fourier_distance(
            first["fourier_descriptor"],
            second["fourier_descriptor"],
        ),
        "shape_context": shape_context_distance(
            first["shape_context"],
            second["shape_context"],
        ),
        "contour": contour_shape_distance(
            first["main_contour"],
            second["main_contour"],
        ),
        "graph": graph_distance(
            first["skeleton_graph"],
            second["skeleton_graph"],
        ),
        "critical_points": critical_point_distance(
            first["critical_points"],
            second["critical_points"],
        ),
    }
    return combine_metric_distances(
        distances,
        weights=geometry_config["metric_weights"],
    )


def run_structural_ai_branch(
    questioned_signature: Any,
    reference_signatures: list[Any],
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Run Branch 2 using a Branch 1 cleaned crop or any image-like input."""

    if len(reference_signatures) < 3:
        raise ValueError("Branch 2 needs at least three reference signatures")
    config = load_geometry_config(config_path)
    questioned = extract_geometry_descriptor(questioned_signature, config)
    references = [
        extract_geometry_descriptor(reference, config)
        for reference in reference_signatures
    ]
    comparisons = []
    comparison_visuals = []
    for index, reference in enumerate(references, start=1):
        score = compare_geometry_descriptors(questioned, reference, config)
        visual = create_geometry_overlay(
            questioned["ink_mask"],
            reference["ink_mask"],
            tolerance_pixels=float(config["overlay_tolerance_pixels"]),
        )
        comparisons.append(
            {
                "reference_index": index,
                **score,
            }
        )
        comparison_visuals.append(
            {
                "reference_index": index,
                **visual,
            }
        )

    reference_pair_scores = []
    for first, second in itertools.combinations(references, 2):
        pair_result = compare_geometry_descriptors(first, second, config)
        reference_pair_scores.append(
            pair_result["combined_shape_similarity"]
        )
    questioned_scores = [
        comparison["combined_shape_similarity"]
        for comparison in comparisons
    ]
    public_questioned = _public_descriptor(questioned)
    public_references = [
        _public_descriptor(reference) for reference in references
    ]
    return {
        "branch": "Branch 2 - Structural AI",
        "questioned_geometry": public_questioned,
        "reference_geometries": public_references,
        "comparisons": comparisons,
        "reference_variation": compare_to_reference_variation(
            questioned_scores,
            reference_pair_scores,
        ),
        "interpretation": (
            "Geometry similarity is supporting evidence only. The relative "
            "status compares this questioned sample with variation observed "
            "among the supplied references; it is not an authenticity verdict."
        ),
        "visuals": {
            "questioned": questioned["visuals"],
            "comparisons": comparison_visuals,
        },
    }


def save_structural_ai_result(
    result: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, str]:
    """Save JSON measurements and all Branch 2 visual explanations."""

    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "branch2_geometry_result.json"
    serializable = {
        key: value for key, value in result.items() if key != "visuals"
    }
    report_path.write_text(
        json.dumps(json_safe(serializable), indent=2),
        encoding="utf-8",
    )
    questioned_visuals = result["visuals"]["questioned"]
    saved = {"report": str(report_path)}
    for name, rgb in questioned_visuals.items():
        path = directory / f"questioned_{name}.png"
        _save_rgb(path, rgb)
        saved[f"questioned_{name}"] = str(path)
    for visual in result["visuals"]["comparisons"]:
        index = int(visual["reference_index"])
        for name in ("overlay", "mismatch_heatmap"):
            path = directory / f"reference_{index:02d}_{name}.png"
            _save_rgb(path, visual[name])
            saved[f"reference_{index:02d}_{name}"] = str(path)
    return saved


def _public_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    """Remove large working arrays while retaining interpretable features."""

    skeleton_points = descriptor["skeleton_points"]
    critical_points = descriptor["critical_points"]
    return json_safe(
        {
            "skeleton": {
                **descriptor["skeleton_measurements"],
                "endpoint_count": skeleton_points["endpoint_count"],
                "junction_count": skeleton_points["junction_count"],
                "raw_branch_pixel_count": skeleton_points[
                    "raw_branch_pixel_count"
                ],
            },
            "contours": descriptor["contour_measurements"],
            "curvature": descriptor["curvature_summary"],
            "critical_points": {
                key: value
                for key, value in critical_points.items()
                if not key.endswith("_xy")
            },
            "hu_moments": descriptor["hu_moments"],
            "fourier_descriptor": descriptor["fourier_descriptor"],
            "skeleton_graph": descriptor["skeleton_graph"],
        }
    )


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    """Save one RGB visualization and fail loudly on write errors."""

    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save image: {path}")
