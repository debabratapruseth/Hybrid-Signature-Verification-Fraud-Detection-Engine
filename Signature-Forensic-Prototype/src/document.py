"""Load document pages and create conservative analysis images.

This module is the first stage of the signature-forensics prototype. It keeps
the decoded source page unchanged and creates a separate working image for later
signature detection. Images returned by this module use RGB channel order.

The quality measurements are screening checks, not forensic conclusions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

import cv2
import fitz
import numpy as np
import yaml


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class QualityCheck(TypedDict):
    """One understandable pass/fail document-quality check."""

    name: str
    passed: bool
    message: str
    value: float | int | str | None


class DocumentQuality(TypedDict):
    """Quality summary calculated before signature detection."""

    passed: bool
    quality_score: float
    checks: list[QualityCheck]
    warnings: list[str]
    measurements: dict[str, float | int]


class DocumentResult(TypedDict):
    """Data contract returned by :func:`load_document`."""

    source_path: str
    page_number: int
    original_image: np.ndarray
    preprocessed_image: np.ndarray
    width: int
    height: int
    quality: DocumentQuality


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read the YAML configuration used by document processing.

    Args:
        config_path: Location of the project configuration file.

    Returns:
        The parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML is empty or lacks the ``document`` section.
        yaml.YAMLError: If the YAML syntax is invalid.
    """

    resolved_path = Path(config_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict) or not isinstance(config.get("document"), dict):
        raise ValueError(
            f"Configuration must contain a 'document' mapping: {resolved_path}"
        )
    return config


def resize_preserving_aspect(
    image: np.ndarray,
    maximum_dimension: int,
) -> np.ndarray:
    """Shrink an image without stretching it or changing its aspect ratio.

    Images smaller than ``maximum_dimension`` are copied rather than enlarged.
    """

    _validate_image_array(image)
    if maximum_dimension <= 0:
        raise ValueError("maximum_dimension must be greater than zero")

    height, width = image.shape[:2]
    largest_dimension = max(height, width)
    if largest_dimension <= maximum_dimension:
        return image.copy()

    scale = maximum_dimension / largest_dimension
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    return cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def correct_perspective(
    image: np.ndarray,
    corner_points: np.ndarray,
) -> np.ndarray:
    """Rectify a page using four manually or externally identified corners.

    Automatic perspective correction is intentionally not applied during normal
    preprocessing because a bad corner estimate can damage the analysis image.

    Args:
        image: RGB or grayscale page image.
        corner_points: Four ``(x, y)`` points in this order: top-left,
            top-right, bottom-right, bottom-left.

    Returns:
        A perspective-corrected copy of the page.
    """

    _validate_image_array(image)
    points = np.asarray(corner_points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("corner_points must have shape (4, 2)")

    top_left, top_right, bottom_right, bottom_left = points
    top_width = np.linalg.norm(top_right - top_left)
    bottom_width = np.linalg.norm(bottom_right - bottom_left)
    left_height = np.linalg.norm(bottom_left - top_left)
    right_height = np.linalg.norm(bottom_right - top_right)

    output_width = max(1, round(max(top_width, bottom_width)))
    output_height = max(1, round(max(left_height, right_height)))
    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(
        image,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def estimate_skew_degrees(grayscale_image: np.ndarray) -> float:
    """Estimate page skew from long, nearly horizontal edges.

    A zero value is returned when the page contains too little reliable line
    information. This is safer than forcing a rotation based on weak evidence.
    """

    _validate_image_array(grayscale_image)
    if grayscale_image.ndim != 2:
        raise ValueError("estimate_skew_degrees expects a grayscale image")

    edges = cv2.Canny(grayscale_image, 50, 150, apertureSize=3)
    minimum_line_length = max(30, grayscale_image.shape[1] // 5)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=minimum_line_length,
        maxLineGap=20,
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    # OpenCV versions return either (N, 1, 4) or (N, 4). Reshaping supports
    # both layouts without changing the coordinate values.
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if -45.0 <= angle <= 45.0:
            angles.append(angle)

    if not angles:
        return 0.0
    return float(np.median(angles))


def preprocess_page(
    original_image: np.ndarray,
    document_config: dict[str, Any],
) -> tuple[np.ndarray, float]:
    """Create a mild contrast-enhanced, denoised, and deskewed working copy.

    The input array is never modified. Perspective correction is excluded from
    this automatic path because it requires reliable page-corner coordinates.

    Returns:
        A tuple containing the RGB working image and estimated skew in degrees.
    """

    _validate_image_array(original_image)
    working_image = original_image.copy()
    maximum_dimension = int(document_config["maximum_analysis_dimension"])
    working_image = resize_preserving_aspect(working_image, maximum_dimension)

    grayscale = _to_grayscale(working_image)
    clip_limit = float(document_config["contrast_clip_limit"])
    tile_size = int(document_config["contrast_tile_grid_size"])
    if tile_size <= 0:
        raise ValueError("contrast_tile_grid_size must be greater than zero")

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    enhanced = clahe.apply(grayscale)

    denoise_strength = int(document_config["denoise_strength"])
    if denoise_strength < 0:
        raise ValueError("denoise_strength cannot be negative")
    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=denoise_strength,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    skew_degrees = estimate_skew_degrees(denoised)
    extreme_skew = float(document_config["extreme_skew_degrees"])
    if 0.1 <= abs(skew_degrees) <= extreme_skew:
        denoised = _rotate_without_cropping(denoised, skew_degrees)
    elif abs(skew_degrees) > extreme_skew:
        LOGGER.warning(
            "Estimated skew %.2f° is too large for automatic correction",
            skew_degrees,
        )

    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB), skew_degrees


def assess_document_quality(
    original_image: np.ndarray,
    skew_degrees: float,
    document_config: dict[str, Any],
) -> DocumentQuality:
    """Calculate interpretable quality checks for a decoded source page."""

    _validate_image_array(original_image)
    grayscale = _to_grayscale(original_image)
    height, width = grayscale.shape
    brightness = float(np.mean(grayscale))
    blur_variance = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())

    minimum_width = int(document_config["minimum_width"])
    minimum_height = int(document_config["minimum_height"])
    dark_threshold = float(document_config["dark_brightness_threshold"])
    bright_threshold = float(document_config["bright_brightness_threshold"])
    blur_threshold = float(document_config["blur_variance_threshold"])
    extreme_skew = float(document_config["extreme_skew_degrees"])

    checks: list[QualityCheck] = [
        _make_check(
            "valid_dimensions",
            width > 0 and height > 0,
            f"Decoded page dimensions are {width} × {height} pixels",
            f"Invalid page dimensions: {width} × {height} pixels",
            f"{width}x{height}",
        ),
        _make_check(
            "resolution_adequate",
            width >= minimum_width and height >= minimum_height,
            f"Resolution meets the configured minimum of "
            f"{minimum_width} × {minimum_height} pixels",
            f"Resolution is below the configured minimum of "
            f"{minimum_width} × {minimum_height} pixels",
            f"{width}x{height}",
        ),
        _make_check(
            "not_too_dark",
            brightness >= dark_threshold,
            f"Mean brightness {brightness:.1f} is above the dark-page limit",
            f"Mean brightness {brightness:.1f} suggests a very dark page",
            round(brightness, 2),
        ),
        _make_check(
            "not_too_bright",
            brightness <= bright_threshold,
            f"Mean brightness {brightness:.1f} is below the bright-page limit",
            f"Mean brightness {brightness:.1f} suggests a washed-out page",
            round(brightness, 2),
        ),
        _make_check(
            "acceptable_blur",
            blur_variance >= blur_threshold,
            f"Sharpness measure {blur_variance:.1f} meets the prototype limit",
            f"Sharpness measure {blur_variance:.1f} suggests possible blur",
            round(blur_variance, 2),
        ),
        _make_check(
            "acceptable_skew",
            abs(skew_degrees) <= extreme_skew,
            f"Estimated skew {skew_degrees:.2f}° is within the correction range",
            f"Estimated skew {skew_degrees:.2f}° is too large for safe "
            "automatic correction",
            round(skew_degrees, 2),
        ),
    ]

    passed_count = sum(check["passed"] for check in checks)
    quality_score = passed_count / len(checks)
    warnings = [
        check["message"] for check in checks if not check["passed"]
    ]
    return {
        "passed": len(warnings) == 0,
        "quality_score": round(quality_score, 3),
        "checks": checks,
        "warnings": warnings,
        "measurements": {
            "width": width,
            "height": height,
            "mean_brightness": round(brightness, 2),
            "blur_variance": round(blur_variance, 2),
            "estimated_skew_degrees": round(skew_degrees, 2),
        },
    }


def load_document(
    source_path: str | Path,
    page_number: int = 0,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> DocumentResult:
    """Load one document page, preserve it, and make an analysis copy.

    Args:
        source_path: PDF, PNG, JPG, or JPEG file.
        page_number: Zero-based PDF page index. Image files only accept zero.
        config_path: Project YAML configuration.

    Returns:
        A :class:`DocumentResult` containing independent original and working
        RGB arrays plus quality results.

    Raises:
        FileNotFoundError: If the source file does not exist.
        ValueError: For an unsupported, unreadable, empty, encrypted, or
            incorrectly indexed document.
    """

    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Document not found: {source}")
    if page_number < 0:
        raise ValueError("page_number cannot be negative")

    config = load_config(config_path)
    document_config = config["document"]
    supported_extensions = {
        str(extension).lower()
        for extension in document_config["supported_extensions"]
    }
    extension = source.suffix.lower()
    if extension not in supported_extensions:
        supported = ", ".join(sorted(supported_extensions))
        raise ValueError(
            f"Unsupported document format '{extension}'. Supported: {supported}"
        )

    if extension == ".pdf":
        original_image = _load_pdf_page(
            source,
            page_number,
            int(document_config["pdf_render_dpi"]),
        )
    else:
        if page_number != 0:
            raise ValueError("Image files contain one page; page_number must be 0")
        original_image = _load_image_file(source)

    # Force independent arrays so later processing cannot alter the evidence copy.
    original_evidence = np.ascontiguousarray(original_image.copy())
    preprocessed_image, skew_degrees = preprocess_page(
        original_evidence,
        document_config,
    )
    quality = assess_document_quality(
        original_evidence,
        skew_degrees,
        document_config,
    )
    height, width = original_evidence.shape[:2]

    LOGGER.info(
        "Loaded %s page %d at %d × %d pixels",
        source.name,
        page_number,
        width,
        height,
    )
    return {
        "source_path": str(source),
        "page_number": page_number,
        "original_image": original_evidence,
        "preprocessed_image": preprocessed_image,
        "width": width,
        "height": height,
        "quality": quality,
    }


def _load_image_file(source_path: Path) -> np.ndarray:
    """Decode a local image as RGB without relying on ASCII-only paths."""

    encoded_bytes = np.fromfile(source_path, dtype=np.uint8)
    decoded = cv2.imdecode(encoded_bytes, cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.size == 0:
        raise ValueError(f"Image is empty or unreadable: {source_path}")

    if decoded.ndim == 2:
        return cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
    if decoded.shape[2] == 4:
        return cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB)
    if decoded.shape[2] == 3:
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported image channel layout: {decoded.shape}")


def _load_pdf_page(
    source_path: Path,
    page_number: int,
    render_dpi: int,
) -> np.ndarray:
    """Render one PDF page as an RGB NumPy array."""

    if render_dpi <= 0:
        raise ValueError("pdf_render_dpi must be greater than zero")

    try:
        document = fitz.open(source_path)
    except (fitz.FileDataError, RuntimeError) as error:
        raise ValueError(f"PDF is unreadable: {source_path}") from error

    try:
        if document.needs_pass:
            raise ValueError("Password-protected PDFs are not supported")
        if document.page_count == 0:
            raise ValueError("PDF contains no pages")
        if page_number >= document.page_count:
            raise ValueError(
                f"PDF page {page_number} does not exist; valid range is "
                f"0 to {document.page_count - 1}"
            )

        page = document.load_page(page_number)
        scale = render_dpi / 72.0
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale),
            colorspace=fitz.csRGB,
            alpha=False,
        )
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )
        if image.size == 0:
            raise ValueError(f"Rendered PDF page {page_number} is empty")
        return np.ascontiguousarray(image[:, :, :3])
    finally:
        document.close()


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale, or copy an existing grayscale image."""

    if image.ndim == 2:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _rotate_without_cropping(
    grayscale_image: np.ndarray,
    angle_degrees: float,
) -> np.ndarray:
    """Rotate a grayscale page while expanding the canvas to avoid clipping."""

    height, width = grayscale_image.shape
    center = (width / 2.0, height / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cosine = abs(rotation[0, 0])
    sine = abs(rotation[0, 1])
    output_width = int((height * sine) + (width * cosine))
    output_height = int((height * cosine) + (width * sine))
    rotation[0, 2] += (output_width / 2.0) - center[0]
    rotation[1, 2] += (output_height / 2.0) - center[1]
    return cv2.warpAffine(
        grayscale_image,
        rotation,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _make_check(
    name: str,
    passed: bool,
    passed_message: str,
    failed_message: str,
    value: float | int | str | None,
) -> QualityCheck:
    """Build one quality check with a message appropriate to its result."""

    return {
        "name": name,
        "passed": bool(passed),
        "message": passed_message if passed else failed_message,
        "value": value,
    }


def _validate_image_array(image: np.ndarray) -> None:
    """Reject empty or unsupported image arrays with a clear error."""

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.size == 0:
        raise ValueError("image array is empty")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must have 2 or 3 dimensions, received {image.ndim}")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError(f"unsupported image channel count: {image.shape[2]}")
