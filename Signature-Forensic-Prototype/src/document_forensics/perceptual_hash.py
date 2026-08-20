"""Cryptographic and perceptual image hashing for duplicate screening."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from .common import load_rgb_image


def calculate_image_hashes(
    image: object,
    hash_size: int = 16,
) -> dict[str, str]:
    """Calculate normalized-pixel SHA-256 and three perceptual hashes."""

    rgb = load_rgb_image(image)
    normalized = cv2.resize(
        rgb,
        (512, 256),
        interpolation=cv2.INTER_AREA,
    )
    pil = Image.fromarray(normalized)
    return {
        "normalized_pixel_sha256": hashlib.sha256(
            normalized.tobytes()
        ).hexdigest(),
        "phash": str(imagehash.phash(pil, hash_size=hash_size)),
        "dhash": str(imagehash.dhash(pil, hash_size=hash_size)),
        "whash": str(imagehash.whash(pil, hash_size=hash_size)),
    }


def calculate_file_sha256(path: str | Path) -> str:
    """Calculate the exact source-file hash without decoding the image."""

    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_hashes(
    first: dict[str, str],
    second: dict[str, str],
) -> dict[str, object]:
    """Return exact normalized equality and perceptual Hamming distances."""

    distances = {
        name: int(
            imagehash.hex_to_hash(first[name])
            - imagehash.hex_to_hash(second[name])
        )
        for name in ("phash", "dhash", "whash")
    }
    return {
        "exact_normalized_pixel_match": (
            first["normalized_pixel_sha256"]
            == second["normalized_pixel_sha256"]
        ),
        "perceptual_distances": distances,
        "minimum_perceptual_distance": min(distances.values()),
        "mean_perceptual_distance": round(
            float(np.mean(list(distances.values()))),
            6,
        ),
    }
