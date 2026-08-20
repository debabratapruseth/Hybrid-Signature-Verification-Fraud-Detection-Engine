"""Untouched CEDAR external validation for a BHSig260-trained verifier.

This module is intentionally separate from model training. CEDAR writers never
enter optimization or threshold calibration. The saved report measures how the
already-trained model and already-selected BHSig260 threshold behave after a
dataset shift.

External validation does not improve the model by itself. It provides evidence
about generalization and is used by Branch 1 only as a reliability factor.
"""

from __future__ import annotations

import itertools
import random
import re
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .verification import SignaturePair, collect_pair_scores


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


class CedarWriter(NamedTuple):
    """CEDAR genuine and forged images associated with one writer ID."""

    writer_id: str
    genuine: list[Path]
    forged: list[Path]


def discover_cedar(dataset_directory: str | Path) -> list[CedarWriter]:
    """Discover common CEDAR filename layouts and group samples by writer.

    Common files are named like ``original_1_1.png`` and
    ``forgeries_1_1.png``. Folder names containing ``genuine``, ``original``,
    ``forged``, or ``forgery`` are also supported.
    """

    root = Path(dataset_directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CEDAR directory not found: {root}")

    grouped: dict[str, dict[str, list[Path]]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        sample_type = _cedar_sample_type(path)
        writer_id = _cedar_writer_id(path)
        if sample_type is None or writer_id is None:
            continue
        grouped.setdefault(
            writer_id,
            {"genuine": [], "forged": []},
        )[sample_type].append(path)

    writers = [
        CedarWriter(
            writer_id=writer_id,
            genuine=values["genuine"],
            forged=values["forged"],
        )
        for writer_id, values in sorted(grouped.items())
        if values["genuine"] and values["forged"]
    ]
    if not writers:
        raise ValueError(
            "No complete CEDAR writers were found. Expected original/genuine "
            "and forged/forgery filenames or folders."
        )
    return writers


def create_cedar_external_pairs(
    writers: list[CedarWriter],
    *,
    pairs_per_class_per_writer: int = 120,
    random_seed: int = 42,
) -> list[SignaturePair]:
    """Create balanced untouched genuine and skilled-forgery pairs.

    Positive pairs contain two genuine samples from the same CEDAR writer.
    Negative pairs contain a genuine sample and a skilled forgery attributed to
    that same writer. Sampling repeats only when a writer has fewer unique
    candidates than requested.
    """

    if not writers:
        raise ValueError("At least one CEDAR writer is required.")
    if pairs_per_class_per_writer <= 0:
        raise ValueError("pairs_per_class_per_writer must be positive.")

    rng = random.Random(random_seed)
    pairs: list[SignaturePair] = []
    for writer in writers:
        positive_candidates = list(
            itertools.combinations(writer.genuine, 2)
        )
        negative_candidates = list(
            itertools.product(writer.genuine, writer.forged)
        )
        for first, second in _sample_pairs(
            positive_candidates,
            pairs_per_class_per_writer,
            rng,
        ):
            pairs.append(
                SignaturePair(
                    first,
                    second,
                    1,
                    "cedar_genuine_genuine",
                )
            )
        for first, second in _sample_pairs(
            negative_candidates,
            pairs_per_class_per_writer,
            rng,
        ):
            pairs.append(
                SignaturePair(
                    first,
                    second,
                    0,
                    "cedar_genuine_skilled_forgery",
                )
            )
    rng.shuffle(pairs)
    return pairs


def calculate_external_validation_report(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    bhsig_fixed_threshold: float,
    writer_count: int,
    genuine_image_count: int,
    forged_image_count: int,
) -> dict[str, Any]:
    """Calculate operational metrics at the untouched BHSig260 threshold.

    The CEDAR-derived EER threshold is saved for description only. Using it as
    the operational threshold would tune the system on the external test and
    invalidate the claim that CEDAR remained untouched.
    """

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(scores) != len(labels) or not len(scores):
        raise ValueError("Scores and labels must be non-empty and equally sized.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("External validation requires both pair classes.")

    predictions = (scores >= float(bhsig_fixed_threshold)).astype(np.int64)
    positive = labels == 1
    negative = labels == 0
    false_acceptance = float(np.mean(predictions[negative] == 1))
    false_rejection = float(np.mean(predictions[positive] == 0))
    accuracy = float(np.mean(predictions == labels))

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels,
        scores,
    )
    false_negative_rate = 1.0 - true_positive_rate
    eer_index = int(
        np.nanargmin(
            np.abs(false_positive_rate - false_negative_rate)
        )
    )
    descriptive_eer = float(
        (
            false_positive_rate[eer_index]
            + false_negative_rate[eer_index]
        )
        / 2.0
    )
    return {
        "evaluation_type": "untouched_external_test",
        "training_dataset": "BHSig260",
        "external_test_dataset": "CEDAR",
        "pair_count": int(len(scores)),
        "positive_pair_count": int(positive.sum()),
        "negative_pair_count": int(negative.sum()),
        "bhsig_fixed_threshold": round(float(bhsig_fixed_threshold), 6),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 6),
        "accuracy_at_bhsig_threshold": round(accuracy, 6),
        "false_acceptance_rate_at_bhsig_threshold": round(
            false_acceptance,
            6,
        ),
        "false_rejection_rate_at_bhsig_threshold": round(
            false_rejection,
            6,
        ),
        "genuine_score_mean": round(float(scores[positive].mean()), 6),
        "forgery_score_mean": round(float(scores[negative].mean()), 6),
        "cedar_descriptive_eer": round(descriptive_eer, 6),
        "cedar_descriptive_eer_threshold": round(
            float(thresholds[eer_index]),
            6,
        ),
        "threshold_note": (
            "Operational metrics use the untouched BHSig260 validation "
            "threshold. The CEDAR EER threshold is descriptive only."
        ),
        "dataset_summary": {
            "writers": int(writer_count),
            "genuine_images": int(genuine_image_count),
            "forged_images": int(forged_image_count),
        },
    }


def evaluate_cedar_external(
    *,
    model: Any,
    writers: list[CedarWriter],
    verification_config: dict[str, Any],
    calibration: dict[str, Any],
    device: str,
    pairs_per_class_per_writer: int = 120,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Score untouched CEDAR pairs and return a JSON-safe report."""

    pairs = create_cedar_external_pairs(
        writers,
        pairs_per_class_per_writer=pairs_per_class_per_writer,
        random_seed=random_seed,
    )
    scores, labels = collect_pair_scores(
        model,
        pairs,
        verification_config,
        device,
    )
    return calculate_external_validation_report(
        scores=scores,
        labels=labels,
        bhsig_fixed_threshold=float(calibration["eer_threshold"]),
        writer_count=len(writers),
        genuine_image_count=sum(len(writer.genuine) for writer in writers),
        forged_image_count=sum(len(writer.forged) for writer in writers),
    )


def _cedar_sample_type(path: Path) -> str | None:
    """Infer genuine/forged status from CEDAR path components."""

    text = " ".join(part.lower() for part in path.parts)
    if any(token in text for token in ("forgeries", "forgery", "forged")):
        return "forged"
    if any(token in text for token in ("original", "genuine")):
        return "genuine"
    return None


def _cedar_writer_id(path: Path) -> str | None:
    """Extract the first numeric writer identifier from the filename."""

    numbers = re.findall(r"\d+", path.stem)
    return numbers[0] if numbers else None


def _sample_pairs(
    candidates: list[tuple[Path, Path]],
    count: int,
    rng: random.Random,
) -> list[tuple[Path, Path]]:
    """Sample unique candidates where possible, otherwise with replacement."""

    if not candidates:
        raise ValueError("A CEDAR writer has no available pair candidates.")
    if count <= len(candidates):
        return rng.sample(candidates, count)
    return [rng.choice(candidates) for _ in range(count)]
