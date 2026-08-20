"""Writer-independent Siamese signature verification with ResNet-18.

The Siamese architecture applies one shared encoder to both signatures. It
learns to place signatures from the same writer closer in embedding space and
dissimilar signatures farther apart.

This module reports similarity observations. It does not prove that a signature
is genuine or forged, and thresholds must be calibrated on validation data.
"""

from __future__ import annotations

import itertools
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import yaml
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


class WriterSamples(NamedTuple):
    """Genuine and forged samples belonging to one writer."""

    writer_id: str
    script: str
    genuine: list[Path]
    forged: list[Path]


class SignaturePair(NamedTuple):
    """One training or evaluation pair."""

    first_path: Path
    second_path: Path
    label: int
    pair_type: str


class VerificationResult(TypedDict):
    """Similarity summary for one questioned signature."""

    reference_scores: list[float]
    mean_similarity: float
    median_similarity: float
    minimum_similarity: float
    maximum_similarity: float
    score_spread: float
    verification_status: str


def load_verification_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Read the ``verification`` section from config.yaml."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a dictionary")
    if not isinstance(config.get("verification"), dict):
        raise ValueError("config.yaml must contain a 'verification' section")
    return config["verification"]


class SiameseResNet18(nn.Module):
    """Shared ResNet-18 encoder producing L2-normalized embeddings."""

    def __init__(
        self,
        embedding_dimension: int = 256,
        use_pretrained_weights: bool = True,
    ) -> None:
        """Create the shared encoder and trainable embedding projection.

        ``use_pretrained_weights`` is used for initial training. Saved-checkpoint
        inference constructs the same architecture without downloading or
        reapplying ImageNet weights before loading trained parameters.
        """

        super().__init__()
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")

        weights = (
            ResNet18_Weights.DEFAULT
            if use_pretrained_weights
            else None
        )
        backbone = resnet18(weights=weights)
        feature_dimension = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projection = nn.Linear(
            feature_dimension,
            embedding_dimension,
        )
        self.embedding_dimension = embedding_dimension

    def encode(self, signature: torch.Tensor) -> torch.Tensor:
        """Encode one batch using the shared branch."""

        features = self.encoder(signature)
        embeddings = self.projection(features)
        return functional.normalize(
            embeddings,
            p=2,
            dim=1,
        )

    def forward(
        self,
        first_signature: torch.Tensor,
        second_signature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode both inputs with exactly the same weights."""

        return (
            self.encode(first_signature),
            self.encode(second_signature),
        )


class ContrastiveLoss(nn.Module):
    """Contrastive loss where label 1 means similar and 0 dissimilar."""

    def __init__(self, margin: float = 1.0) -> None:
        """Store the minimum desired separation for dissimilar pairs."""

        super().__init__()
        if margin <= 0:
            raise ValueError("margin must be positive")
        self.margin = margin

    def forward(
        self,
        first_embedding: torch.Tensor,
        second_embedding: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate mean contrastive loss for one batch."""

        distances = functional.pairwise_distance(
            first_embedding,
            second_embedding,
        )
        positive_loss = labels * distances.pow(2)
        negative_loss = (1.0 - labels) * functional.relu(
            self.margin - distances
        ).pow(2)
        return 0.5 * (positive_loss + negative_loss).mean()


def discover_bhsig260(
    bengali_directory: str | Path,
    hindi_directory: str | Path,
) -> list[WriterSamples]:
    """Discover BHSig260 writers from Bengali and Hindi directories.

    Expected layout:

    ``BHSig260-Bengali/<writer folder>/<signature images>``

    ``BHSig260-Hindi/<writer folder>/<signature images>``

    Filenames normally contain a standalone ``G`` for genuine or ``F`` for
    forged. Directories named ``genuine`` or ``forged`` are also supported.
    """

    dataset_roots = {
        "bengali": Path(bengali_directory).expanduser().resolve(),
        "hindi": Path(hindi_directory).expanduser().resolve(),
    }
    all_writers: list[WriterSamples] = []

    for script, root in dataset_roots.items():
        if not root.is_dir():
            raise FileNotFoundError(
                f"{script.title()} dataset directory not found: {root}"
            )

        writer_directories = sorted(
            directory
            for directory in root.iterdir()
            if directory.is_dir()
        )
        if not writer_directories:
            raise ValueError(
                f"No writer directories found under {root}"
            )

        for writer_directory in writer_directories:
            image_paths = sorted(
                path
                for path in writer_directory.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in SUPPORTED_IMAGE_EXTENSIONS
                )
            )
            genuine = []
            forged = []
            for image_path in image_paths:
                sample_type = classify_bhsig_sample(image_path)
                if sample_type == "genuine":
                    genuine.append(image_path)
                elif sample_type == "forged":
                    forged.append(image_path)

            # Ignore unrelated folders, but reject partially parsed writers.
            if not genuine and not forged:
                continue
            if not genuine or not forged:
                raise ValueError(
                    f"Could not identify both genuine and forged samples "
                    f"for writer folder {writer_directory}"
                )

            writer_id = f"{script}_{writer_directory.name}"
            all_writers.append(
                WriterSamples(
                    writer_id=writer_id,
                    script=script,
                    genuine=genuine,
                    forged=forged,
                )
            )

    if not all_writers:
        raise ValueError("No BHSig260 writers were discovered")
    return all_writers


def classify_bhsig_sample(image_path: Path) -> str | None:
    """Classify a BHSig260 filename/path as genuine or forged."""

    lower_parts = [part.lower() for part in image_path.parts]
    if any(part in {"genuine", "real", "original"} for part in lower_parts):
        return "genuine"
    if any(
        part in {"forged", "forge", "forgery", "fake"}
        for part in lower_parts
    ):
        return "forged"

    tokens = [
        token.upper()
        for token in re.split(r"[-_.\s]+", image_path.stem)
        if token
    ]
    if "G" in tokens:
        return "genuine"
    if "F" in tokens:
        return "forged"
    return None


def summarize_writers(
    writers: list[WriterSamples],
) -> dict[str, int]:
    """Count writers and samples for dataset validation."""

    return {
        "writers": len(writers),
        "bengali_writers": sum(
            writer.script == "bengali" for writer in writers
        ),
        "hindi_writers": sum(
            writer.script == "hindi" for writer in writers
        ),
        "genuine_samples": sum(
            len(writer.genuine) for writer in writers
        ),
        "forged_samples": sum(
            len(writer.forged) for writer in writers
        ),
    }


def create_writer_disjoint_split(
    writers: list[WriterSamples],
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
    random_seed: int = 42,
) -> dict[str, list[WriterSamples]]:
    """Split writers within each script without writer leakage."""

    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Train and validation fractions must leave a test set")

    random_generator = random.Random(random_seed)
    split = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for script in sorted({writer.script for writer in writers}):
        script_writers = [
            writer for writer in writers if writer.script == script
        ]
        random_generator.shuffle(script_writers)
        writer_count = len(script_writers)
        train_count = max(1, int(writer_count * train_fraction))
        validation_count = max(
            1,
            int(writer_count * validation_fraction),
        )
        if train_count + validation_count >= writer_count:
            raise ValueError(
                f"Not enough {script} writers for three disjoint splits"
            )
        split["train"].extend(script_writers[:train_count])
        split["validation"].extend(
            script_writers[
                train_count : train_count + validation_count
            ]
        )
        split["test"].extend(
            script_writers[train_count + validation_count :]
        )

    assert_writer_disjoint(split)
    return split


def assert_writer_disjoint(
    split: dict[str, list[WriterSamples]],
) -> None:
    """Raise an error if any writer occurs in multiple splits."""

    writer_sets = {
        split_name: {
            writer.writer_id for writer in split_writers
        }
        for split_name, split_writers in split.items()
    }
    split_names = list(writer_sets)
    for first_index, first_name in enumerate(split_names):
        for second_name in split_names[first_index + 1 :]:
            overlap = (
                writer_sets[first_name]
                & writer_sets[second_name]
            )
            if overlap:
                raise ValueError(
                    f"Writer leakage between {first_name} and "
                    f"{second_name}: {sorted(overlap)}"
                )


def save_writer_split(
    split: dict[str, list[WriterSamples]],
    output_path: str | Path,
) -> None:
    """Save writer IDs so the experiment can be reproduced."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        split_name: [
            {
                "writer_id": writer.writer_id,
                "script": writer.script,
                "genuine_count": len(writer.genuine),
                "forged_count": len(writer.forged),
            }
            for writer in split_writers
        ]
        for split_name, split_writers in split.items()
    }
    path.write_text(
        json.dumps(serializable, indent=2),
        encoding="utf-8",
    )


def generate_balanced_pairs(
    writers: list[WriterSamples],
    positive_pairs_per_writer: int,
    skilled_negative_pairs_per_writer: int,
    random_negative_pairs_per_writer: int,
    random_seed: int,
) -> list[SignaturePair]:
    """Create balanced positive, skilled-forgery, and random-impostor pairs."""

    if not writers:
        raise ValueError("At least one writer is required")
    requested_counts = (
        positive_pairs_per_writer,
        skilled_negative_pairs_per_writer,
        random_negative_pairs_per_writer,
    )
    if any(count < 0 for count in requested_counts):
        raise ValueError("Pair counts cannot be negative")
    if (
        skilled_negative_pairs_per_writer
        + random_negative_pairs_per_writer
        != positive_pairs_per_writer
    ):
        raise ValueError(
            "Total negative pairs per writer must equal positive pairs "
            "to keep labels balanced"
        )

    random_generator = random.Random(random_seed)
    pairs: list[SignaturePair] = []
    writers_by_id = {
        writer.writer_id: writer for writer in writers
    }

    for writer in writers:
        positive_candidates = list(
            itertools.combinations(writer.genuine, 2)
        )
        random_generator.shuffle(positive_candidates)
        for first_path, second_path in _sample_with_replacement(
            positive_candidates,
            positive_pairs_per_writer,
            random_generator,
        ):
            pairs.append(
                SignaturePair(
                    first_path,
                    second_path,
                    1,
                    "genuine_genuine",
                )
            )

        skilled_candidates = list(
            itertools.product(
                writer.genuine,
                writer.forged,
            )
        )
        for first_path, second_path in _sample_with_replacement(
            skilled_candidates,
            skilled_negative_pairs_per_writer,
            random_generator,
        ):
            pairs.append(
                SignaturePair(
                    first_path,
                    second_path,
                    0,
                    "genuine_skilled_forgery",
                )
            )

        other_writers = [
            candidate
            for candidate_id, candidate in writers_by_id.items()
            if candidate_id != writer.writer_id
        ]
        if random_negative_pairs_per_writer and not other_writers:
            raise ValueError(
                "Random-impostor pairs require at least two writers"
            )
        for _ in range(random_negative_pairs_per_writer):
            other_writer = random_generator.choice(other_writers)
            pairs.append(
                SignaturePair(
                    random_generator.choice(writer.genuine),
                    random_generator.choice(other_writer.genuine),
                    0,
                    "genuine_random_impostor",
                )
            )

    random_generator.shuffle(pairs)
    return pairs


def _sample_with_replacement(
    candidates: list[tuple[Path, Path]],
    count: int,
    random_generator: random.Random,
) -> list[tuple[Path, Path]]:
    """Sample pair candidates, repeating only when necessary."""

    if count == 0:
        return []
    if not candidates:
        raise ValueError("No candidate pairs are available")
    if count <= len(candidates):
        return random_generator.sample(candidates, count)
    return [
        random_generator.choice(candidates)
        for _ in range(count)
    ]


def prepare_signature_image(
    image: Image.Image,
    input_width: int,
    input_height: int,
    padding: int = 8,
) -> Image.Image:
    """Place a grayscale signature on a white canvas without stretching."""

    grayscale = image.convert("L")
    available_size = (
        input_width - 2 * padding,
        input_height - 2 * padding,
    )
    if min(available_size) <= 0:
        raise ValueError("Padding leaves no usable image area")
    contained = ImageOps.contain(
        grayscale,
        available_size,
        method=Image.Resampling.LANCZOS,
    )
    canvas = Image.new(
        "L",
        (input_width, input_height),
        color=255,
    )
    paste_position = (
        (input_width - contained.width) // 2,
        (input_height - contained.height) // 2,
    )
    canvas.paste(contained, paste_position)
    return canvas.convert("RGB")


def build_signature_transform(
    verification_config: dict[str, Any],
    training: bool,
) -> Any:
    """Create preprocessing compatible with pretrained ResNet-18."""

    input_width = int(verification_config["input_width"])
    input_height = int(verification_config["input_height"])
    operations = [
        transforms.Lambda(
            lambda image: prepare_signature_image(
                image,
                input_width,
                input_height,
            )
        )
    ]
    if training:
        operations.extend(
            [
                transforms.RandomAffine(
                    degrees=3,
                    translate=(0.02, 0.02),
                    scale=(0.95, 1.05),
                    fill=255,
                ),
                transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                ),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=verification_config["imagenet_mean"],
                std=verification_config["imagenet_std"],
            ),
        ]
    )
    return transforms.Compose(operations)


class SignaturePairDataset(Dataset):
    """Lazy image loader for a fixed list of signature pairs."""

    def __init__(
        self,
        pairs: list[SignaturePair],
        transform: Any,
    ) -> None:
        """Retain pair metadata and the preprocessing/augmentation transform."""

        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of fixed pairs available to the data loader."""

        return len(self.pairs)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load, transform, and label one pair only when it is requested."""

        pair = self.pairs[index]
        with Image.open(pair.first_path) as first_image:
            first_tensor = self.transform(first_image.copy())
        with Image.open(pair.second_path) as second_image:
            second_tensor = self.transform(second_image.copy())
        label = torch.tensor(
            float(pair.label),
            dtype=torch.float32,
        )
        return first_tensor, second_tensor, label


def train_siamese_model(
    model: SiameseResNet18,
    train_pairs: list[SignaturePair],
    validation_pairs: list[SignaturePair],
    verification_config: dict[str, Any],
    checkpoint_path: str | Path,
    history_path: str | Path,
    device: str,
) -> list[dict[str, float | int]]:
    """Train, validate, and persist the best Siamese checkpoint."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    history_output = Path(history_path).expanduser().resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history_output.parent.mkdir(parents=True, exist_ok=True)

    train_dataset = SignaturePairDataset(
        train_pairs,
        build_signature_transform(
            verification_config,
            training=True,
        ),
    )
    validation_dataset = SignaturePairDataset(
        validation_pairs,
        build_signature_transform(
            verification_config,
            training=False,
        ),
    )
    batch_size = int(verification_config["batch_size"])
    workers = int(verification_config.get("data_loader_workers", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device == "cuda",
    )

    model.to(device)
    loss_function = ContrastiveLoss(
        margin=float(verification_config["contrastive_margin"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(verification_config["learning_rate"]),
        weight_decay=float(verification_config["weight_decay"]),
    )
    epochs = int(verification_config["epochs"])
    best_validation_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        training_loss = _run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
        )
        validation_loss = _run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            optimizer=None,
        )
        epoch_result = {
            "epoch": epoch,
            "training_loss": round(training_loss, 6),
            "validation_loss": round(validation_loss, 6),
        }
        history.append(epoch_result)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train {training_loss:.5f} | "
            f"validation {validation_loss:.5f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embedding_dimension": model.embedding_dimension,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "verification_config": verification_config,
                },
                checkpoint,
            )

        history_output.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    return history


def _run_epoch(
    model: SiameseResNet18,
    data_loader: DataLoader,
    loss_function: ContrastiveLoss,
    device: str,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """Run one training or validation epoch."""

    training = optimizer is not None
    model.train(training)
    accumulated_loss = 0.0
    sample_count = 0

    for first, second, labels in data_loader:
        first = first.to(device, non_blocking=True)
        second = second.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            first_embedding, second_embedding = model(
                first,
                second,
            )
            loss = loss_function(
                first_embedding,
                second_embedding,
                labels,
            )
            if training:
                loss.backward()
                optimizer.step()

        batch_count = labels.shape[0]
        accumulated_loss += float(loss.item()) * batch_count
        sample_count += batch_count

    if not sample_count:
        raise ValueError("Data loader contains no samples")
    return accumulated_loss / sample_count


def load_siamese_checkpoint(
    checkpoint_path: str | Path,
    device: str,
) -> tuple[SiameseResNet18, dict[str, Any]]:
    """Load saved weights for a fresh inference session."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Siamese checkpoint not found: {path}")
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    model = SiameseResNet18(
        embedding_dimension=int(
            checkpoint["embedding_dimension"]
        ),
        use_pretrained_weights=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def collect_pair_scores(
    model: SiameseResNet18,
    pairs: list[SignaturePair],
    verification_config: dict[str, Any],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate cosine similarity and labels for calibration/evaluation."""

    dataset = SignaturePairDataset(
        pairs,
        build_signature_transform(
            verification_config,
            training=False,
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(verification_config["batch_size"]),
        shuffle=False,
        num_workers=int(
            verification_config.get("data_loader_workers", 2)
        ),
        pin_memory=device == "cuda",
    )
    scores = []
    labels_output = []
    model.eval()
    with torch.no_grad():
        for first, second, labels in loader:
            first_embedding, second_embedding = model(
                first.to(device),
                second.to(device),
            )
            similarities = functional.cosine_similarity(
                first_embedding,
                second_embedding,
            )
            scores.extend(similarities.cpu().numpy().tolist())
            labels_output.extend(labels.numpy().tolist())
    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(labels_output, dtype=np.int64),
    )


def calibrate_similarity_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    uncertainty_margin: float = 0.05,
) -> dict[str, Any]:
    """Calculate ROC AUC, EER, and provisional three-way boundaries."""

    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Calibration requires positive and negative labels")
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
    eer = float(
        (
            false_positive_rate[eer_index]
            + false_negative_rate[eer_index]
        )
        / 2.0
    )
    eer_threshold = float(thresholds[eer_index])
    elevated_boundary = max(
        -1.0,
        eer_threshold - uncertainty_margin,
    )
    consistent_boundary = min(
        1.0,
        eer_threshold + uncertainty_margin,
    )
    return {
        "roc_auc": round(float(roc_auc_score(labels, scores)), 6),
        "equal_error_rate": round(eer, 6),
        "eer_threshold": round(eer_threshold, 6),
        "uncertainty_margin": uncertainty_margin,
        "elevated_inconsistency_below": round(
            elevated_boundary,
            6,
        ),
        "provisionally_consistent_at_or_above": round(
            consistent_boundary,
            6,
        ),
        "positive_score_mean": round(
            float(scores[labels == 1].mean()),
            6,
        ),
        "negative_score_mean": round(
            float(scores[labels == 0].mean()),
            6,
        ),
        "calibration_pair_count": int(len(scores)),
    }


def save_calibration(
    calibration: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Persist validation-derived thresholds for future inference."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(calibration, indent=2),
        encoding="utf-8",
    )


def load_calibration(
    calibration_path: str | Path,
) -> dict[str, Any]:
    """Load saved validation thresholds."""

    path = Path(calibration_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def embed_signature(
    model: SiameseResNet18,
    signature: np.ndarray | Image.Image | str | Path,
    verification_config: dict[str, Any],
    device: str,
) -> torch.Tensor:
    """Create one normalized embedding from an array, PIL image, or path."""

    image = _open_signature(signature)
    transform = build_signature_transform(
        verification_config,
        training=False,
    )
    tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        return model.encode(tensor).squeeze(0).cpu()


def compare_with_references(
    model: SiameseResNet18,
    questioned_signature: np.ndarray | Image.Image | str | Path,
    reference_signatures: list[
        np.ndarray | Image.Image | str | Path
    ],
    verification_config: dict[str, Any],
    calibration: dict[str, Any],
    device: str,
) -> VerificationResult:
    """Compare one questioned signature with three to five references."""

    minimum_references = int(
        verification_config["minimum_reference_count"]
    )
    if len(reference_signatures) < minimum_references:
        raise ValueError(
            f"At least {minimum_references} references are required"
        )

    questioned_embedding = embed_signature(
        model,
        questioned_signature,
        verification_config,
        device,
    )
    scores = []
    for reference in reference_signatures:
        reference_embedding = embed_signature(
            model,
            reference,
            verification_config,
            device,
        )
        similarity = functional.cosine_similarity(
            questioned_embedding.unsqueeze(0),
            reference_embedding.unsqueeze(0),
        ).item()
        scores.append(float(similarity))

    mean_similarity = float(np.mean(scores))
    median_similarity = float(np.median(scores))
    minimum_similarity = float(np.min(scores))
    maximum_similarity = float(np.max(scores))
    score_spread = maximum_similarity - minimum_similarity
    consistent_boundary = float(
        calibration["provisionally_consistent_at_or_above"]
    )
    elevated_boundary = float(
        calibration["elevated_inconsistency_below"]
    )
    if score_spread > float(
        verification_config["score_spread_warning"]
    ):
        status = "manual_review"
    elif median_similarity >= consistent_boundary:
        status = "provisionally_consistent"
    elif median_similarity < elevated_boundary:
        status = "elevated_inconsistency"
    else:
        status = "manual_review"

    return {
        "reference_scores": [
            round(score, 6) for score in scores
        ],
        "mean_similarity": round(mean_similarity, 6),
        "median_similarity": round(median_similarity, 6),
        "minimum_similarity": round(minimum_similarity, 6),
        "maximum_similarity": round(maximum_similarity, 6),
        "score_spread": round(score_spread, 6),
        "verification_status": status,
    }


def validate_reference_signatures(
    reference_paths: list[str | Path],
    minimum_reference_count: int = 3,
) -> dict[str, Any]:
    """Check reference count, readability, and exact file duplication."""

    resolved_paths = [
        Path(path).expanduser().resolve()
        for path in reference_paths
    ]
    missing = [
        str(path) for path in resolved_paths if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Reference signature files not found: {missing}"
        )

    hashes = []
    for path in resolved_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.append(digest)
        with Image.open(path) as image:
            image.verify()

    duplicate_count = len(hashes) - len(set(hashes))
    warnings = []
    if len(resolved_paths) < minimum_reference_count:
        warnings.append(
            f"Only {len(resolved_paths)} references were provided; "
            f"at least {minimum_reference_count} are required"
        )
    if duplicate_count:
        warnings.append(
            f"{duplicate_count} exact duplicate reference file(s) found"
        )
    return {
        "passed": not warnings,
        "reference_count": len(resolved_paths),
        "exact_duplicate_count": duplicate_count,
        "warnings": warnings,
    }


def _open_signature(
    signature: np.ndarray | Image.Image | str | Path,
) -> Image.Image:
    """Convert supported signature input to a copied PIL image."""

    if isinstance(signature, Image.Image):
        return signature.copy()
    if isinstance(signature, np.ndarray):
        if signature.size == 0:
            raise ValueError("Signature array is empty")
        if signature.ndim == 2:
            return Image.fromarray(signature.astype(np.uint8), mode="L")
        if signature.ndim == 3 and signature.shape[2] == 3:
            return Image.fromarray(signature.astype(np.uint8), mode="RGB")
        raise ValueError(
            f"Unsupported signature array shape: {signature.shape}"
        )

    path = Path(signature).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Signature image not found: {path}")
    with Image.open(path) as image:
        return image.copy()
