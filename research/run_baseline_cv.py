"""Leakage-safe cross-validation baselines for speaker embeddings.

This script consumes the artifacts produced by
``research.extract_manifest_embeddings``.  It deliberately uses the persisted
fold assignment instead of creating a new random split.  For every fold, all
data-dependent preprocessing (including ``StandardScaler``) is fitted on that
fold's training partition only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from model_config import MODEL_CONFIGS, get_model_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDINGS_ROOT = (
    PROJECT_ROOT
    / "research_results"
    / "embeddings"
    / "librispeech_dev_clean"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "research_results"
    / "baseline_cv"
)
CLASSIFIERS = (
    "linear_svm",
    "rbf_svm",
    "logistic_regression",
    "cosine_centroid",
    "knn",
    "random_forest",
    "decision_tree",
    "xgboost",
)
DEFAULT_KNN_NEIGHBORS = 5
DEFAULT_FOREST_ESTIMATORS = 300
DEFAULT_XGB_ESTIMATORS = 200
DEFAULT_XGB_MAX_DEPTH = 6
DEFAULT_XGB_LEARNING_RATE = 0.1
REQUIRED_ARTIFACTS = (
    "X.npy",
    "y.npy",
    "clip_ids.npy",
    "folds.npy",
    "metadata.json",
)
METRIC_ORDER = (
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
)
TIMING_ORDER = (
    "scaler_fit_seconds",
    "scaler_transform_seconds",
    "classifier_fit_seconds",
    "prediction_seconds",
    "prediction_ms_per_sample",
    "fold_total_seconds",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or all speaker-embedding models with fixed "
            "manifest folds and leakage-safe preprocessing."
        )
    )
    model_selection = parser.add_mutually_exclusive_group(required=True)
    model_selection.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS),
        help="One configured embedding model.",
    )
    model_selection.add_argument(
        "--all-models",
        action="store_true",
        help="Evaluate all configured embedding models.",
    )
    classifier_selection = parser.add_mutually_exclusive_group()
    classifier_selection.add_argument(
        "--classifier",
        choices=CLASSIFIERS,
        default="linear_svm",
        help="Classifier baseline (default: linear_svm).",
    )
    classifier_selection.add_argument(
        "--all-classifiers",
        action="store_true",
        help="Evaluate all supported classifier baselines.",
    )
    parser.add_argument(
        "--tag",
        default="full",
        help="Embedding artifact tag created by the extractor.",
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=DEFAULT_EMBEDDINGS_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--protocol-file",
        type=Path,
        default=None,
        help=(
            "Optional clip_id/speaker_label/source_group/fold CSV. "
            "When supplied, embeddings are selected and reordered by clip_id "
            "and persisted folds.npy values are replaced for evaluation."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic estimator seed (default: 42).",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="SVM/logistic inverse regularization strength.",
    )
    parser.add_argument(
        "--gamma",
        default="scale",
        help="RBF SVM gamma: scale, auto, or a positive number.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Maximum logistic-regression iterations.",
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=DEFAULT_KNN_NEIGHBORS,
        help=(
            "KNN neighbor count "
            f"(default: {DEFAULT_KNN_NEIGHBORS})."
        ),
    )
    parser.add_argument(
        "--forest-estimators",
        type=int,
        default=DEFAULT_FOREST_ESTIMATORS,
        help=(
            "Random-forest tree count "
            f"(default: {DEFAULT_FOREST_ESTIMATORS})."
        ),
    )
    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=None,
        help=(
            "Optional maximum depth for the single decision tree "
            "(default: unlimited)."
        ),
    )
    parser.add_argument(
        "--xgb-estimators",
        type=int,
        default=DEFAULT_XGB_ESTIMATORS,
        help=(
            "XGBoost boosting-round count "
            f"(default: {DEFAULT_XGB_ESTIMATORS})."
        ),
    )
    parser.add_argument(
        "--xgb-max-depth",
        type=int,
        default=DEFAULT_XGB_MAX_DEPTH,
        help=(
            "XGBoost maximum tree depth "
            f"(default: {DEFAULT_XGB_MAX_DEPTH})."
        ),
    )
    parser.add_argument(
        "--xgb-learning-rate",
        type=float,
        default=DEFAULT_XGB_LEARNING_RATE,
        help=(
            "XGBoost shrinkage/learning rate "
            f"(default: {DEFAULT_XGB_LEARNING_RATE})."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing result directory.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(
            value,
            output_file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, value, allow_pickle=False)
    os.replace(temporary, path)


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, object]],
) -> None:
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temporary, path)


def prepare_output(
    output_directory: Path,
    overwrite: bool,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    existing = list(output_directory.iterdir())

    if existing and not overwrite:
        raise FileExistsError(
            f"Result directory is not empty:\n{output_directory}\n"
            "Use --overwrite or choose another --output-root."
        )

    if overwrite:
        known_files = [
            "run_metadata.json",
            "fold_metrics.csv",
            "aggregate_metrics.csv",
            "aggregate_metrics.json",
            "all_predictions.csv",
            "confusion_matrix.csv",
            "confusion_matrix.npy",
            "class_labels.npy",
            "leakage_audit.csv",
        ]

        for filename in known_files:
            path = output_directory / filename

            if path.is_file():
                path.unlink()

        for fold_directory in output_directory.glob("fold_*"):
            if not fold_directory.is_dir():
                continue

            for child in fold_directory.iterdir():
                if child.is_file():
                    child.unlink()

            try:
                fold_directory.rmdir()
            except OSError:
                raise RuntimeError(
                    "Cannot safely replace non-file content in:\n"
                    f"{fold_directory}"
                )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        value = json.load(input_file)

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in:\n{path}")

    return value


def load_manifest_index(
    metadata: dict[str, Any],
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    manifest_value = metadata.get("manifest_path")

    if not manifest_value:
        raise ValueError(
            "metadata.json does not contain manifest_path."
        )

    manifest_path = Path(str(manifest_value))

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Source manifest was not found:\n{manifest_path}"
        )

    actual_hash = file_sha256(manifest_path)
    expected_hash = str(metadata.get("manifest_sha256", "")).lower()

    if not expected_hash:
        raise ValueError(
            "metadata.json does not contain manifest_sha256."
        )

    if actual_hash.lower() != expected_hash:
        raise ValueError(
            "Source manifest hash differs from extraction metadata. "
            "Refusing to evaluate potentially misaligned artifacts."
        )

    index: dict[str, dict[str, str]] = {}

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        reader = csv.DictReader(manifest_file)
        required = {
            "clip_id",
            "speaker_label",
            "fold",
            "source_group",
        }

        if reader.fieldnames is None:
            raise ValueError("Source manifest has no header.")

        missing = required - set(reader.fieldnames)

        if missing:
            raise ValueError(
                "Source manifest is missing columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            clip_id = row["clip_id"].strip()

            if clip_id in index:
                raise ValueError(
                    f"Duplicate clip_id in manifest: {clip_id}"
                )

            index[clip_id] = row

    return index, {
        "path": str(manifest_path.resolve()),
        "sha256": actual_hash,
        "row_count": len(index),
    }


def load_and_validate_artifacts(
    artifact_directory: Path,
    model_id: str,
) -> dict[str, Any]:
    missing = [
        filename
        for filename in REQUIRED_ARTIFACTS
        if not (artifact_directory / filename).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"Embedding artifacts are incomplete in:\n"
            f"{artifact_directory}\nMissing: {', '.join(missing)}"
        )

    metadata = load_json(artifact_directory / "metadata.json")

    if metadata.get("model_id") != model_id:
        raise ValueError(
            "Artifact model_id does not match the requested model: "
            f"{metadata.get('model_id')!r} != {model_id!r}"
        )

    X = np.load(
        artifact_directory / "X.npy",
        allow_pickle=False,
    )
    y = np.load(
        artifact_directory / "y.npy",
        allow_pickle=False,
    ).astype(str)
    clip_ids = np.load(
        artifact_directory / "clip_ids.npy",
        allow_pickle=False,
    ).astype(str)
    raw_folds = np.load(
        artifact_directory / "folds.npy",
        allow_pickle=False,
    )

    if X.ndim != 2:
        raise ValueError(
            f"X.npy must be 2-D, got shape {X.shape}."
        )

    if any(array.ndim != 1 for array in (y, clip_ids, raw_folds)):
        raise ValueError(
            "y.npy, clip_ids.npy, and folds.npy must be 1-D."
        )

    lengths = {
        len(X),
        len(y),
        len(clip_ids),
        len(raw_folds),
    }

    if len(lengths) != 1:
        raise ValueError(
            "Artifact arrays have inconsistent sample counts."
        )

    if len(X) == 0:
        raise ValueError("Embedding artifacts contain no samples.")

    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("X.npy must contain numeric values.")

    X = np.asarray(X, dtype=np.float64)

    if not np.isfinite(X).all():
        raise ValueError("X.npy contains NaN or infinite values.")

    if len(set(clip_ids.tolist())) != len(clip_ids):
        duplicates = [
            clip_id
            for clip_id, count in Counter(clip_ids).items()
            if count > 1
        ]
        raise ValueError(
            "Duplicate clip IDs would cause split leakage: "
            + ", ".join(duplicates[:10])
        )

    if any(not label.strip() for label in y):
        raise ValueError("y.npy contains an empty speaker label.")

    if not np.issubdtype(raw_folds.dtype, np.integer):
        numeric_folds = np.asarray(raw_folds, dtype=np.float64)

        if (
            not np.isfinite(numeric_folds).all()
            or not np.equal(
                numeric_folds,
                np.floor(numeric_folds),
            ).all()
        ):
            raise ValueError("folds.npy contains non-integer values.")

    folds = np.asarray(raw_folds, dtype=np.int64)
    unique_folds = np.unique(folds)

    if len(unique_folds) < 2 or np.any(unique_folds < 1):
        raise ValueError(
            "At least two positive persisted folds are required."
        )

    expected_successes = metadata.get("successful_rows")

    if (
        expected_successes is not None
        and int(expected_successes) != len(X)
    ):
        raise ValueError(
            "metadata successful_rows does not match arrays."
        )

    expected_dimension = metadata.get("embedding_dimension")

    if (
        expected_dimension is not None
        and int(expected_dimension) != X.shape[1]
    ):
        raise ValueError(
            "metadata embedding_dimension does not match X.npy."
        )

    manifest_index, manifest_info = load_manifest_index(metadata)
    source_groups: list[str] = []

    for sample_index, (clip_id, label, fold) in enumerate(
        zip(clip_ids, y, folds, strict=True)
    ):
        row = manifest_index.get(str(clip_id))

        if row is None:
            raise ValueError(
                f"clip_ids.npy[{sample_index}] is absent from manifest: "
                f"{clip_id}"
            )

        if row["speaker_label"].strip() != str(label):
            raise ValueError(
                f"Label misalignment for {clip_id}: artifact={label!r}, "
                f"manifest={row['speaker_label']!r}"
            )

        if int(row["fold"]) != int(fold):
            raise ValueError(
                f"Fold misalignment for {clip_id}: artifact={fold}, "
                f"manifest={row['fold']}"
            )

        source_groups.append(row["source_group"].strip())

    return {
        "X": X,
        "y": y,
        "clip_ids": clip_ids,
        "folds": folds,
        "source_groups": np.asarray(source_groups, dtype=str),
        "unique_folds": unique_folds,
        "metadata": metadata,
        "manifest": manifest_info,
        "artifact_hashes": {
            filename: file_sha256(artifact_directory / filename)
            for filename in REQUIRED_ARTIFACTS
        },
    }


def safe_protocol_name(path: Path, protocol_hash: str) -> str:
    stem = path.stem.lower()

    if stem.endswith("_protocol"):
        stem = stem[: -len("_protocol")]

    stem = re.sub(r"[^a-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")

    if not stem:
        stem = "external_protocol"

    return f"{stem}__{protocol_hash[:12]}"


def load_evaluation_protocol(path: Path) -> dict[str, Any]:
    protocol_path = path.resolve()

    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"Evaluation protocol was not found:\n{protocol_path}"
        )

    protocol_hash = file_sha256(protocol_path)
    rows: list[dict[str, object]] = []
    seen_clip_ids: set[str] = set()
    group_to_fold: dict[str, int] = {}

    with protocol_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as protocol_file:
        reader = csv.DictReader(protocol_file)
        required = {
            "clip_id",
            "speaker_label",
            "source_group",
            "fold",
        }

        if reader.fieldnames is None:
            raise ValueError("Evaluation protocol has no header.")

        missing = required - set(reader.fieldnames)

        if missing:
            raise ValueError(
                "Evaluation protocol is missing columns: "
                + ", ".join(sorted(missing))
            )

        for row_number, source_row in enumerate(reader, start=2):
            clip_id = source_row["clip_id"].strip()
            speaker_label = source_row["speaker_label"].strip()
            source_group = source_row["source_group"].strip()
            fold_text = source_row["fold"].strip()

            if not clip_id:
                raise ValueError(
                    f"Protocol row {row_number} has an empty clip_id."
                )

            if clip_id in seen_clip_ids:
                raise ValueError(
                    f"Duplicate protocol clip_id: {clip_id}"
                )

            if not speaker_label:
                raise ValueError(
                    f"Protocol row {row_number} has an empty speaker_label."
                )

            if not source_group:
                raise ValueError(
                    f"Protocol row {row_number} has an empty source_group."
                )

            try:
                fold = int(fold_text)
            except ValueError as error:
                raise ValueError(
                    f"Protocol row {row_number} has a non-integer fold: "
                    f"{fold_text!r}"
                ) from error

            if str(fold) != fold_text or fold < 1:
                raise ValueError(
                    f"Protocol row {row_number} fold must be a positive "
                    f"canonical integer, got {fold_text!r}."
                )

            previous_fold = group_to_fold.get(source_group)

            if previous_fold is not None and previous_fold != fold:
                raise ValueError(
                    "A source_group occurs in multiple protocol folds: "
                    f"{source_group!r} -> {previous_fold}, {fold}"
                )

            group_to_fold[source_group] = fold
            seen_clip_ids.add(clip_id)
            rows.append(
                {
                    "clip_id": clip_id,
                    "speaker_label": speaker_label,
                    "source_group": source_group,
                    "fold": fold,
                }
            )

    if not rows:
        raise ValueError("Evaluation protocol contains no data rows.")

    fold_values = sorted({int(row["fold"]) for row in rows})

    if len(fold_values) < 2:
        raise ValueError(
            "Evaluation protocol must contain at least two folds."
        )

    speakers = sorted(
        {str(row["speaker_label"]) for row in rows}
    )

    for fold in fold_values:
        test_speakers = {
            str(row["speaker_label"])
            for row in rows
            if int(row["fold"]) == fold
        }
        train_speakers = {
            str(row["speaker_label"])
            for row in rows
            if int(row["fold"]) != fold
        }
        test_groups = {
            str(row["source_group"])
            for row in rows
            if int(row["fold"]) == fold
        }
        train_groups = {
            str(row["source_group"])
            for row in rows
            if int(row["fold"]) != fold
        }
        test_only_speakers = sorted(test_speakers - train_speakers)
        group_overlap = sorted(test_groups & train_groups)

        if test_only_speakers:
            raise ValueError(
                f"Protocol fold {fold} has test-only speakers: "
                + ", ".join(test_only_speakers[:10])
            )

        if group_overlap:
            raise ValueError(
                f"Protocol fold {fold} has source_group leakage: "
                + ", ".join(group_overlap[:10])
            )

    assignment_digest = hashlib.sha256()

    for row in rows:
        canonical = (
            f"{row['clip_id']}\t{row['speaker_label']}\t"
            f"{row['source_group']}\t{row['fold']}\n"
        )
        assignment_digest.update(canonical.encode("utf-8"))

    return {
        "rows": rows,
        "path": str(protocol_path),
        "sha256": protocol_hash,
        "protocol_id": safe_protocol_name(
            protocol_path,
            protocol_hash,
        ),
        "ordered_assignment_sha256": assignment_digest.hexdigest(),
        "row_count": len(rows),
        "fold_values": fold_values,
        "fold_counts": {
            str(fold): sum(
                int(row["fold"]) == fold for row in rows
            )
            for fold in fold_values
        },
        "speaker_count": len(speakers),
        "source_group_count": len(group_to_fold),
    }


def apply_evaluation_protocol(
    artifacts: dict[str, Any],
    protocol: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    artifact_clip_ids = artifacts["clip_ids"]
    artifact_index = {
        str(clip_id): index
        for index, clip_id in enumerate(artifact_clip_ids)
    }
    protocol_rows = protocol["rows"]
    missing = [
        str(row["clip_id"])
        for row in protocol_rows
        if str(row["clip_id"]) not in artifact_index
    ]

    if missing:
        raise ValueError(
            f"{model_id} is missing {len(missing)} of "
            f"{len(protocol_rows)} protocol clips. No silent model-specific "
            "intersection is allowed because it would make comparisons "
            "irreproducible. Missing examples: "
            + ", ".join(missing[:10])
        )

    selected_indices = np.asarray(
        [
            artifact_index[str(row["clip_id"])]
            for row in protocol_rows
        ],
        dtype=np.int64,
    )
    selected_labels = artifacts["y"][selected_indices]
    selected_groups = artifacts["source_groups"][selected_indices]
    protocol_labels = np.asarray(
        [str(row["speaker_label"]) for row in protocol_rows],
        dtype=str,
    )
    protocol_groups = np.asarray(
        [str(row["source_group"]) for row in protocol_rows],
        dtype=str,
    )
    protocol_folds = np.asarray(
        [int(row["fold"]) for row in protocol_rows],
        dtype=np.int64,
    )

    label_mismatches = np.flatnonzero(
        selected_labels != protocol_labels
    )

    if len(label_mismatches):
        index = int(label_mismatches[0])
        raise ValueError(
            "Protocol label mismatch for "
            f"{protocol_rows[index]['clip_id']}: "
            f"artifact={selected_labels[index]!r}, "
            f"protocol={protocol_labels[index]!r}"
        )

    group_mismatches = np.flatnonzero(
        selected_groups != protocol_groups
    )

    if len(group_mismatches):
        index = int(group_mismatches[0])
        raise ValueError(
            "Protocol source_group mismatch for "
            f"{protocol_rows[index]['clip_id']}: "
            f"manifest={selected_groups[index]!r}, "
            f"protocol={protocol_groups[index]!r}"
        )

    unique_folds = np.unique(protocol_folds)

    for fold in unique_folds:
        test_mask = protocol_folds == fold
        train_mask = ~test_mask
        train_groups = set(protocol_groups[train_mask].tolist())
        test_groups = set(protocol_groups[test_mask].tolist())
        overlap = train_groups & test_groups

        if overlap:
            raise ValueError(
                f"Protocol fold {int(fold)} has source_group leakage: "
                + ", ".join(sorted(overlap)[:10])
            )

        train_labels = set(protocol_labels[train_mask].tolist())
        test_labels = set(protocol_labels[test_mask].tolist())
        missing_train = sorted(test_labels - train_labels)

        if missing_train:
            raise ValueError(
                f"Protocol fold {int(fold)} has test-only speakers: "
                + ", ".join(missing_train[:10])
            )

    selected = dict(artifacts)
    selected.update(
        {
            "X": artifacts["X"][selected_indices],
            "y": protocol_labels,
            "clip_ids": np.asarray(
                [str(row["clip_id"]) for row in protocol_rows],
                dtype=str,
            ),
            "folds": protocol_folds,
            "source_groups": protocol_groups,
            "unique_folds": unique_folds,
            "protocol": {
                key: value
                for key, value in protocol.items()
                if key != "rows"
            },
            "available_artifact_samples": len(artifact_clip_ids),
            "excluded_artifact_samples": (
                len(artifact_clip_ids) - len(selected_indices)
            ),
        }
    )
    return selected


def validate_cross_model_alignment(
    embeddings_root: Path,
    model_ids: list[str],
    tag: str,
    evaluation_protocol: dict[str, Any] | None = None,
) -> None:
    """Require an identical ordered evaluation population for every model."""

    reference_model_id: str | None = None
    reference_clip_ids: np.ndarray | None = None
    reference_labels: np.ndarray | None = None
    reference_folds: np.ndarray | None = None
    reference_manifest_hash: str | None = None

    for model_id in model_ids:
        artifact_directory = embeddings_root / model_id / tag
        artifacts = load_and_validate_artifacts(
            artifact_directory,
            model_id,
        )

        if evaluation_protocol is not None:
            artifacts = apply_evaluation_protocol(
                artifacts,
                evaluation_protocol,
                model_id,
            )

        clip_ids = artifacts["clip_ids"]
        labels = artifacts["y"]
        folds = artifacts["folds"]
        manifest_hash = str(
            artifacts["metadata"].get("manifest_sha256", "")
        ).lower()

        if reference_model_id is None:
            reference_model_id = model_id
            reference_clip_ids = clip_ids
            reference_labels = labels
            reference_folds = folds
            reference_manifest_hash = manifest_hash
            continue

        assert reference_clip_ids is not None
        assert reference_labels is not None
        assert reference_folds is not None

        differences: list[str] = []

        if manifest_hash != reference_manifest_hash:
            differences.append("manifest hash")

        if not np.array_equal(clip_ids, reference_clip_ids):
            differences.append(
                f"clip IDs/order ({len(clip_ids)} vs "
                f"{len(reference_clip_ids)} samples)"
            )

        if not np.array_equal(labels, reference_labels):
            differences.append("speaker labels/order")

        if not np.array_equal(folds, reference_folds):
            differences.append("evaluation fold assignments/order")

        if differences:
            raise ValueError(
                "Multi-model comparison requires identical samples and "
                "folds for every model. "
                f"{model_id} differs from {reference_model_id} in: "
                + ", ".join(differences)
                + ". Create one explicit common-success protocol instead "
                "of using a model-dependent runtime intersection."
            )


def parse_gamma(value: str) -> str | float:
    normalized = str(value).strip().lower()

    if normalized in {"scale", "auto"}:
        return normalized

    try:
        numeric = float(normalized)
    except ValueError as error:
        raise ValueError(
            "--gamma must be scale, auto, or a positive number."
        ) from error

    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError("--gamma must be positive.")

    return numeric


def make_classifier(
    classifier_name: str,
    seed: int,
    c_value: float,
    gamma: str | float,
    max_iter: int,
    knn_neighbors: int = DEFAULT_KNN_NEIGHBORS,
    forest_estimators: int = DEFAULT_FOREST_ESTIMATORS,
    tree_max_depth: int | None = None,
    xgb_estimators: int = DEFAULT_XGB_ESTIMATORS,
    xgb_max_depth: int = DEFAULT_XGB_MAX_DEPTH,
    xgb_learning_rate: float = DEFAULT_XGB_LEARNING_RATE,
):
    if c_value <= 0 or not math.isfinite(c_value):
        raise ValueError("--c must be a positive finite number.")

    if max_iter < 1:
        raise ValueError("--max-iter must be at least one.")

    if classifier_name == "linear_svm":
        from sklearn.svm import SVC

        return SVC(
            C=c_value,
            kernel="linear",
            probability=False,
            decision_function_shape="ovr",
            random_state=seed,
            cache_size=2048,
        )

    if classifier_name == "rbf_svm":
        from sklearn.svm import SVC

        return SVC(
            C=c_value,
            kernel="rbf",
            gamma=gamma,
            probability=False,
            decision_function_shape="ovr",
            random_state=seed,
            cache_size=2048,
        )

    if classifier_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=seed,
            n_jobs=1,
        )

    if classifier_name == "cosine_centroid":
        return CosineCentroidClassifier()

    if classifier_name == "knn":
        from sklearn.neighbors import KNeighborsClassifier

        if knn_neighbors < 1:
            raise ValueError(
                "--knn-neighbors must be at least one."
            )

        return KNeighborsClassifier(
            n_neighbors=knn_neighbors,
            weights="uniform",
            algorithm="auto",
            leaf_size=30,
            p=2,
            metric="minkowski",
            n_jobs=1,
        )

    if classifier_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        if forest_estimators < 1:
            raise ValueError(
                "--forest-estimators must be at least one."
            )

        return RandomForestClassifier(
            n_estimators=forest_estimators,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=1,
            random_state=seed,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None,
        )

    if classifier_name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        if tree_max_depth is not None and tree_max_depth < 1:
            raise ValueError(
                "--tree-max-depth must be at least one when supplied."
            )

        return DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=tree_max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            random_state=seed,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0,
        )

    if classifier_name == "xgboost":
        ensure_xgboost_available()

        if xgb_estimators < 1:
            raise ValueError(
                "--xgb-estimators must be at least one."
            )

        if xgb_max_depth < 1:
            raise ValueError(
                "--xgb-max-depth must be at least one."
            )

        if (
            not math.isfinite(xgb_learning_rate)
            or xgb_learning_rate <= 0
        ):
            raise ValueError(
                "--xgb-learning-rate must be a positive finite number."
            )

        return StringLabelXGBClassifier(
            n_estimators=xgb_estimators,
            max_depth=xgb_max_depth,
            learning_rate=xgb_learning_rate,
            random_state=seed,
        )

    raise ValueError(f"Unknown classifier: {classifier_name}")


def ensure_xgboost_available() -> None:
    if importlib.util.find_spec("xgboost") is None:
        raise ModuleNotFoundError(
            "The optional 'xgboost' package is not installed in this "
            "Python environment. Install a project-approved XGBoost "
            "version before selecting --classifier xgboost. The research "
            "runner does not install dependencies automatically."
        )


def optional_package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


class StringLabelXGBClassifier:
    """Deterministic CPU XGBoost wrapper for string speaker labels."""

    def __init__(
        self,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
        random_state: int,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        self.classes_: np.ndarray | None = None
        self.model_: object | None = None

    def get_params(self, deep: bool = True) -> dict[str, object]:
        del deep
        return {
            "objective": "multi:softprob",
            "num_class": (
                int(len(self.classes_))
                if self.classes_ is not None
                else None
            ),
            "eval_metric": "mlogloss",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": self.random_state,
            "verbosity": 0,
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "StringLabelXGBClassifier":
        ensure_xgboost_available()
        from xgboost import XGBClassifier

        labels = np.asarray(y).astype(str)
        classes = np.unique(labels)

        if len(classes) < 2:
            raise ValueError(
                "XGBoost requires at least two speaker classes."
            )

        label_to_index = {
            label: index
            for index, label in enumerate(classes)
        }
        encoded_labels = np.asarray(
            [label_to_index[label] for label in labels],
            dtype=np.int64,
        )
        self.classes_ = classes
        parameters = self.get_params()
        self.model_ = XGBClassifier(
            objective=parameters["objective"],
            num_class=parameters["num_class"],
            eval_metric=parameters["eval_metric"],
            n_estimators=parameters["n_estimators"],
            max_depth=parameters["max_depth"],
            learning_rate=parameters["learning_rate"],
            min_child_weight=parameters["min_child_weight"],
            gamma=parameters["gamma"],
            subsample=parameters["subsample"],
            colsample_bytree=parameters["colsample_bytree"],
            reg_alpha=parameters["reg_alpha"],
            reg_lambda=parameters["reg_lambda"],
            tree_method=parameters["tree_method"],
            n_jobs=parameters["n_jobs"],
            random_state=parameters["random_state"],
            verbosity=parameters["verbosity"],
        )
        self.model_.fit(X, encoded_labels)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("Classifier has not been fitted.")

        probabilities = np.asarray(
            self.model_.predict_proba(X),
            dtype=np.float64,
        )

        if probabilities.ndim != 2:
            raise RuntimeError(
                "XGBoost returned a non-matrix probability output."
            )

        expected_shape = (len(X), len(self.classes_))

        if probabilities.shape != expected_shape:
            raise RuntimeError(
                "XGBoost probability shape does not match the encoded "
                f"speaker classes: {probabilities.shape} != "
                f"{expected_shape}."
            )

        if not np.isfinite(probabilities).all():
            raise RuntimeError(
                "XGBoost returned NaN or infinite probabilities."
            )

        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Classifier has not been fitted.")

        probabilities = self.predict_proba(X)
        encoded_predictions = np.argmax(
            probabilities,
            axis=1,
        )
        return self.classes_[encoded_predictions]


class CosineCentroidClassifier:
    """Nearest class centroid using cosine similarity."""

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None

    def get_params(self, deep: bool = True) -> dict[str, object]:
        del deep
        return {
            "metric": "cosine_similarity",
            "class_prototype": "mean_normalized_training_embedding",
            "input_l2_normalization": True,
            "centroid_l2_normalization": True,
        }

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix, dtype=np.float64),
            where=norms > 0,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "CosineCentroidClassifier":
        classes = np.unique(y)

        if len(classes) < 2:
            raise ValueError(
                "Cosine centroid requires at least two classes."
            )

        normalized_X = self._normalize(X)
        centroids = np.vstack(
            [
                normalized_X[y == label].mean(axis=0)
                for label in classes
            ]
        )
        self.classes_ = classes
        self.centroids_ = self._normalize(centroids)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("Classifier has not been fitted.")

        return self._normalize(X) @ self.centroids_.T

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Classifier has not been fitted.")

        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]


def json_safe_parameter(value: object) -> object:
    if isinstance(value, np.generic):
        return json_safe_parameter(value.item())

    if isinstance(value, dict):
        return {
            str(key): json_safe_parameter(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }

    if isinstance(value, (list, tuple)):
        return [json_safe_parameter(item) for item in value]

    if isinstance(value, float) and not math.isfinite(value):
        return str(value)

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    return repr(value)


def classifier_parameters(classifier: object) -> dict[str, object]:
    if not hasattr(classifier, "get_params"):
        raise TypeError(
            "Classifier does not expose reproducible parameters."
        )

    parameters = classifier.get_params(deep=False)

    if not isinstance(parameters, dict):
        raise TypeError(
            "Classifier get_params() did not return a dictionary."
        )

    return {
        str(key): json_safe_parameter(value)
        for key, value in sorted(parameters.items())
    }


def prediction_strength(
    classifier: object,
    X_test: np.ndarray,
) -> tuple[np.ndarray, str]:
    if hasattr(classifier, "predict_proba"):
        probabilities = np.asarray(
            classifier.predict_proba(X_test),
            dtype=np.float64,
        )

        if (
            probabilities.ndim != 2
            or probabilities.shape[0] != len(X_test)
            or probabilities.shape[1] < 2
        ):
            raise RuntimeError(
                "Classifier returned an invalid predict_proba shape: "
                f"{probabilities.shape}."
            )

        if not np.isfinite(probabilities).all():
            raise RuntimeError(
                "Classifier returned non-finite probabilities."
            )

        return probabilities.max(axis=1), "maximum_class_probability"

    if hasattr(classifier, "decision_function"):
        scores = np.asarray(
            classifier.decision_function(X_test),
            dtype=np.float64,
        )

        if scores.ndim == 1:
            if scores.shape[0] != len(X_test):
                raise RuntimeError(
                    "Classifier returned an invalid decision score shape: "
                    f"{scores.shape}."
                )
            values = np.abs(scores)
        elif scores.ndim == 2 and scores.shape[0] == len(X_test):
            values = scores.max(axis=1)
        else:
            raise RuntimeError(
                "Classifier returned an invalid decision score shape: "
                f"{scores.shape}."
            )

        if not np.isfinite(values).all():
            raise RuntimeError(
                "Classifier returned non-finite decision scores."
            )

        return values, "maximum_decision_score"

    return (
        np.full(len(X_test), np.nan, dtype=np.float64),
        "unavailable",
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_weighted": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def student_t_critical_975(degrees_of_freedom: int) -> float:
    """Two-sided 95% Student-t critical value without a SciPy dependency."""

    table = {
        1: 12.706205,
        2: 4.302653,
        3: 3.182446,
        4: 2.776445,
        5: 2.570582,
        6: 2.446912,
        7: 2.364624,
        8: 2.306004,
        9: 2.262157,
        10: 2.228139,
        11: 2.200985,
        12: 2.178813,
        13: 2.160369,
        14: 2.144787,
        15: 2.13145,
        16: 2.119905,
        17: 2.109816,
        18: 2.100922,
        19: 2.093024,
        20: 2.085963,
        21: 2.079614,
        22: 2.073873,
        23: 2.068658,
        24: 2.063899,
        25: 2.059539,
        26: 2.055529,
        27: 2.051831,
        28: 2.048407,
        29: 2.04523,
        30: 2.042272,
    }

    if degrees_of_freedom < 1:
        return 0.0

    return table.get(degrees_of_freedom, 1.959964)


def summarize_values(values: list[float]) -> dict[str, float | int]:
    finite_values = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]
    count = len(finite_values)

    if count == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "ci95_half_width": 0.0,
            "ci95_lower": 0.0,
            "ci95_upper": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
        }

    mean = statistics.fmean(finite_values)
    std = statistics.stdev(finite_values) if count > 1 else 0.0
    critical = student_t_critical_975(count - 1)
    half_width = (
        critical * std / math.sqrt(count)
        if count > 1
        else 0.0
    )

    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci95_half_width": half_width,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
        "minimum": min(finite_values),
        "maximum": max(finite_values),
    }


def confusion_rows(
    matrix: np.ndarray,
    class_labels: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for label, counts in zip(
        class_labels,
        matrix,
        strict=True,
    ):
        row: dict[str, object] = {
            "true_label": str(label),
        }
        row.update(
            {
                f"predicted::{predicted_label}": int(count)
                for predicted_label, count in zip(
                    class_labels,
                    counts,
                    strict=True,
                )
            }
        )
        rows.append(row)

    return rows


def save_confusion_matrix(
    output_directory: Path,
    matrix: np.ndarray,
    class_labels: np.ndarray,
) -> None:
    atomic_npy(
        output_directory / "confusion_matrix.npy",
        np.asarray(matrix, dtype=np.int64),
    )
    fields = ["true_label"] + [
        f"predicted::{label}"
        for label in class_labels
    ]
    write_csv(
        output_directory / "confusion_matrix.csv",
        fields,
        confusion_rows(matrix, class_labels),
    )


def evaluate_model_classifier(
    model_id: str,
    classifier_name: str,
    artifact_directory: Path,
    output_directory: Path,
    seed: int,
    c_value: float,
    gamma: str | float,
    max_iter: int,
    overwrite: bool,
    evaluation_protocol: dict[str, Any] | None = None,
    knn_neighbors: int = DEFAULT_KNN_NEIGHBORS,
    forest_estimators: int = DEFAULT_FOREST_ESTIMATORS,
    tree_max_depth: int | None = None,
    xgb_estimators: int = DEFAULT_XGB_ESTIMATORS,
    xgb_max_depth: int = DEFAULT_XGB_MAX_DEPTH,
    xgb_learning_rate: float = DEFAULT_XGB_LEARNING_RATE,
) -> dict[str, Any]:
    from sklearn import __version__ as sklearn_version
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import StandardScaler

    if classifier_name == "xgboost":
        ensure_xgboost_available()

    artifacts = load_and_validate_artifacts(
        artifact_directory,
        model_id,
    )

    if evaluation_protocol is not None:
        artifacts = apply_evaluation_protocol(
            artifacts,
            evaluation_protocol,
            model_id,
        )

    X = artifacts["X"]
    y = artifacts["y"]
    clip_ids = artifacts["clip_ids"]
    folds = artifacts["folds"]
    source_groups = artifacts["source_groups"]
    unique_folds = artifacts["unique_folds"]
    class_labels = np.unique(y)

    if len(class_labels) < 2:
        raise ValueError(
            "Cross-validation requires at least two speakers."
        )

    prepare_output(output_directory, overwrite)
    atomic_npy(
        output_directory / "class_labels.npy",
        class_labels.astype(str),
    )

    fold_results: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    aggregate_confusion = np.zeros(
        (len(class_labels), len(class_labels)),
        dtype=np.int64,
    )
    tested_indices: list[int] = []
    run_started = time.perf_counter()
    recorded_classifier_parameters: dict[str, object] | None = None

    print()
    print("=" * 72)
    print("LEAKAGE-SAFE BASELINE CROSS-VALIDATION")
    print("=" * 72)
    print(f"Model      : {get_model_config(model_id)['display_name']}")
    print(f"Classifier : {classifier_name}")

    if evaluation_protocol is not None:
        print(
            "Protocol   : "
            f"{evaluation_protocol['protocol_id']}"
        )

    print(f"Samples    : {len(X)}")
    print(f"Speakers   : {len(class_labels)}")
    print(
        "Folds      : "
        + ", ".join(str(value) for value in unique_folds)
    )
    print(f"Output     : {output_directory}")
    print("=" * 72)

    for fold_value in unique_folds:
        fold_started = time.perf_counter()
        test_mask = folds == fold_value
        train_mask = ~test_mask
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)

        if len(train_indices) == 0 or len(test_indices) == 0:
            raise ValueError(
                f"Fold {fold_value} has an empty train/test partition."
            )

        train_clip_ids = set(clip_ids[train_mask].tolist())
        test_clip_ids = set(clip_ids[test_mask].tolist())
        clip_overlap = train_clip_ids & test_clip_ids

        if clip_overlap:
            raise RuntimeError(
                f"Fold {fold_value} has clip leakage: "
                + ", ".join(sorted(clip_overlap)[:10])
            )

        train_labels = set(y[train_mask].tolist())
        test_labels = set(y[test_mask].tolist())
        missing_train_labels = sorted(test_labels - train_labels)

        if missing_train_labels:
            raise ValueError(
                f"Fold {fold_value} has test-only speakers: "
                + ", ".join(missing_train_labels)
            )

        train_groups = set(source_groups[train_mask].tolist())
        test_groups = set(source_groups[test_mask].tolist())
        shared_source_groups = train_groups & test_groups

        if evaluation_protocol is not None and shared_source_groups:
            raise RuntimeError(
                f"Protocol fold {int(fold_value)} violates chapter "
                "separation. Shared source groups: "
                + ", ".join(sorted(shared_source_groups)[:10])
            )

        scaler = StandardScaler(
            copy=True,
            with_mean=True,
            with_std=True,
        )
        scaler_fit_started = time.perf_counter()
        scaler.fit(X[train_mask])
        scaler_fit_seconds = (
            time.perf_counter() - scaler_fit_started
        )
        scaler_transform_started = time.perf_counter()
        X_train = scaler.transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])
        scaler_transform_seconds = (
            time.perf_counter() - scaler_transform_started
        )

        if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
            raise RuntimeError(
                f"Non-finite scaled values in fold {fold_value}."
            )

        classifier = make_classifier(
            classifier_name,
            seed=seed,
            c_value=c_value,
            gamma=gamma,
            max_iter=max_iter,
            knn_neighbors=knn_neighbors,
            forest_estimators=forest_estimators,
            tree_max_depth=tree_max_depth,
            xgb_estimators=xgb_estimators,
            xgb_max_depth=xgb_max_depth,
            xgb_learning_rate=xgb_learning_rate,
        )
        fit_started = time.perf_counter()
        classifier.fit(X_train, y[train_mask])
        classifier_fit_seconds = time.perf_counter() - fit_started
        current_classifier_parameters = classifier_parameters(
            classifier
        )

        if recorded_classifier_parameters is None:
            recorded_classifier_parameters = (
                current_classifier_parameters
            )
        elif (
            current_classifier_parameters
            != recorded_classifier_parameters
        ):
            raise RuntimeError(
                "Classifier parameters changed between folds."
            )

        prediction_started = time.perf_counter()
        y_pred = np.asarray(classifier.predict(X_test)).astype(str)
        prediction_seconds = (
            time.perf_counter() - prediction_started
        )
        scores, score_type = prediction_strength(
            classifier,
            X_test,
        )

        metrics = compute_metrics(y[test_mask], y_pred)
        matrix = confusion_matrix(
            y[test_mask],
            y_pred,
            labels=class_labels,
        ).astype(np.int64)
        aggregate_confusion += matrix
        tested_indices.extend(test_indices.tolist())
        fold_total_seconds = time.perf_counter() - fold_started
        timing = {
            "scaler_fit_seconds": scaler_fit_seconds,
            "scaler_transform_seconds": scaler_transform_seconds,
            "classifier_fit_seconds": classifier_fit_seconds,
            "prediction_seconds": prediction_seconds,
            "prediction_ms_per_sample": (
                1000.0 * prediction_seconds / len(test_indices)
            ),
            "fold_total_seconds": fold_total_seconds,
        }
        fold_directory = (
            output_directory
            / f"fold_{int(fold_value):02d}"
        )
        fold_directory.mkdir(parents=True, exist_ok=False)
        fold_prediction_rows: list[dict[str, object]] = []

        for local_index, sample_index in enumerate(test_indices):
            prediction_row = {
                "fold": int(fold_value),
                "sample_index": int(sample_index),
                "clip_id": str(clip_ids[sample_index]),
                "source_group": str(source_groups[sample_index]),
                "true_label": str(y[sample_index]),
                "predicted_label": str(y_pred[local_index]),
                "correct": int(
                    str(y[sample_index]) == str(y_pred[local_index])
                ),
                "prediction_score": float(scores[local_index]),
                "score_type": score_type,
            }
            fold_prediction_rows.append(prediction_row)
            all_prediction_rows.append(prediction_row)

        prediction_fields = [
            "fold",
            "sample_index",
            "clip_id",
            "source_group",
            "true_label",
            "predicted_label",
            "correct",
            "prediction_score",
            "score_type",
        ]
        write_csv(
            fold_directory / "predictions.csv",
            prediction_fields,
            fold_prediction_rows,
        )
        save_confusion_matrix(
            fold_directory,
            matrix,
            class_labels,
        )
        atomic_npy(
            fold_directory / "scaler_mean.npy",
            np.asarray(scaler.mean_, dtype=np.float64),
        )
        atomic_npy(
            fold_directory / "scaler_scale.npy",
            np.asarray(scaler.scale_, dtype=np.float64),
        )
        fold_result = {
            "fold": int(fold_value),
            "train_samples": int(len(train_indices)),
            "test_samples": int(len(test_indices)),
            "train_speakers": int(len(train_labels)),
            "test_speakers": int(len(test_labels)),
            "metrics": metrics,
            "timing": timing,
            "score_type": score_type,
            "leakage_audit": {
                "clip_overlap_count": len(clip_overlap),
                "source_group_overlap_count": len(
                    shared_source_groups
                ),
                "test_only_speaker_count": len(
                    missing_train_labels
                ),
                "scaler_fit_samples": int(len(train_indices)),
                "scaler_fit_partition": "train_only",
            },
        }
        atomic_json(
            fold_directory / "metrics.json",
            fold_result,
        )
        fold_results.append(fold_result)
        leakage_rows.append(
            {
                "fold": int(fold_value),
                "train_samples": len(train_indices),
                "test_samples": len(test_indices),
                "clip_overlap_count": len(clip_overlap),
                "source_group_overlap_count": len(
                    shared_source_groups
                ),
                "test_only_speaker_count": len(
                    missing_train_labels
                ),
                "scaler_fit_samples": len(train_indices),
                "scaler_fit_partition": "train_only",
            }
        )

        print(
            f"Fold {int(fold_value):>2}: "
            f"accuracy={metrics['accuracy'] * 100:6.2f}%  "
            f"macro-F1={metrics['f1_macro']:.4f}  "
            f"train={len(train_indices)} test={len(test_indices)}"
        )

    tested_counter = Counter(tested_indices)
    missing_test_indices = sorted(
        set(range(len(X))) - set(tested_counter)
    )
    repeated_test_indices = sorted(
        index
        for index, count in tested_counter.items()
        if count != 1
    )

    if missing_test_indices or repeated_test_indices:
        raise RuntimeError(
            "Persisted folds do not test every sample exactly once."
        )

    all_prediction_rows.sort(
        key=lambda row: int(row["sample_index"])
    )
    write_csv(
        output_directory / "all_predictions.csv",
        [
            "fold",
            "sample_index",
            "clip_id",
            "source_group",
            "true_label",
            "predicted_label",
            "correct",
            "prediction_score",
            "score_type",
        ],
        all_prediction_rows,
    )
    save_confusion_matrix(
        output_directory,
        aggregate_confusion,
        class_labels,
    )
    write_csv(
        output_directory / "leakage_audit.csv",
        [
            "fold",
            "train_samples",
            "test_samples",
            "clip_overlap_count",
            "source_group_overlap_count",
            "test_only_speaker_count",
            "scaler_fit_samples",
            "scaler_fit_partition",
        ],
        leakage_rows,
    )

    flat_fold_rows: list[dict[str, object]] = []

    for fold_result in fold_results:
        row: dict[str, object] = {
            "fold": fold_result["fold"],
            "train_samples": fold_result["train_samples"],
            "test_samples": fold_result["test_samples"],
        }
        row.update(fold_result["metrics"])
        row.update(fold_result["timing"])
        flat_fold_rows.append(row)

    write_csv(
        output_directory / "fold_metrics.csv",
        [
            "fold",
            "train_samples",
            "test_samples",
            *METRIC_ORDER,
            *TIMING_ORDER,
        ],
        flat_fold_rows,
    )
    aggregate: dict[str, dict[str, float | int]] = {}
    aggregate_rows: list[dict[str, object]] = []

    for category, keys in (
        ("metric", METRIC_ORDER),
        ("timing", TIMING_ORDER),
    ):
        for key in keys:
            summary = summarize_values(
                [
                    float(fold_result[
                        "metrics" if category == "metric" else "timing"
                    ][key])
                    for fold_result in fold_results
                ]
            )
            aggregate[key] = summary
            aggregate_rows.append(
                {
                    "category": category,
                    "name": key,
                    **summary,
                }
            )

    write_csv(
        output_directory / "aggregate_metrics.csv",
        [
            "category",
            "name",
            "n",
            "mean",
            "std",
            "ci95_half_width",
            "ci95_lower",
            "ci95_upper",
            "minimum",
            "maximum",
        ],
        aggregate_rows,
    )
    aggregate_payload = {
        "confidence_interval": (
            "two-sided 95% Student-t interval over fold values"
        ),
        "confidence_interval_caution": (
            "Cross-validation folds are correlated, so this is a "
            "descriptive fold-level interval rather than an independent "
            "population interval. Intervals based on fewer than three "
            "folds are especially unstable and should not be used as a "
            "primary paper result."
        ),
        "values": aggregate,
    }
    atomic_json(
        output_directory / "aggregate_metrics.json",
        aggregate_payload,
    )

    if evaluation_protocol is None:
        protocol_metadata = {
            "fold_source": "persisted folds.npy",
            "fold_values": [
                int(value)
                for value in unique_folds
            ],
            "number_of_folds": len(unique_folds),
            "samples": len(X),
            "speakers": len(class_labels),
            "embedding_dimension": int(X.shape[1]),
            "each_sample_tested_once": True,
            "exact_clip_leakage_detected": False,
            "source_group_overlap_is_reported_not_blocked": True,
        }
        experiment_name = "fixed_fold_speaker_identification_baseline"
    else:
        protocol_metadata = {
            "mode": "external_protocol_file",
            "protocol_id": evaluation_protocol["protocol_id"],
            "path": evaluation_protocol["path"],
            "sha256": evaluation_protocol["sha256"],
            "ordered_assignment_sha256": (
                evaluation_protocol["ordered_assignment_sha256"]
            ),
            "row_count": evaluation_protocol["row_count"],
            "fold_source": "external protocol CSV",
            "fold_values": [
                int(value)
                for value in unique_folds
            ],
            "fold_counts": evaluation_protocol["fold_counts"],
            "number_of_folds": len(unique_folds),
            "samples": len(X),
            "speakers": len(class_labels),
            "source_groups": evaluation_protocol[
                "source_group_count"
            ],
            "embedding_dimension": int(X.shape[1]),
            "selection_policy": "exact_protocol_clip_ids",
            "artifact_persisted_folds_overridden": True,
            "source_group_disjoint_enforced": True,
            "each_sample_tested_once": True,
            "exact_clip_leakage_detected": False,
            "source_group_overlap_is_reported_not_blocked": False,
            "available_artifact_samples": artifacts[
                "available_artifact_samples"
            ],
            "excluded_artifact_samples": artifacts[
                "excluded_artifact_samples"
            ],
        }
        experiment_name = (
            "external_protocol_speaker_identification_sensitivity"
        )

    run_metadata = {
        "schema_version": 2 if evaluation_protocol is not None else 1,
        "experiment": experiment_name,
        "completed_at_utc": utc_now(),
        "model_id": model_id,
        "model": {
            key: get_model_config(model_id).get(key)
            for key in (
                "display_name",
                "provider",
                "architecture",
                "pretrained_name",
                "source",
            )
        },
        "classifier": {
            "name": classifier_name,
            "standard_scaler": {
                "copy": True,
                "with_mean": True,
                "with_std": True,
                "fit_partition": "training_fold_only",
            },
            "hyperparameters": (
                recorded_classifier_parameters or {}
            ),
            # Retained for backward compatibility with earlier metadata.
            "c": c_value,
            "gamma": gamma,
            "max_iter": max_iter,
            "seed": seed,
        },
        "protocol": protocol_metadata,
        "embedding_artifacts": {
            "directory": str(artifact_directory.resolve()),
            "source_metadata": artifacts["metadata"],
            "sha256": artifacts["artifact_hashes"],
        },
        "manifest": artifacts["manifest"],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn_version,
            "xgboost": optional_package_version("xgboost"),
            "platform": platform.platform(),
        },
        "wall_seconds": time.perf_counter() - run_started,
        "output_directory": str(output_directory.resolve()),
    }
    atomic_json(
        output_directory / "run_metadata.json",
        run_metadata,
    )

    print("-" * 72)
    accuracy_summary = aggregate["accuracy"]
    print(
        "Accuracy   : "
        f"{float(accuracy_summary['mean']) * 100:.2f}% "
        f"± {float(accuracy_summary['std']) * 100:.2f}% SD"
    )
    if int(accuracy_summary["n"]) >= 3:
        print(
            "95% CI     : "
            f"[{float(accuracy_summary['ci95_lower']) * 100:.2f}%, "
            f"{float(accuracy_summary['ci95_upper']) * 100:.2f}%]"
        )
    else:
        print(
            "95% CI     : not reported (<3 fold values; "
            "mean and SD are descriptive)"
        )
    print(f"Artifacts  : {output_directory}")
    print("=" * 72)

    return {
        "model_id": model_id,
        "classifier": classifier_name,
        "output_directory": str(output_directory),
        "aggregate": aggregate,
    }


def main() -> int:
    arguments = parse_arguments()
    gamma = parse_gamma(arguments.gamma)
    evaluation_protocol = (
        load_evaluation_protocol(arguments.protocol_file)
        if arguments.protocol_file is not None
        else None
    )
    model_ids = (
        list(MODEL_CONFIGS)
        if arguments.all_models
        else [arguments.model]
    )
    classifier_names = (
        list(CLASSIFIERS)
        if arguments.all_classifiers
        else [arguments.classifier]
    )

    if len(model_ids) > 1:
        validate_cross_model_alignment(
            arguments.embeddings_root.resolve(),
            model_ids,
            arguments.tag,
            evaluation_protocol,
        )

    failures: list[tuple[str, str, str]] = []

    for model_id in model_ids:
        artifact_directory = (
            arguments.embeddings_root.resolve()
            / model_id
            / arguments.tag
        )

        for classifier_name in classifier_names:
            if evaluation_protocol is None:
                output_directory = (
                    arguments.output_root.resolve()
                    / model_id
                    / classifier_name
                    / arguments.tag
                )
            else:
                output_directory = (
                    arguments.output_root.resolve()
                    / "protocols"
                    / str(evaluation_protocol["protocol_id"])
                    / model_id
                    / classifier_name
                    / arguments.tag
                )

            try:
                evaluate_model_classifier(
                    model_id=model_id,
                    classifier_name=classifier_name,
                    artifact_directory=artifact_directory,
                    output_directory=output_directory,
                    seed=arguments.seed,
                    c_value=arguments.c,
                    gamma=gamma,
                    max_iter=arguments.max_iter,
                    overwrite=arguments.overwrite,
                    evaluation_protocol=evaluation_protocol,
                    knn_neighbors=arguments.knn_neighbors,
                    forest_estimators=arguments.forest_estimators,
                    tree_max_depth=arguments.tree_max_depth,
                    xgb_estimators=arguments.xgb_estimators,
                    xgb_max_depth=arguments.xgb_max_depth,
                    xgb_learning_rate=(
                        arguments.xgb_learning_rate
                    ),
                )
            except Exception as error:
                failures.append(
                    (
                        model_id,
                        classifier_name,
                        str(error),
                    )
                )
                print()
                print(
                    f"FAILED: {model_id} / {classifier_name}\n{error}",
                    file=sys.stderr,
                )

    if failures:
        print()
        print("=" * 72, file=sys.stderr)
        print("BASELINE RUN FINISHED WITH FAILURES", file=sys.stderr)
        print("=" * 72, file=sys.stderr)

        for model_id, classifier_name, message in failures:
            print(
                f"- {model_id} / {classifier_name}: {message}",
                file=sys.stderr,
            )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
