"""Summarize leakage-free nested SVM hyperparameter-tuning results.

The nested tuner selects an SVM configuration independently inside every
outer training fold.  This script validates the complete four-model result
set, verifies that the strict chapter-held-out protocol was used everywhere,
and compares the untouched outer-fold scores against the matching fixed
linear- and RBF-SVM baselines.

The strict protocol currently has two outer folds.  Consequently all
fold-level standard deviations and deltas generated here are descriptive.
This module deliberately performs no null-hypothesis significance test and
does not claim statistical significance from two correlated outer folds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from research.summarize_classifier_benchmark import (
        MODEL_LABELS,
        ProtocolContext,
        load_model_ids,
        load_protocol_context,
        read_json,
    )
except ModuleNotFoundError:  # Direct-file execution from research/.
    from summarize_classifier_benchmark import (  # type: ignore[no-redef]
        MODEL_LABELS,
        ProtocolContext,
        load_model_ids,
        load_protocol_context,
        read_json,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TUNING_ROOT = PROJECT_ROOT / "research_results" / "hyperparameter_tuning"
DEFAULT_BASELINE_ROOT = PROJECT_ROOT / "research_results" / "baseline_cv"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "research_baseline.json"

TUNED_RUN_FILES = (
    "run_metadata.json",
    "fold_metrics.csv",
    "aggregate_metrics.json",
    "selected_parameters.csv",
    "leakage_audit.csv",
)
BASELINE_RUN_FILES = (
    "run_metadata.json",
    "fold_metrics.csv",
    "aggregate_metrics.json",
)
DEFAULT_CLASSIFIERS = ("linear_svm", "rbf_svm")
DEFAULT_CLASSIFIER_LABELS = {
    "linear_svm": "Fixed Linear SVM",
    "rbf_svm": "Fixed RBF SVM",
    "tuned_nested_svm": "Nested tuned SVM",
}

PERFORMANCE_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
)
TUNING_TIMING_METRICS = (
    "grid_search_seconds",
    "classifier_fit_seconds",
    "prediction_seconds",
    "prediction_ms_per_sample",
    "fold_total_seconds",
)

MODEL_LEVEL_FIELDS = (
    "model_order",
    "model_id",
    "model_display_name",
    "embedding_dimension",
    "outer_folds",
    "outer_test_samples_total",
    "tuned_accuracy_mean",
    "tuned_accuracy_fold_sd",
    "linear_svm_accuracy_mean",
    "rbf_svm_accuracy_mean",
    "tuned_minus_linear_accuracy",
    "tuned_minus_rbf_accuracy",
    "tuned_f1_macro_mean",
    "tuned_f1_macro_fold_sd",
    "linear_svm_f1_macro_mean",
    "rbf_svm_f1_macro_mean",
    "tuned_minus_linear_f1_macro",
    "tuned_minus_rbf_f1_macro",
    "grid_search_seconds_mean",
    "grid_search_seconds_total",
    "classifier_fit_seconds_mean",
    "classifier_fit_seconds_total",
    "prediction_ms_per_sample_mean",
    "fold_total_seconds_mean",
    "unique_selected_parameter_sets",
)

FOLD_LEVEL_FIELDS = (
    "model_order",
    "model_id",
    "model_display_name",
    "outer_fold",
    "train_samples",
    "test_samples",
    "inner_folds",
    "best_kernel",
    "best_C",
    "best_gamma",
    "best_inner_macro_f1",
    "tuned_accuracy",
    "linear_svm_accuracy",
    "rbf_svm_accuracy",
    "tuned_minus_linear_accuracy",
    "tuned_minus_rbf_accuracy",
    "tuned_f1_macro",
    "linear_svm_f1_macro",
    "rbf_svm_f1_macro",
    "tuned_minus_linear_f1_macro",
    "tuned_minus_rbf_f1_macro",
    "grid_search_seconds",
    "classifier_fit_seconds",
    "prediction_seconds",
    "prediction_ms_per_sample",
    "fold_total_seconds",
)

PARAMETER_FREQUENCY_FIELDS = (
    "scope",
    "model_order",
    "model_id",
    "model_display_name",
    "kernel",
    "C",
    "gamma",
    "selection_count",
    "available_outer_folds",
    "selection_percent",
)

TIMING_FIELDS = (
    "model_order",
    "model_id",
    "model_display_name",
    "outer_folds",
    "grid_search_seconds_mean",
    "grid_search_seconds_total",
    "classifier_fit_seconds_mean",
    "classifier_fit_seconds_total",
    "prediction_seconds_mean",
    "prediction_seconds_total",
    "prediction_ms_per_sample_mean",
    "fold_total_seconds_mean",
    "fold_total_seconds_total",
)


@dataclass
class FoldResult:
    """One validated untouched outer-test-fold result."""

    outer_fold: int
    train_samples: int
    test_samples: int
    inner_folds: int
    best_inner_macro_f1: float
    best_kernel: str
    best_c: float
    best_gamma: str
    metrics: dict[str, float]
    timings: dict[str, float]


@dataclass
class TunedRun:
    """One complete nested-tuning run after structural validation."""

    model_order: int
    model_id: str
    model_display_name: str
    provider: str
    architecture: str
    embedding_dimension: int
    tag: str
    directory: Path
    metadata: dict[str, Any]
    embedding_hashes: dict[str, str]
    manifest_sha256: str
    fold_results: list[FoldResult]


@dataclass
class BaselineRun:
    """Matching fixed SVM baseline used only for paired descriptive deltas."""

    model_id: str
    classifier_id: str
    directory: Path
    embedding_hashes: dict[str, str]
    manifest_sha256: str
    fold_metrics: dict[int, dict[str, float | int]]


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize four-model nested SVM tuning results, "
            "then compare them with strict fixed linear/RBF baselines."
        )
    )
    parser.add_argument(
        "--protocol-file",
        type=Path,
        required=True,
        help="Strict chapter-held-out protocol CSV used by every run.",
    )
    parser.add_argument(
        "--tag",
        default="full",
        help="Embedding/result tag to summarize (default: full).",
    )
    parser.add_argument(
        "--tuning-root",
        type=Path,
        default=DEFAULT_TUNING_ROOT,
        help="Root containing protocol-specific nested tuning results.",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_BASELINE_ROOT,
        help="Root containing strict fixed-classifier baseline results.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Research config declaring the required four-model order.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Exact output directory. Default: "
            "<tuning-root>/protocols/<protocol-id>/summary/svm/<tag>."
        ),
    )
    return parser.parse_args(arguments)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(
            value,
            output_file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        output_file.write("\n")
    os.replace(temporary, path)


def atomic_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(fieldnames),
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def require_files(directory: Path, filenames: Sequence[str], label: str) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory was not found:\n{directory}")

    missing = [name for name in filenames if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{label} is incomplete:\n{directory}\nMissing: "
            + ", ".join(missing)
        )


def require_mapping(
    parent: dict[str, Any],
    key: str,
    source: Path,
) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing object '{key}' in:\n{source}")
    return value


def finite_number(value: object, name: str, source: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric '{name}' in:\n{source}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite '{name}' in:\n{source}")
    return number


def parse_integer(value: str, name: str, source: Path, row_number: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid integer '{name}' at row {row_number} in:\n{source}"
        ) from error
    return number


def parse_csv_float(
    value: str,
    name: str,
    source: Path,
    row_number: int,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid numeric '{name}' at row {row_number} in:\n{source}"
        ) from error
    if not math.isfinite(number):
        raise ValueError(
            f"Non-finite '{name}' at row {row_number} in:\n{source}"
        )
    return number


def canonical_gamma(value: object, kernel: str | None = None) -> str:
    text = str(value).strip()
    if not text:
        if kernel == "linear":
            return "not_applicable"
        raise ValueError("SVM gamma cannot be empty.")
    try:
        numeric = float(text)
    except ValueError:
        return text.lower()
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"Invalid numeric SVM gamma: {text!r}")
    return format(numeric, ".12g")


def canonical_c(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid SVM C value: {value!r}") from error
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"SVM C must be positive and finite, got {value!r}.")
    return number


def display_name(model_id: str, metadata_name: object) -> str:
    if model_id in MODEL_LABELS:
        return MODEL_LABELS[model_id]
    return str(metadata_name or model_id).replace("Ã¢â‚¬â€", "—")


def protocol_expectations(protocol: ProtocolContext) -> dict[str, object]:
    return {
        "mode": "external_protocol_file",
        "protocol_id": protocol.protocol_id,
        "sha256": protocol.sha256,
        "ordered_assignment_sha256": protocol.ordered_assignment_sha256,
        "row_count": protocol.row_count,
        "fold_values": list(protocol.fold_values),
        "fold_counts": protocol.fold_counts,
        "number_of_folds": len(protocol.fold_values),
        "samples": protocol.row_count,
        "speakers": protocol.speakers,
        "source_groups": protocol.source_groups,
        "source_group_disjoint_enforced": True,
        "each_sample_tested_once": True,
        "exact_clip_leakage_detected": False,
    }


def validate_protocol_metadata(
    value: dict[str, Any],
    protocol: ProtocolContext,
    source: Path,
) -> None:
    for key, expected in protocol_expectations(protocol).items():
        observed = value.get(key)
        # Nested-tuning schema version 1 used explicit count suffixes for
        # these two identities.  Accept the aliases while still comparing
        # the exact values from the re-hashed protocol.
        if key == "samples" and observed is None:
            observed = value.get("row_count")
        elif key == "speakers" and observed is None:
            observed = value.get("speaker_count")
        elif key == "source_groups" and observed is None:
            observed = value.get("source_group_count")
        elif key == "exact_clip_leakage_detected" and observed is None:
            # Nested runs record zero overlap per outer fold in the mandatory
            # leakage_audit.csv, which is validated independently below.
            observed = False
        if observed != expected:
            raise ValueError(
                f"Protocol metadata mismatch for '{key}' in:\n{source}\n"
                f"Expected: {expected!r}\nObserved: {observed!r}"
            )


def resolve_tag(metadata: dict[str, Any], source: Path) -> str:
    direct = metadata.get("embedding_tag", metadata.get("tag"))
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    artifacts = require_mapping(metadata, "embedding_artifacts", source)
    source_metadata = require_mapping(artifacts, "source_metadata", source)
    value = source_metadata.get("tag")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing embedding tag in:\n{source}")
    return value.strip()


def load_embedding_identity(
    metadata: dict[str, Any],
    source: Path,
) -> tuple[dict[str, str], str, int]:
    artifacts = require_mapping(metadata, "embedding_artifacts", source)
    hashes = require_mapping(artifacts, "sha256", source)
    required_hashes = ("X.npy", "y.npy", "clip_ids.npy")
    cleaned_hashes: dict[str, str] = {}
    for key in required_hashes:
        value = hashes.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Missing SHA-256 for {key!r} in:\n{source}")
        cleaned_hashes[key] = value.lower()
    for key, value in hashes.items():
        if isinstance(key, str) and isinstance(value, str):
            cleaned_hashes[key] = value.lower()

    source_metadata = require_mapping(artifacts, "source_metadata", source)
    dimension = int(
        finite_number(
            source_metadata.get("embedding_dimension"),
            "embedding_dimension",
            source,
        )
    )
    if dimension < 1:
        raise ValueError(f"Invalid embedding dimension in:\n{source}")

    raw_manifest = metadata.get("manifest")
    if isinstance(raw_manifest, dict):
        manifest = raw_manifest
    else:
        manifest = require_mapping(artifacts, "manifest", source)
    manifest_sha = manifest.get("sha256")
    if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
        raise ValueError(f"Missing manifest SHA-256 in:\n{source}")

    return cleaned_hashes, manifest_sha.lower(), dimension


def load_tuned_fold_metrics(
    path: Path,
    protocol: ProtocolContext,
) -> list[FoldResult]:
    required = {
        "outer_fold",
        "train_samples",
        "test_samples",
        "inner_folds",
        "best_inner_macro_f1",
        "best_kernel",
        "best_C",
        "best_gamma",
        *PERFORMANCE_METRICS,
        *TUNING_TIMING_METRICS,
    }
    rows: list[FoldResult] = []
    seen_folds: set[int] = set()

    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Fold metrics CSV has no header:\n{path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Nested fold metrics are missing columns in:\n{path}\n"
                + ", ".join(sorted(missing))
            )

        for row_number, source_row in enumerate(reader, start=2):
            fold = parse_integer(
                source_row["outer_fold"], "outer_fold", path, row_number
            )
            train_samples = parse_integer(
                source_row["train_samples"], "train_samples", path, row_number
            )
            test_samples = parse_integer(
                source_row["test_samples"], "test_samples", path, row_number
            )
            inner_folds = parse_integer(
                source_row["inner_folds"], "inner_folds", path, row_number
            )

            if fold in seen_folds:
                raise ValueError(f"Duplicate outer fold {fold} in:\n{path}")
            if fold not in protocol.fold_values:
                raise ValueError(f"Unexpected outer fold {fold} in:\n{path}")
            expected_test = protocol.fold_counts[str(fold)]
            expected_train = protocol.row_count - expected_test
            if (train_samples, test_samples) != (expected_train, expected_test):
                raise ValueError(
                    f"Outer-fold {fold} train/test counts disagree with the "
                    f"strict protocol in:\n{path}\nExpected "
                    f"{expected_train}/{expected_test}; observed "
                    f"{train_samples}/{test_samples}."
                )
            if inner_folds < 2:
                raise ValueError(
                    f"Outer-fold {fold} has fewer than two inner folds in:\n{path}"
                )

            kernel = source_row["best_kernel"].strip().lower()
            if kernel not in {"linear", "rbf"}:
                raise ValueError(
                    f"Unsupported selected kernel {kernel!r} in:\n{path}"
                )
            best_c = canonical_c(source_row["best_C"])
            best_gamma = canonical_gamma(source_row["best_gamma"], kernel)
            best_inner = parse_csv_float(
                source_row["best_inner_macro_f1"],
                "best_inner_macro_f1",
                path,
                row_number,
            )
            if not 0.0 <= best_inner <= 1.0:
                raise ValueError(
                    f"best_inner_macro_f1 is outside [0, 1] in:\n{path}"
                )

            metrics: dict[str, float] = {}
            for metric in PERFORMANCE_METRICS:
                value = parse_csv_float(
                    source_row[metric], metric, path, row_number
                )
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"{metric} is outside [0, 1] in:\n{path}"
                    )
                metrics[metric] = value

            timings: dict[str, float] = {}
            for metric in TUNING_TIMING_METRICS:
                value = parse_csv_float(
                    source_row[metric], metric, path, row_number
                )
                if value < 0.0:
                    raise ValueError(f"Negative {metric} in:\n{path}")
                timings[metric] = value

            rows.append(
                FoldResult(
                    outer_fold=fold,
                    train_samples=train_samples,
                    test_samples=test_samples,
                    inner_folds=inner_folds,
                    best_inner_macro_f1=best_inner,
                    best_kernel=kernel,
                    best_c=best_c,
                    best_gamma=best_gamma,
                    metrics=metrics,
                    timings=timings,
                )
            )
            seen_folds.add(fold)

    rows.sort(key=lambda row: row.outer_fold)
    observed = tuple(row.outer_fold for row in rows)
    if observed != protocol.fold_values:
        raise ValueError(
            f"Nested fold results are incomplete in:\n{path}\nExpected "
            f"{list(protocol.fold_values)}, observed {list(observed)}."
        )
    return rows


def validate_selected_parameters(
    path: Path,
    fold_results: list[FoldResult],
) -> None:
    required = {
        "outer_fold",
        "best_kernel",
        "best_C",
        "best_gamma",
        "best_inner_macro_f1",
        "rank_test_score",
    }
    observed: dict[int, tuple[str, float, str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Selected-parameters CSV has no header:\n{path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Selected-parameters CSV is missing columns in:\n{path}\n"
                + ", ".join(sorted(missing))
            )
        for row_number, row in enumerate(reader, start=2):
            fold = parse_integer(
                row["outer_fold"], "outer_fold", path, row_number
            )
            if fold in observed:
                raise ValueError(f"Duplicate selected fold {fold} in:\n{path}")
            rank = parse_integer(
                row["rank_test_score"], "rank_test_score", path, row_number
            )
            if rank != 1:
                raise ValueError(
                    f"Selected parameters must have rank_test_score=1 in:\n{path}"
                )
            kernel = row["best_kernel"].strip().lower()
            observed[fold] = (
                kernel,
                canonical_c(row["best_C"]),
                canonical_gamma(row["best_gamma"], kernel),
                parse_csv_float(
                    row["best_inner_macro_f1"],
                    "best_inner_macro_f1",
                    path,
                    row_number,
                ),
            )

    expected_folds = {row.outer_fold for row in fold_results}
    if set(observed) != expected_folds:
        raise ValueError(
            f"Selected-parameter folds do not match fold metrics in:\n{path}"
        )
    for fold in fold_results:
        selected = observed[fold.outer_fold]
        expected = (
            fold.best_kernel,
            fold.best_c,
            fold.best_gamma,
            fold.best_inner_macro_f1,
        )
        if selected[:3] != expected[:3] or not math.isclose(
            selected[3], expected[3], rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError(
                f"Selected parameters disagree for outer fold "
                f"{fold.outer_fold} in:\n{path}"
            )


def false_text(value: str) -> bool:
    return value.strip().lower() in {"false", "0", "no"}


def validate_leakage_audit(
    path: Path,
    fold_results: list[FoldResult],
) -> None:
    required = {
        "outer_fold",
        "train_samples",
        "test_samples",
        "clip_overlap_count",
        "source_group_overlap_count",
        "test_only_speaker_count",
        "inner_cv_partition",
        "outer_test_used_in_tuning",
        "inner_cv_type",
        "inner_folds",
    }
    observed_folds: set[int] = set()
    expected = {row.outer_fold: row for row in fold_results}

    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Leakage audit has no header:\n{path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Leakage audit is missing columns in:\n{path}\n"
                + ", ".join(sorted(missing))
            )
        for row_number, row in enumerate(reader, start=2):
            fold = parse_integer(
                row["outer_fold"], "outer_fold", path, row_number
            )
            if fold in observed_folds or fold not in expected:
                raise ValueError(
                    f"Unexpected or duplicate leakage-audit fold {fold} in:\n{path}"
                )
            expected_row = expected[fold]
            integer_expectations = {
                "train_samples": expected_row.train_samples,
                "test_samples": expected_row.test_samples,
                "clip_overlap_count": 0,
                "source_group_overlap_count": 0,
                "test_only_speaker_count": 0,
                "inner_folds": expected_row.inner_folds,
            }
            for key, expected_value in integer_expectations.items():
                observed = parse_integer(row[key], key, path, row_number)
                if observed != expected_value:
                    raise ValueError(
                        f"Leakage audit '{key}' mismatch for outer fold "
                        f"{fold} in:\n{path}"
                    )
            if not false_text(row["outer_test_used_in_tuning"]):
                raise ValueError(
                    "Outer test data was used during tuning according to:\n"
                    f"{path}"
                )
            if not row["inner_cv_partition"].strip():
                raise ValueError(f"Missing inner_cv_partition in:\n{path}")
            if not row["inner_cv_type"].strip():
                raise ValueError(f"Missing inner_cv_type in:\n{path}")
            observed_folds.add(fold)

    if observed_folds != set(expected):
        raise ValueError(f"Leakage audit is incomplete in:\n{path}")


def aggregate_values(
    metadata: dict[str, Any],
    source: Path,
) -> dict[str, dict[str, float | int]]:
    raw_values = metadata.get("values")
    if not isinstance(raw_values, dict):
        raise ValueError(f"Missing aggregate 'values' object in:\n{source}")
    output: dict[str, dict[str, float | int]] = {}
    for metric, raw in raw_values.items():
        if not isinstance(metric, str) or not isinstance(raw, dict):
            raise ValueError(f"Invalid aggregate metric in:\n{source}")
        output[metric] = {
            "n": int(finite_number(raw.get("n"), f"{metric}.n", source)),
            "mean": finite_number(raw.get("mean"), f"{metric}.mean", source),
            "std": finite_number(raw.get("std"), f"{metric}.std", source),
        }
    return output


def validate_tuned_aggregates(
    path: Path,
    fold_results: list[FoldResult],
) -> None:
    aggregates = aggregate_values(read_json(path), path)
    value_lookup: dict[str, list[float]] = {
        metric: [row.metrics[metric] for row in fold_results]
        for metric in PERFORMANCE_METRICS
    }
    for metric in TUNING_TIMING_METRICS:
        value_lookup[metric] = [row.timings[metric] for row in fold_results]

    for metric, values in value_lookup.items():
        if metric not in aggregates:
            raise ValueError(f"Missing aggregate metric {metric!r} in:\n{path}")
        expected_mean = statistics.fmean(values)
        expected_sd = statistics.stdev(values) if len(values) > 1 else 0.0
        observed = aggregates[metric]
        if observed["n"] != len(values):
            raise ValueError(f"Aggregate {metric}.n mismatch in:\n{path}")
        if not math.isclose(
            float(observed["mean"]),
            expected_mean,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Aggregate {metric}.mean mismatch in:\n{path}")
        if not math.isclose(
            float(observed["std"]),
            expected_sd,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Aggregate {metric}.std mismatch in:\n{path}")


def load_tuned_run(
    directory: Path,
    model_id: str,
    model_order: int,
    tag: str,
    protocol: ProtocolContext,
) -> TunedRun:
    require_files(directory, TUNED_RUN_FILES, "Nested SVM tuning run")
    metadata_path = directory / "run_metadata.json"
    metadata = read_json(metadata_path)
    if metadata.get("experiment") != "nested_svm_hyperparameter_tuning":
        raise ValueError(f"Unexpected experiment in:\n{metadata_path}")
    if metadata.get("model_id") != model_id:
        raise ValueError(f"Model ID mismatch in:\n{metadata_path}")
    observed_tag = resolve_tag(metadata, metadata_path)
    if observed_tag != tag:
        raise ValueError(
            f"Embedding tag mismatch in {metadata_path}: "
            f"expected {tag!r}, observed {observed_tag!r}."
        )

    protocol_metadata = require_mapping(metadata, "protocol", metadata_path)
    validate_protocol_metadata(protocol_metadata, protocol, metadata_path)
    embedding_hashes, manifest_sha, dimension = load_embedding_identity(
        metadata, metadata_path
    )
    model_metadata = require_mapping(metadata, "model", metadata_path)

    selection_metric = metadata.get("selection_metric")
    if selection_metric is None and isinstance(metadata.get("tuning"), dict):
        selection_metric = metadata["tuning"].get("selection_metric")
    if selection_metric != "f1_macro":
        raise ValueError(
            f"Nested selection metric must be 'f1_macro' in:\n{metadata_path}"
        )

    fold_results = load_tuned_fold_metrics(
        directory / "fold_metrics.csv", protocol
    )
    validate_selected_parameters(
        directory / "selected_parameters.csv", fold_results
    )
    validate_leakage_audit(directory / "leakage_audit.csv", fold_results)
    validate_tuned_aggregates(
        directory / "aggregate_metrics.json", fold_results
    )

    return TunedRun(
        model_order=model_order,
        model_id=model_id,
        model_display_name=display_name(
            model_id, model_metadata.get("display_name")
        ),
        provider=str(model_metadata.get("provider", "")),
        architecture=str(model_metadata.get("architecture", "")),
        embedding_dimension=dimension,
        tag=tag,
        directory=directory,
        metadata=metadata,
        embedding_hashes=embedding_hashes,
        manifest_sha256=manifest_sha,
        fold_results=fold_results,
    )


def load_baseline_fold_metrics(
    path: Path,
    protocol: ProtocolContext,
) -> dict[int, dict[str, float | int]]:
    required = {
        "fold",
        "train_samples",
        "test_samples",
        *PERFORMANCE_METRICS,
    }
    output: dict[int, dict[str, float | int]] = {}
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Baseline fold metrics have no header:\n{path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Baseline fold metrics are missing columns in:\n{path}\n"
                + ", ".join(sorted(missing))
            )
        for row_number, row in enumerate(reader, start=2):
            fold = parse_integer(row["fold"], "fold", path, row_number)
            train = parse_integer(
                row["train_samples"], "train_samples", path, row_number
            )
            test = parse_integer(
                row["test_samples"], "test_samples", path, row_number
            )
            if fold in output or fold not in protocol.fold_values:
                raise ValueError(
                    f"Unexpected or duplicate baseline fold {fold} in:\n{path}"
                )
            expected_test = protocol.fold_counts[str(fold)]
            if test != expected_test or train != protocol.row_count - expected_test:
                raise ValueError(
                    f"Baseline counts mismatch for fold {fold} in:\n{path}"
                )
            item: dict[str, float | int] = {
                "fold": fold,
                "train_samples": train,
                "test_samples": test,
            }
            for metric in PERFORMANCE_METRICS:
                value = parse_csv_float(row[metric], metric, path, row_number)
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"Baseline {metric} is outside [0, 1] in:\n{path}"
                    )
                item[metric] = value
            output[fold] = item
    if tuple(sorted(output)) != protocol.fold_values:
        raise ValueError(f"Baseline folds are incomplete in:\n{path}")
    return output


def validate_baseline_aggregates(
    path: Path,
    fold_metrics: dict[int, dict[str, float | int]],
) -> None:
    aggregates = aggregate_values(read_json(path), path)
    for metric in PERFORMANCE_METRICS:
        values = [float(row[metric]) for _, row in sorted(fold_metrics.items())]
        expected_mean = statistics.fmean(values)
        expected_sd = statistics.stdev(values) if len(values) > 1 else 0.0
        if metric not in aggregates:
            raise ValueError(f"Missing baseline aggregate {metric!r} in:\n{path}")
        observed = aggregates[metric]
        if observed["n"] != len(values):
            raise ValueError(f"Baseline aggregate n mismatch in:\n{path}")
        if not math.isclose(
            float(observed["mean"]),
            expected_mean,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Baseline aggregate mean mismatch in:\n{path}")
        if not math.isclose(
            float(observed["std"]),
            expected_sd,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Baseline aggregate SD mismatch in:\n{path}")


def load_baseline_run(
    directory: Path,
    model_id: str,
    classifier_id: str,
    tag: str,
    protocol: ProtocolContext,
) -> BaselineRun:
    require_files(directory, BASELINE_RUN_FILES, "Strict baseline run")
    metadata_path = directory / "run_metadata.json"
    metadata = read_json(metadata_path)
    if metadata.get("model_id") != model_id:
        raise ValueError(f"Baseline model ID mismatch in:\n{metadata_path}")
    classifier = require_mapping(metadata, "classifier", metadata_path)
    if classifier.get("name") != classifier_id:
        raise ValueError(f"Baseline classifier mismatch in:\n{metadata_path}")
    observed_tag = resolve_tag(metadata, metadata_path)
    if observed_tag != tag:
        raise ValueError(f"Baseline tag mismatch in:\n{metadata_path}")
    validate_protocol_metadata(
        require_mapping(metadata, "protocol", metadata_path),
        protocol,
        metadata_path,
    )
    hashes, manifest_sha, _ = load_embedding_identity(metadata, metadata_path)
    fold_metrics = load_baseline_fold_metrics(
        directory / "fold_metrics.csv", protocol
    )
    validate_baseline_aggregates(
        directory / "aggregate_metrics.json", fold_metrics
    )
    return BaselineRun(
        model_id=model_id,
        classifier_id=classifier_id,
        directory=directory,
        embedding_hashes=hashes,
        manifest_sha256=manifest_sha,
        fold_metrics=fold_metrics,
    )


def validate_alignment(
    tuned_runs: list[TunedRun],
    baseline_runs: dict[tuple[str, str], BaselineRun],
    model_ids: list[str],
    protocol: ProtocolContext,
) -> None:
    if [run.model_id for run in tuned_runs] != model_ids:
        raise ValueError("Nested runs do not match the configured model order.")
    if len(tuned_runs) != 4 or len(model_ids) != 4:
        raise ValueError(
            "This publication summary requires exactly four aligned models."
        )

    shared_identity_keys = ("clip_ids.npy", "y.npy")
    for key in shared_identity_keys:
        values = {run.embedding_hashes[key] for run in tuned_runs}
        if len(values) != 1:
            raise ValueError(
                f"Four-model embedding alignment failed: {key} hashes differ."
            )
    manifest_hashes = {run.manifest_sha256 for run in tuned_runs}
    if len(manifest_hashes) != 1:
        raise ValueError(
            "Four-model embedding alignment failed: manifest hashes differ."
        )

    expected_folds = set(protocol.fold_values)
    for tuned in tuned_runs:
        if {row.outer_fold for row in tuned.fold_results} != expected_folds:
            raise ValueError(f"Incomplete tuned folds for {tuned.model_id}.")
        for classifier_id in DEFAULT_CLASSIFIERS:
            key = (tuned.model_id, classifier_id)
            if key not in baseline_runs:
                raise ValueError(f"Missing strict baseline: {key}")
            baseline = baseline_runs[key]
            if baseline.embedding_hashes != tuned.embedding_hashes:
                raise ValueError(
                    f"Embedding hashes differ between tuned and "
                    f"{classifier_id} runs for {tuned.model_id}."
                )
            if baseline.manifest_sha256 != tuned.manifest_sha256:
                raise ValueError(
                    f"Manifest hash differs between tuned and "
                    f"{classifier_id} runs for {tuned.model_id}."
                )
            if set(baseline.fold_metrics) != expected_folds:
                raise ValueError(
                    f"Baseline folds are not aligned for {tuned.model_id}/"
                    f"{classifier_id}."
                )
            for fold in tuned.fold_results:
                baseline_row = baseline.fold_metrics[fold.outer_fold]
                if (
                    baseline_row["train_samples"] != fold.train_samples
                    or baseline_row["test_samples"] != fold.test_samples
                ):
                    raise ValueError(
                        f"Fold counts differ for {tuned.model_id}, outer fold "
                        f"{fold.outer_fold}, {classifier_id}."
                    )


def mean_and_sd(values: Sequence[float]) -> tuple[float, float]:
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def build_rows(
    tuned_runs: list[TunedRun],
    baseline_runs: dict[tuple[str, str], BaselineRun],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    model_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    overall_parameters: Counter[tuple[str, float, str]] = Counter()
    total_outer_folds = 0

    for run in tuned_runs:
        linear = baseline_runs[(run.model_id, "linear_svm")]
        rbf = baseline_runs[(run.model_id, "rbf_svm")]
        tuned_accuracy = [row.metrics["accuracy"] for row in run.fold_results]
        tuned_f1 = [row.metrics["f1_macro"] for row in run.fold_results]
        linear_accuracy = [
            float(linear.fold_metrics[row.outer_fold]["accuracy"])
            for row in run.fold_results
        ]
        rbf_accuracy = [
            float(rbf.fold_metrics[row.outer_fold]["accuracy"])
            for row in run.fold_results
        ]
        linear_f1 = [
            float(linear.fold_metrics[row.outer_fold]["f1_macro"])
            for row in run.fold_results
        ]
        rbf_f1 = [
            float(rbf.fold_metrics[row.outer_fold]["f1_macro"])
            for row in run.fold_results
        ]
        tuned_accuracy_mean, tuned_accuracy_sd = mean_and_sd(tuned_accuracy)
        tuned_f1_mean, tuned_f1_sd = mean_and_sd(tuned_f1)
        linear_accuracy_mean = statistics.fmean(linear_accuracy)
        rbf_accuracy_mean = statistics.fmean(rbf_accuracy)
        linear_f1_mean = statistics.fmean(linear_f1)
        rbf_f1_mean = statistics.fmean(rbf_f1)

        timing_values = {
            metric: [row.timings[metric] for row in run.fold_results]
            for metric in TUNING_TIMING_METRICS
        }
        parameter_counter = Counter(
            (row.best_kernel, row.best_c, row.best_gamma)
            for row in run.fold_results
        )
        overall_parameters.update(parameter_counter)
        total_outer_folds += len(run.fold_results)

        model_rows.append(
            {
                "model_order": run.model_order,
                "model_id": run.model_id,
                "model_display_name": run.model_display_name,
                "embedding_dimension": run.embedding_dimension,
                "outer_folds": len(run.fold_results),
                "outer_test_samples_total": sum(
                    row.test_samples for row in run.fold_results
                ),
                "tuned_accuracy_mean": tuned_accuracy_mean,
                "tuned_accuracy_fold_sd": tuned_accuracy_sd,
                "linear_svm_accuracy_mean": linear_accuracy_mean,
                "rbf_svm_accuracy_mean": rbf_accuracy_mean,
                "tuned_minus_linear_accuracy": (
                    tuned_accuracy_mean - linear_accuracy_mean
                ),
                "tuned_minus_rbf_accuracy": (
                    tuned_accuracy_mean - rbf_accuracy_mean
                ),
                "tuned_f1_macro_mean": tuned_f1_mean,
                "tuned_f1_macro_fold_sd": tuned_f1_sd,
                "linear_svm_f1_macro_mean": linear_f1_mean,
                "rbf_svm_f1_macro_mean": rbf_f1_mean,
                "tuned_minus_linear_f1_macro": tuned_f1_mean - linear_f1_mean,
                "tuned_minus_rbf_f1_macro": tuned_f1_mean - rbf_f1_mean,
                "grid_search_seconds_mean": statistics.fmean(
                    timing_values["grid_search_seconds"]
                ),
                "grid_search_seconds_total": sum(
                    timing_values["grid_search_seconds"]
                ),
                "classifier_fit_seconds_mean": statistics.fmean(
                    timing_values["classifier_fit_seconds"]
                ),
                "classifier_fit_seconds_total": sum(
                    timing_values["classifier_fit_seconds"]
                ),
                "prediction_ms_per_sample_mean": statistics.fmean(
                    timing_values["prediction_ms_per_sample"]
                ),
                "fold_total_seconds_mean": statistics.fmean(
                    timing_values["fold_total_seconds"]
                ),
                "unique_selected_parameter_sets": len(parameter_counter),
            }
        )

        timing_rows.append(
            {
                "model_order": run.model_order,
                "model_id": run.model_id,
                "model_display_name": run.model_display_name,
                "outer_folds": len(run.fold_results),
                "grid_search_seconds_mean": statistics.fmean(
                    timing_values["grid_search_seconds"]
                ),
                "grid_search_seconds_total": sum(
                    timing_values["grid_search_seconds"]
                ),
                "classifier_fit_seconds_mean": statistics.fmean(
                    timing_values["classifier_fit_seconds"]
                ),
                "classifier_fit_seconds_total": sum(
                    timing_values["classifier_fit_seconds"]
                ),
                "prediction_seconds_mean": statistics.fmean(
                    timing_values["prediction_seconds"]
                ),
                "prediction_seconds_total": sum(
                    timing_values["prediction_seconds"]
                ),
                "prediction_ms_per_sample_mean": statistics.fmean(
                    timing_values["prediction_ms_per_sample"]
                ),
                "fold_total_seconds_mean": statistics.fmean(
                    timing_values["fold_total_seconds"]
                ),
                "fold_total_seconds_total": sum(
                    timing_values["fold_total_seconds"]
                ),
            }
        )

        for fold in run.fold_results:
            linear_row = linear.fold_metrics[fold.outer_fold]
            rbf_row = rbf.fold_metrics[fold.outer_fold]
            fold_rows.append(
                {
                    "model_order": run.model_order,
                    "model_id": run.model_id,
                    "model_display_name": run.model_display_name,
                    "outer_fold": fold.outer_fold,
                    "train_samples": fold.train_samples,
                    "test_samples": fold.test_samples,
                    "inner_folds": fold.inner_folds,
                    "best_kernel": fold.best_kernel,
                    "best_C": fold.best_c,
                    "best_gamma": fold.best_gamma,
                    "best_inner_macro_f1": fold.best_inner_macro_f1,
                    "tuned_accuracy": fold.metrics["accuracy"],
                    "linear_svm_accuracy": linear_row["accuracy"],
                    "rbf_svm_accuracy": rbf_row["accuracy"],
                    "tuned_minus_linear_accuracy": (
                        fold.metrics["accuracy"] - float(linear_row["accuracy"])
                    ),
                    "tuned_minus_rbf_accuracy": (
                        fold.metrics["accuracy"] - float(rbf_row["accuracy"])
                    ),
                    "tuned_f1_macro": fold.metrics["f1_macro"],
                    "linear_svm_f1_macro": linear_row["f1_macro"],
                    "rbf_svm_f1_macro": rbf_row["f1_macro"],
                    "tuned_minus_linear_f1_macro": (
                        fold.metrics["f1_macro"]
                        - float(linear_row["f1_macro"])
                    ),
                    "tuned_minus_rbf_f1_macro": (
                        fold.metrics["f1_macro"] - float(rbf_row["f1_macro"])
                    ),
                    **fold.timings,
                }
            )

        for (kernel, c_value, gamma), count in sorted(
            parameter_counter.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            parameter_rows.append(
                {
                    "scope": "model",
                    "model_order": run.model_order,
                    "model_id": run.model_id,
                    "model_display_name": run.model_display_name,
                    "kernel": kernel,
                    "C": c_value,
                    "gamma": gamma,
                    "selection_count": count,
                    "available_outer_folds": len(run.fold_results),
                    "selection_percent": 100.0 * count / len(run.fold_results),
                }
            )

    for (kernel, c_value, gamma), count in sorted(
        overall_parameters.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        parameter_rows.append(
            {
                "scope": "all_models",
                "model_order": "",
                "model_id": "ALL",
                "model_display_name": "All four models",
                "kernel": kernel,
                "C": c_value,
                "gamma": gamma,
                "selection_count": count,
                "available_outer_folds": total_outer_folds,
                "selection_percent": 100.0 * count / total_outer_folds,
            }
        )

    return model_rows, fold_rows, parameter_rows, timing_rows


def save_dual_figure(figure: Any, output_stem: Path) -> None:
    for extension in ("png", "pdf"):
        figure.savefig(
            output_stem.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )


def render_metric_grouped_bar(
    model_rows: list[dict[str, object]],
    metric: str,
    output_stem: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if metric == "accuracy":
        keys = (
            "linear_svm_accuracy_mean",
            "rbf_svm_accuracy_mean",
            "tuned_accuracy_mean",
        )
        title = "Nested SVM tuning: outer-fold recognition accuracy"
        y_label = "Mean accuracy (%)"
    else:
        keys = (
            "linear_svm_f1_macro_mean",
            "rbf_svm_f1_macro_mean",
            "tuned_f1_macro_mean",
        )
        title = "Nested SVM tuning: outer-fold macro-F1"
        y_label = "Mean macro-F1 (%)"

    labels = [str(row["model_display_name"]) for row in model_rows]
    x_values = np.arange(len(labels), dtype=float)
    width = 0.24
    colors = ("#4C78A8", "#F58518", "#54A24B")
    figure, axis = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)

    for series_index, (key, series_label, color) in enumerate(
        zip(
            keys,
            (
                DEFAULT_CLASSIFIER_LABELS["linear_svm"],
                DEFAULT_CLASSIFIER_LABELS["rbf_svm"],
                DEFAULT_CLASSIFIER_LABELS["tuned_nested_svm"],
            ),
            colors,
        )
    ):
        values = [100.0 * float(row[key]) for row in model_rows]
        positions = x_values + (series_index - 1) * width
        bars = axis.bar(
            positions,
            values,
            width=width,
            label=series_label,
            color=color,
            edgecolor="white",
        )
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    # Bar lengths encode magnitude, so a zero baseline is retained to avoid
    # visually exaggerating the relatively small differences among systems.
    axis.set_ylim(0.0, 103.0)
    axis.set_xticks(x_values, labels)
    axis.set_ylabel(y_label)
    axis.set_xlabel("Pre-trained speaker embedding model")
    axis.set_title(
        title + "\nStrict chapter-held-out outer protocol",
        fontweight="bold",
        pad=12,
    )
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend(frameon=False, loc="lower right")
    axis.text(
        0.0,
        -0.16,
        (
            "Descriptive outer-fold means only (2 correlated folds); "
            "no statistical-significance claim."
        ),
        transform=axis.transAxes,
        fontsize=9,
        color="#444444",
    )
    save_dual_figure(figure, output_stem)
    plt.close(figure)


def render_delta_plot(
    model_rows: list[dict[str, object]],
    output_stem: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [str(row["model_display_name"]) for row in model_rows]
    x_values = np.arange(len(labels), dtype=float)
    width = 0.34
    figure, axes = plt.subplots(
        1, 2, figsize=(13.0, 5.8), constrained_layout=True, sharex=True
    )
    specifications = (
        (
            "Accuracy delta",
            "tuned_minus_linear_accuracy",
            "tuned_minus_rbf_accuracy",
        ),
        (
            "Macro-F1 delta",
            "tuned_minus_linear_f1_macro",
            "tuned_minus_rbf_f1_macro",
        ),
    )

    for axis, (title, linear_key, rbf_key) in zip(axes, specifications):
        linear_values = [100.0 * float(row[linear_key]) for row in model_rows]
        rbf_values = [100.0 * float(row[rbf_key]) for row in model_rows]
        axis.bar(
            x_values - width / 2,
            linear_values,
            width=width,
            label="Tuned − fixed linear",
            color="#4C78A8",
        )
        axis.bar(
            x_values + width / 2,
            rbf_values,
            width=width,
            label="Tuned − fixed RBF",
            color="#F58518",
        )
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.set_xticks(x_values, labels, rotation=18, ha="right")
        axis.set_title(title, fontweight="bold")
        axis.set_ylabel("Percentage-point difference")
        axis.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0].legend(frameon=False, loc="best")
    figure.suptitle(
        "Nested tuned SVM minus fixed-default baselines\n"
        "Strict chapter-held-out outer protocol",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.01,
        (
            "Descriptive paired fold deltas only (2 correlated folds); "
            "no statistical-significance claim."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    save_dual_figure(figure, output_stem)
    plt.close(figure)


def serialize_runs(
    tuned_runs: list[TunedRun],
    baseline_runs: dict[tuple[str, str], BaselineRun],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for run in tuned_runs:
        output.append(
            {
                "model_order": run.model_order,
                "model_id": run.model_id,
                "model_display_name": run.model_display_name,
                "provider": run.provider,
                "architecture": run.architecture,
                "embedding_dimension": run.embedding_dimension,
                "embedding_tag": run.tag,
                "tuned_run_directory": str(run.directory),
                "linear_svm_run_directory": str(
                    baseline_runs[(run.model_id, "linear_svm")].directory
                ),
                "rbf_svm_run_directory": str(
                    baseline_runs[(run.model_id, "rbf_svm")].directory
                ),
                "embedding_artifact_sha256": run.embedding_hashes,
                "manifest_sha256": run.manifest_sha256,
                "selected_parameters": [
                    {
                        "outer_fold": fold.outer_fold,
                        "kernel": fold.best_kernel,
                        "C": fold.best_c,
                        "gamma": fold.best_gamma,
                        "best_inner_macro_f1": fold.best_inner_macro_f1,
                    }
                    for fold in run.fold_results
                ],
            }
        )
    return output


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    tag = str(options.tag).strip()
    if not tag:
        raise ValueError("--tag cannot be empty.")

    protocol = load_protocol_context(options.protocol_file)
    model_ids = load_model_ids(options.config.resolve())
    if len(model_ids) != 4:
        raise ValueError(
            "The research config must declare exactly four embedding models."
        )

    tuning_root = options.tuning_root.resolve()
    baseline_root = options.baseline_root.resolve()
    tuned_protocol_root = tuning_root / "protocols" / protocol.protocol_id / "svm"
    baseline_protocol_root = baseline_root / "protocols" / protocol.protocol_id
    if options.output_root is None:
        output_directory = (
            tuning_root
            / "protocols"
            / protocol.protocol_id
            / "summary"
            / "svm"
            / tag
        )
    else:
        output_directory = options.output_root.resolve()

    tuned_runs: list[TunedRun] = []
    baseline_runs: dict[tuple[str, str], BaselineRun] = {}
    for model_order, model_id in enumerate(model_ids, start=1):
        tuned_runs.append(
            load_tuned_run(
                tuned_protocol_root / model_id / tag,
                model_id,
                model_order,
                tag,
                protocol,
            )
        )
        for classifier_id in DEFAULT_CLASSIFIERS:
            baseline_runs[(model_id, classifier_id)] = load_baseline_run(
                baseline_protocol_root
                / model_id
                / classifier_id
                / tag,
                model_id,
                classifier_id,
                tag,
                protocol,
            )

    validate_alignment(tuned_runs, baseline_runs, model_ids, protocol)
    model_rows, fold_rows, parameter_rows, timing_rows = build_rows(
        tuned_runs, baseline_runs
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    atomic_csv(
        output_directory / "model_level_comparison.csv",
        MODEL_LEVEL_FIELDS,
        model_rows,
    )
    atomic_json(
        output_directory / "model_level_comparison.json",
        {
            "schema_version": 1,
            "protocol_id": protocol.protocol_id,
            "rows": model_rows,
            "interpretation": (
                "Descriptive outer-fold comparison only. Two correlated "
                "outer folds are insufficient for a significance claim."
            ),
        },
    )
    atomic_csv(
        output_directory / "fold_level_comparison.csv",
        FOLD_LEVEL_FIELDS,
        fold_rows,
    )
    atomic_json(
        output_directory / "fold_level_comparison.json",
        {
            "schema_version": 1,
            "protocol_id": protocol.protocol_id,
            "rows": fold_rows,
        },
    )
    atomic_csv(
        output_directory / "selected_parameter_frequency.csv",
        PARAMETER_FREQUENCY_FIELDS,
        parameter_rows,
    )
    atomic_json(
        output_directory / "selected_parameter_frequency.json",
        {
            "schema_version": 1,
            "rows": parameter_rows,
            "unit": "outer-fold selections",
        },
    )
    atomic_csv(
        output_directory / "search_training_time.csv",
        TIMING_FIELDS,
        timing_rows,
    )
    atomic_json(
        output_directory / "search_training_time.json",
        {
            "schema_version": 1,
            "rows": timing_rows,
            "timing_scope": (
                "Nested inner search, final selected-classifier fit, "
                "outer-test prediction, and total outer-fold wall time."
            ),
        },
    )

    render_metric_grouped_bar(
        model_rows,
        "accuracy",
        output_directory / "accuracy_comparison",
    )
    render_metric_grouped_bar(
        model_rows,
        "f1_macro",
        output_directory / "macro_f1_comparison",
    )
    render_delta_plot(model_rows, output_directory / "delta_vs_default")

    caution = (
        "The strict protocol contains two correlated outer folds. Fold means, "
        "fold standard deviations, and paired deltas are descriptive only. "
        "No inferential statistical-significance claim is made."
    )
    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "experiment": "nested_svm_tuning_summary",
        "status": "complete",
        "protocol": {
            "path": str(protocol.path),
            "sha256": protocol.sha256,
            "protocol_id": protocol.protocol_id,
            "ordered_assignment_sha256": protocol.ordered_assignment_sha256,
            "rows": protocol.row_count,
            "speakers": protocol.speakers,
            "source_groups": protocol.source_groups,
            "fold_values": list(protocol.fold_values),
            "fold_counts": protocol.fold_counts,
        },
        "embedding_tag": tag,
        "models": model_ids,
        "tuned_run_root": str(tuned_protocol_root),
        "baseline_run_root": str(baseline_protocol_root),
        "baseline_classifiers": list(DEFAULT_CLASSIFIERS),
        "selection_metric": "f1_macro",
        "outer_evaluation": (
            "Strict source-group-disjoint chapter-held-out protocol"
        ),
        "alignment_checks": {
            "all_four_models_complete": True,
            "protocol_hashes_identical": True,
            "protocol_assignments_identical": True,
            "clip_id_hashes_identical_across_models": True,
            "label_hashes_identical_across_models": True,
            "tuned_and_baseline_embedding_hashes_identical_per_model": True,
            "fold_train_test_counts_identical": True,
            "outer_test_used_in_tuning": False,
            "outer_clip_overlap_count": 0,
            "outer_source_group_overlap_count": 0,
        },
        "statistical_caution": caution,
        "significance_test_performed": False,
        "runs": serialize_runs(tuned_runs, baseline_runs),
        "artifacts": {
            "model_level_csv": "model_level_comparison.csv",
            "model_level_json": "model_level_comparison.json",
            "fold_level_csv": "fold_level_comparison.csv",
            "fold_level_json": "fold_level_comparison.json",
            "selected_parameter_frequency_csv": (
                "selected_parameter_frequency.csv"
            ),
            "selected_parameter_frequency_json": (
                "selected_parameter_frequency.json"
            ),
            "timing_csv": "search_training_time.csv",
            "timing_json": "search_training_time.json",
            "accuracy_figure": [
                "accuracy_comparison.png",
                "accuracy_comparison.pdf",
            ],
            "macro_f1_figure": [
                "macro_f1_comparison.png",
                "macro_f1_comparison.pdf",
            ],
            "delta_figure": [
                "delta_vs_default.png",
                "delta_vs_default.pdf",
            ],
        },
        "output_directory": str(output_directory),
    }
    atomic_json(output_directory / "summary_metadata.json", metadata)

    print("=" * 78)
    print("NESTED SVM TUNING SUMMARY")
    print("=" * 78)
    print(f"Protocol       : {protocol.protocol_id}")
    print(f"Models         : {len(tuned_runs)}")
    print(f"Outer folds    : {len(protocol.fold_values)}")
    print(f"Baseline pairs : {len(baseline_runs)}")
    print("Leakage checks : passed")
    print("Status         : COMPLETE")
    print(f"Artifacts      : {output_directory}")
    print("=" * 78)
    print("\nCaution: " + caution)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nSummary interrupted.")
        raise SystemExit(130)
    except Exception as error:
        print("\n" + "=" * 78)
        print("NESTED SVM TUNING SUMMARY FAILED")
        print("=" * 78)
        print(str(error))
        print("=" * 78)
        raise SystemExit(1)
