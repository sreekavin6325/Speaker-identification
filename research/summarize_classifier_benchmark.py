"""Summarize the strict multi-model, multi-classifier benchmark.

This module is intentionally separate from ``summarize_baseline_cv.py``.
The baseline summarizer compares embedding models for one classifier, whereas
this script builds the model-by-classifier tables needed for the classifier
ablation in the research paper.

Only results produced with an explicit, hashed protocol file are accepted.
The script verifies the protocol identity, ordered assignments, fold counts,
sample counts, speaker counts, model identity, and embedding-artifact hashes
before combining any measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_ROOT = PROJECT_ROOT / "research_results" / "baseline_cv"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "research_baseline.json"

STRICT_PROTOCOL_LABEL = (
    "Strict chapter-held-out closed-set speaker-identification protocol"
)

REQUIRED_CLASSIFIERS = (
    "linear_svm",
    "rbf_svm",
    "logistic_regression",
    "knn",
    "random_forest",
    "decision_tree",
    "xgboost",
)
OPTIONAL_CLASSIFIERS = ("cosine_centroid",)

CLASSIFIER_LABELS = {
    "linear_svm": "Linear SVM",
    "rbf_svm": "RBF SVM",
    "logistic_regression": "Logistic Regression",
    "knn": "KNN",
    "random_forest": "Random Forest",
    "decision_tree": "Decision Tree",
    "xgboost": "XGBoost",
    "cosine_centroid": "Cosine Centroid",
}

MODEL_LABELS = {
    "speechbrain_ecapa": "ECAPA-TDNN",
    "speechbrain_xvector": "X-Vector",
    "wavlm_base_plus_sv": "WavLM",
    "unispeech_sat_base_plus_sv": "UniSpeech-SAT",
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
TIMING_METRICS = (
    "scaler_fit_seconds",
    "scaler_transform_seconds",
    "classifier_fit_seconds",
    "prediction_seconds",
    "prediction_ms_per_sample",
    "fold_total_seconds",
)
ALL_FOLD_METRICS = PERFORMANCE_METRICS + TIMING_METRICS

REQUIRED_RUN_FILES = (
    "run_metadata.json",
    "aggregate_metrics.json",
    "fold_metrics.csv",
)

LONG_CSV_FIELDS = (
    "protocol_label",
    "protocol_id",
    "protocol_sha256",
    "ordered_assignment_sha256",
    "model_order",
    "model_id",
    "model_display_name",
    "provider",
    "architecture",
    "embedding_dimension",
    "classifier_order",
    "classifier_id",
    "classifier_display_name",
    "embedding_tag",
    "fold",
    "train_samples",
    "test_samples",
    *PERFORMANCE_METRICS,
    *TIMING_METRICS,
    "aggregate_accuracy_mean",
    "aggregate_accuracy_fold_sd",
    "aggregate_f1_macro_mean",
    "aggregate_f1_macro_fold_sd",
    "run_directory",
)


@dataclass(frozen=True)
class ProtocolContext:
    """Canonical identity of the supplied strict evaluation protocol."""

    path: Path
    sha256: str
    protocol_id: str
    ordered_assignment_sha256: str
    row_count: int
    fold_values: tuple[int, ...]
    fold_counts: dict[str, int]
    speakers: int
    source_groups: int


@dataclass
class ValidatedRun:
    """One fully validated model/classifier result."""

    model_order: int
    model_id: str
    model_display_name: str
    provider: str
    architecture: str
    embedding_dimension: int
    classifier_order: int
    classifier_id: str
    classifier_display_name: str
    tag: str
    run_directory: Path
    fold_rows: list[dict[str, int | float]]
    aggregates: dict[str, dict[str, float | int]]
    embedding_hashes: dict[str, str]
    manifest_sha256: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-ready model-by-classifier tables and figures "
            "for the strict chapter-held-out benchmark."
        )
    )
    parser.add_argument(
        "--protocol-file",
        type=Path,
        required=True,
        help=(
            "Required strict protocol CSV used for every benchmark run. "
            "The file is re-hashed and its exact assignments are validated."
        ),
    )
    parser.add_argument(
        "--tag",
        default="full",
        help="Embedding/result tag to summarize (default: full).",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_BASELINE_ROOT,
        help="Root containing protocol-specific baseline results.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Research config declaring the deterministic embedding-model order.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Exact output directory. By default, artifacts are written below "
            "<baseline-root>/protocols/<protocol-id>/summary/"
            "classifier_benchmark/<tag>."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Generate a clearly marked partial summary when classifier runs "
            "are missing or incomplete. Protocol/alignment errors still fail."
        ),
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


def safe_protocol_name(path: Path, protocol_hash: str) -> str:
    """Mirror ``run_baseline_cv.safe_protocol_name`` exactly."""

    stem = path.stem.lower()

    if stem.endswith("_protocol"):
        stem = stem[: -len("_protocol")]

    stem = re.sub(r"[^a-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")

    if not stem:
        stem = "external_protocol"

    return f"{stem}__{protocol_hash[:12]}"


def load_protocol_context(path: Path) -> ProtocolContext:
    protocol_path = path.resolve()

    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"Strict protocol file was not found:\n{protocol_path}"
        )

    protocol_hash = file_sha256(protocol_path)
    rows: list[dict[str, str | int]] = []
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
            raise ValueError("Strict protocol CSV has no header.")

        missing_columns = required - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                "Strict protocol CSV is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, source_row in enumerate(reader, start=2):
            clip_id = source_row["clip_id"].strip()
            speaker_label = source_row["speaker_label"].strip()
            source_group = source_row["source_group"].strip()
            fold_text = source_row["fold"].strip()

            if not clip_id or not speaker_label or not source_group:
                raise ValueError(
                    f"Protocol row {row_number} contains an empty identity field."
                )

            if clip_id in seen_clip_ids:
                raise ValueError(f"Duplicate protocol clip_id: {clip_id}")

            try:
                fold = int(fold_text)
            except ValueError as error:
                raise ValueError(
                    f"Protocol row {row_number} has invalid fold {fold_text!r}."
                ) from error

            if fold < 1 or str(fold) != fold_text:
                raise ValueError(
                    f"Protocol row {row_number} fold must be a canonical "
                    f"positive integer, got {fold_text!r}."
                )

            previous_fold = group_to_fold.get(source_group)
            if previous_fold is not None and previous_fold != fold:
                raise ValueError(
                    "Strict protocol leakage: source_group occurs in more "
                    f"than one fold: {source_group!r}."
                )

            seen_clip_ids.add(clip_id)
            group_to_fold[source_group] = fold
            rows.append(
                {
                    "clip_id": clip_id,
                    "speaker_label": speaker_label,
                    "source_group": source_group,
                    "fold": fold,
                }
            )

    if not rows:
        raise ValueError("Strict protocol CSV contains no data rows.")

    fold_values = tuple(sorted({int(row["fold"]) for row in rows}))
    if len(fold_values) < 2:
        raise ValueError("Strict protocol must contain at least two folds.")

    assignment_digest = hashlib.sha256()
    for row in rows:
        canonical = (
            f"{row['clip_id']}\t{row['speaker_label']}\t"
            f"{row['source_group']}\t{row['fold']}\n"
        )
        assignment_digest.update(canonical.encode("utf-8"))

    return ProtocolContext(
        path=protocol_path,
        sha256=protocol_hash,
        protocol_id=safe_protocol_name(protocol_path, protocol_hash),
        ordered_assignment_sha256=assignment_digest.hexdigest(),
        row_count=len(rows),
        fold_values=fold_values,
        fold_counts={
            str(fold): sum(int(row["fold"]) == fold for row in rows)
            for fold in fold_values
        },
        speakers=len({str(row["speaker_label"]) for row in rows}),
        source_groups=len(group_to_fold),
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file was not found:\n{path}")

    with path.open("r", encoding="utf-8") as input_file:
        value = json.load(input_file)

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in:\n{path}")

    return value


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

    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite '{name}' in:\n{source}")

    return numeric


def load_model_ids(config_path: Path) -> list[str]:
    config = read_json(config_path.resolve())
    value = config.get("embedding_models")

    if not isinstance(value, list) or not value:
        raise ValueError(
            "Research config must contain a non-empty embedding_models list:\n"
            f"{config_path.resolve()}"
        )

    model_ids: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Invalid model ID in:\n{config_path.resolve()}")

        model_id = item.strip()
        if model_id in model_ids:
            raise ValueError(
                f"Duplicate model ID {model_id!r} in:\n{config_path.resolve()}"
            )
        model_ids.append(model_id)

    return model_ids


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
        output_file.write("\n")

    os.replace(temporary, path)


def atomic_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, object]],
) -> None:
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


def display_model_name(model_id: str, metadata_name: object) -> str:
    preferred = MODEL_LABELS.get(model_id)
    if preferred:
        return preferred

    value = str(metadata_name or model_id)
    return value.replace("â€”", "—")


def load_fold_rows(
    path: Path,
    protocol: ProtocolContext,
) -> list[dict[str, int | float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Fold metrics were not found:\n{path}")

    rows: list[dict[str, int | float]] = []
    seen_folds: set[int] = set()

    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"fold", "train_samples", "test_samples"} | set(
            ALL_FOLD_METRICS
        )

        if reader.fieldnames is None:
            raise ValueError(f"Fold metrics CSV has no header:\n{path}")

        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Fold metrics CSV is missing columns in:\n{path}\n"
                + ", ".join(sorted(missing))
            )

        for row_number, source_row in enumerate(reader, start=2):
            try:
                fold = int(source_row["fold"])
                train_samples = int(source_row["train_samples"])
                test_samples = int(source_row["test_samples"])
            except ValueError as error:
                raise ValueError(
                    f"Invalid integer in {path}, row {row_number}."
                ) from error

            if fold in seen_folds:
                raise ValueError(f"Duplicate fold {fold} in:\n{path}")

            if fold not in protocol.fold_values:
                raise ValueError(
                    f"Unexpected fold {fold} in {path}; expected "
                    f"{list(protocol.fold_values)}."
                )

            expected_test = protocol.fold_counts[str(fold)]
            expected_train = protocol.row_count - expected_test
            if test_samples != expected_test or train_samples != expected_train:
                raise ValueError(
                    f"Fold {fold} sample counts do not match the strict "
                    f"protocol in:\n{path}\nExpected train/test "
                    f"{expected_train}/{expected_test}, observed "
                    f"{train_samples}/{test_samples}."
                )

            row: dict[str, int | float] = {
                "fold": fold,
                "train_samples": train_samples,
                "test_samples": test_samples,
            }

            for metric in ALL_FOLD_METRICS:
                try:
                    numeric = float(source_row[metric])
                except ValueError as error:
                    raise ValueError(
                        f"Invalid {metric} in {path}, row {row_number}."
                    ) from error

                if not math.isfinite(numeric):
                    raise ValueError(
                        f"Non-finite {metric} in {path}, row {row_number}."
                    )
                if metric in PERFORMANCE_METRICS and not 0.0 <= numeric <= 1.0:
                    raise ValueError(
                        f"{metric} is outside [0, 1] in {path}, "
                        f"row {row_number}."
                    )
                if metric in TIMING_METRICS and numeric < 0.0:
                    raise ValueError(
                        f"Negative timing {metric} in {path}, row {row_number}."
                    )

                row[metric] = numeric

            rows.append(row)
            seen_folds.add(fold)

    rows.sort(key=lambda row: int(row["fold"]))
    observed = tuple(int(row["fold"]) for row in rows)
    if observed != protocol.fold_values:
        raise ValueError(
            f"Fold metrics are incomplete in:\n{path}\n"
            f"Expected {list(protocol.fold_values)}, observed {list(observed)}."
        )

    return rows


def validate_aggregate(
    metric: str,
    aggregate: dict[str, Any],
    fold_rows: list[dict[str, int | float]],
    source: Path,
) -> dict[str, float | int]:
    values = [float(row[metric]) for row in fold_rows]
    observed_n = int(finite_number(aggregate.get("n"), f"{metric}.n", source))
    observed_mean = finite_number(
        aggregate.get("mean"),
        f"{metric}.mean",
        source,
    )
    observed_std = finite_number(
        aggregate.get("std"),
        f"{metric}.std",
        source,
    )

    expected_mean = statistics.fmean(values)
    expected_std = statistics.stdev(values) if len(values) > 1 else 0.0

    if observed_n != len(values):
        raise ValueError(
            f"Aggregate {metric}.n does not match fold metrics in:\n{source}"
        )

    if not math.isclose(observed_mean, expected_mean, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(
            f"Aggregate {metric}.mean does not match fold metrics in:\n{source}"
        )

    if not math.isclose(observed_std, expected_std, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(
            f"Aggregate {metric}.std does not match fold metrics in:\n{source}"
        )

    validated: dict[str, float | int] = {
        "n": observed_n,
        "mean": observed_mean,
        "std": observed_std,
    }

    for key in (
        "ci95_half_width",
        "ci95_lower",
        "ci95_upper",
        "minimum",
        "maximum",
    ):
        if key in aggregate:
            validated[key] = finite_number(
                aggregate[key],
                f"{metric}.{key}",
                source,
            )

    return validated


def validate_protocol_metadata(
    protocol_metadata: dict[str, Any],
    protocol: ProtocolContext,
    source: Path,
) -> None:
    expected = {
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

    for key, expected_value in expected.items():
        if protocol_metadata.get(key) != expected_value:
            raise ValueError(
                f"Strict protocol metadata mismatch for '{key}' in:\n"
                f"{source}\nExpected: {expected_value!r}\n"
                f"Observed: {protocol_metadata.get(key)!r}"
            )


def validate_run(
    run_directory: Path,
    model_id: str,
    model_order: int,
    classifier_id: str,
    classifier_order: int,
    tag: str,
    protocol: ProtocolContext,
) -> ValidatedRun:
    run_metadata_path = run_directory / "run_metadata.json"
    aggregate_path = run_directory / "aggregate_metrics.json"
    fold_path = run_directory / "fold_metrics.csv"

    run_metadata = read_json(run_metadata_path)
    aggregate_metadata = read_json(aggregate_path)

    if run_metadata.get("model_id") != model_id:
        raise ValueError(
            f"Model ID mismatch in:\n{run_metadata_path}\n"
            f"Expected {model_id!r}, observed {run_metadata.get('model_id')!r}."
        )

    classifier_metadata = require_mapping(
        run_metadata,
        "classifier",
        run_metadata_path,
    )
    if classifier_metadata.get("name") != classifier_id:
        raise ValueError(
            f"Classifier ID mismatch in:\n{run_metadata_path}\n"
            f"Expected {classifier_id!r}, observed "
            f"{classifier_metadata.get('name')!r}."
        )

    protocol_metadata = require_mapping(
        run_metadata,
        "protocol",
        run_metadata_path,
    )
    validate_protocol_metadata(protocol_metadata, protocol, run_metadata_path)

    manifest = require_mapping(run_metadata, "manifest", run_metadata_path)
    manifest_sha = str(manifest.get("sha256", "")).lower().strip()
    if not manifest_sha:
        raise ValueError(f"Missing manifest SHA-256 in:\n{run_metadata_path}")

    model_metadata = require_mapping(run_metadata, "model", run_metadata_path)
    embedding_artifacts = require_mapping(
        run_metadata,
        "embedding_artifacts",
        run_metadata_path,
    )
    raw_hashes = require_mapping(
        embedding_artifacts,
        "sha256",
        run_metadata_path,
    )
    embedding_hashes = {
        str(key): str(value).lower()
        for key, value in raw_hashes.items()
    }
    if not embedding_hashes:
        raise ValueError(
            f"Embedding artifact hashes are empty in:\n{run_metadata_path}"
        )

    source_metadata = require_mapping(
        embedding_artifacts,
        "source_metadata",
        run_metadata_path,
    )
    if source_metadata.get("model_id") != model_id:
        raise ValueError(
            f"Embedding source model mismatch in:\n{run_metadata_path}"
        )
    if source_metadata.get("tag") != tag:
        raise ValueError(
            f"Embedding tag mismatch in:\n{run_metadata_path}\n"
            f"Expected {tag!r}, observed {source_metadata.get('tag')!r}."
        )
    if str(source_metadata.get("manifest_sha256", "")).lower() != manifest_sha:
        raise ValueError(
            f"Embedding/baseline manifest hash mismatch in:\n{run_metadata_path}"
        )

    fold_rows = load_fold_rows(fold_path, protocol)
    aggregate_values = require_mapping(
        aggregate_metadata,
        "values",
        aggregate_path,
    )
    aggregates: dict[str, dict[str, float | int]] = {}

    for metric in ALL_FOLD_METRICS:
        metric_aggregate = require_mapping(
            aggregate_values,
            metric,
            aggregate_path,
        )
        aggregates[metric] = validate_aggregate(
            metric,
            metric_aggregate,
            fold_rows,
            aggregate_path,
        )

    return ValidatedRun(
        model_order=model_order,
        model_id=model_id,
        model_display_name=display_model_name(
            model_id,
            model_metadata.get("display_name"),
        ),
        provider=str(model_metadata.get("provider") or ""),
        architecture=str(model_metadata.get("architecture") or ""),
        embedding_dimension=int(
            finite_number(
                protocol_metadata.get("embedding_dimension"),
                "protocol.embedding_dimension",
                run_metadata_path,
            )
        ),
        classifier_order=classifier_order,
        classifier_id=classifier_id,
        classifier_display_name=CLASSIFIER_LABELS[classifier_id],
        tag=tag,
        run_directory=run_directory.resolve(),
        fold_rows=fold_rows,
        aggregates=aggregates,
        embedding_hashes=embedding_hashes,
        manifest_sha256=manifest_sha,
    )


def discover_and_validate_runs(
    strict_root: Path,
    model_ids: list[str],
    tag: str,
    protocol: ProtocolContext,
    allow_missing: bool,
) -> tuple[list[ValidatedRun], list[dict[str, str]], list[str]]:
    optional_present = any(
        (strict_root / model_id / classifier / tag).exists()
        for model_id in model_ids
        for classifier in OPTIONAL_CLASSIFIERS
    )
    classifier_ids = list(REQUIRED_CLASSIFIERS)
    if optional_present:
        classifier_ids.extend(OPTIONAL_CLASSIFIERS)

    runs: list[ValidatedRun] = []
    missing_runs: list[dict[str, str]] = []

    for model_order, model_id in enumerate(model_ids, start=1):
        for classifier_order, classifier_id in enumerate(
            classifier_ids,
            start=1,
        ):
            run_directory = strict_root / model_id / classifier_id / tag
            missing_files = [
                filename
                for filename in REQUIRED_RUN_FILES
                if not (run_directory / filename).is_file()
            ]

            if missing_files:
                missing_runs.append(
                    {
                        "model_id": model_id,
                        "classifier_id": classifier_id,
                        "run_directory": str(run_directory.resolve()),
                        "reason": (
                            "result directory missing"
                            if not run_directory.exists()
                            else "incomplete run; missing "
                            + ", ".join(missing_files)
                        ),
                    }
                )
                continue

            runs.append(
                validate_run(
                    run_directory=run_directory,
                    model_id=model_id,
                    model_order=model_order,
                    classifier_id=classifier_id,
                    classifier_order=classifier_order,
                    tag=tag,
                    protocol=protocol,
                )
            )

    if missing_runs and not allow_missing:
        details = "\n".join(
            f"- {item['model_id']} / {item['classifier_id']}: "
            f"{item['reason']}"
            for item in missing_runs
        )
        raise FileNotFoundError(
            "Classifier benchmark is incomplete. Every required classifier "
            "must be completed for every model before publication summary.\n"
            f"{details}\nUse --allow-missing only for a clearly marked "
            "partial development summary."
        )

    if not runs:
        raise FileNotFoundError(
            "No complete classifier benchmark runs were found under:\n"
            f"{strict_root}"
        )

    validate_cross_run_alignment(runs)
    return runs, missing_runs, classifier_ids


def validate_cross_run_alignment(runs: list[ValidatedRun]) -> None:
    """Require one immutable embedding artifact set per model."""

    by_model: dict[str, list[ValidatedRun]] = {}
    for run in runs:
        by_model.setdefault(run.model_id, []).append(run)

    manifest_hashes = {run.manifest_sha256 for run in runs}
    if len(manifest_hashes) != 1:
        raise ValueError(
            "Cross-run alignment failed: source manifest SHA-256 differs "
            "between benchmark runs."
        )

    for model_id, model_runs in by_model.items():
        reference = model_runs[0]
        for run in model_runs[1:]:
            differences: list[str] = []
            if run.embedding_hashes != reference.embedding_hashes:
                differences.append("embedding artifact SHA-256 map")
            if run.embedding_dimension != reference.embedding_dimension:
                differences.append("embedding dimension")
            if run.model_display_name != reference.model_display_name:
                differences.append("model display name")
            if run.provider != reference.provider:
                differences.append("provider")
            if run.architecture != reference.architecture:
                differences.append("architecture")

            if differences:
                raise ValueError(
                    f"Classifier runs for model {model_id!r} are not aligned. "
                    "They differ in: "
                    + ", ".join(differences)
                    + ". Re-run classifiers from the same immutable embedding "
                    "artifacts."
                )


def build_long_rows(
    runs: list[ValidatedRun],
    protocol: ProtocolContext,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for run in sorted(
        runs,
        key=lambda item: (item.model_order, item.classifier_order),
    ):
        for fold_row in run.fold_rows:
            output: dict[str, object] = {
                "protocol_label": STRICT_PROTOCOL_LABEL,
                "protocol_id": protocol.protocol_id,
                "protocol_sha256": protocol.sha256,
                "ordered_assignment_sha256": (
                    protocol.ordered_assignment_sha256
                ),
                "model_order": run.model_order,
                "model_id": run.model_id,
                "model_display_name": run.model_display_name,
                "provider": run.provider,
                "architecture": run.architecture,
                "embedding_dimension": run.embedding_dimension,
                "classifier_order": run.classifier_order,
                "classifier_id": run.classifier_id,
                "classifier_display_name": run.classifier_display_name,
                "embedding_tag": run.tag,
                "fold": fold_row["fold"],
                "train_samples": fold_row["train_samples"],
                "test_samples": fold_row["test_samples"],
                "aggregate_accuracy_mean": run.aggregates["accuracy"]["mean"],
                "aggregate_accuracy_fold_sd": run.aggregates["accuracy"]["std"],
                "aggregate_f1_macro_mean": run.aggregates["f1_macro"]["mean"],
                "aggregate_f1_macro_fold_sd": run.aggregates["f1_macro"]["std"],
                "run_directory": str(run.run_directory),
            }
            for metric in ALL_FOLD_METRICS:
                output[metric] = fold_row[metric]

            rows.append(output)

    return rows


def matrix_values(
    runs: list[ValidatedRun],
    model_ids: list[str],
    classifier_ids: list[str],
    metric: str,
    multiplier: float = 1.0,
) -> list[list[float | None]]:
    lookup = {
        (run.model_id, run.classifier_id): (
            float(run.aggregates[metric]["mean"]) * multiplier
        )
        for run in runs
    }
    return [
        [lookup.get((model_id, classifier_id)) for classifier_id in classifier_ids]
        for model_id in model_ids
    ]


def matrix_rows(
    model_ids: list[str],
    classifier_ids: list[str],
    values: list[list[float | None]],
) -> tuple[list[str], list[dict[str, object]]]:
    fields = ["model_id", "model_display_name"] + list(classifier_ids)
    rows: list[dict[str, object]] = []

    for model_id, row_values in zip(model_ids, values):
        row: dict[str, object] = {
            "model_id": model_id,
            "model_display_name": MODEL_LABELS.get(model_id, model_id),
        }
        for classifier_id, value in zip(classifier_ids, row_values):
            row[classifier_id] = "" if value is None else value
        rows.append(row)

    return fields, rows


def render_heatmap(
    values: list[list[float | None]],
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    colorbar_label: str,
    output_stem: Path,
    annotation_format: str,
    fixed_range: tuple[float, float] | None = None,
    logarithmic: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    array = np.array(
        [
            [np.nan if value is None else float(value) for value in row]
            for row in values
        ],
        dtype=np.float64,
    )
    masked = np.ma.masked_invalid(array)
    valid_values = array[np.isfinite(array)]

    if valid_values.size == 0:
        return

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#D9D9D9")

    figure_width = max(10.5, 1.35 * len(column_labels) + 3.5)
    figure_height = max(4.8, 0.9 * len(row_labels) + 2.4)
    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )

    image_kwargs: dict[str, object] = {
        "cmap": cmap,
        "aspect": "auto",
    }
    if logarithmic and float(np.min(valid_values)) > 0.0:
        lower = max(float(np.min(valid_values)) * 0.8, 1e-8)
        upper = max(float(np.max(valid_values)) * 1.2, lower * 1.01)
        image_kwargs["norm"] = LogNorm(vmin=lower, vmax=upper)
    elif fixed_range is not None:
        image_kwargs["vmin"] = fixed_range[0]
        image_kwargs["vmax"] = fixed_range[1]

    image = axis.imshow(masked, **image_kwargs)
    axis.set_xticks(range(len(column_labels)), column_labels)
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.tick_params(axis="x", labelrotation=35)
    for label in axis.get_xticklabels():
        label.set_ha("right")

    axis.set_xlabel("Classifier")
    axis.set_ylabel("Pre-trained speaker embedding model")
    axis.set_title(
        title + "\nStrict chapter-held-out protocol",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    for row_index in range(array.shape[0]):
        for column_index in range(array.shape[1]):
            value = array[row_index, column_index]
            if not math.isfinite(float(value)):
                annotation = "NA"
                text_color = "#555555"
            else:
                annotation = format(float(value), annotation_format)
                normalized = image.norm(float(value))
                text_color = "white" if normalized < 0.42 else "black"

            axis.text(
                column_index,
                row_index,
                annotation,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=text_color,
            )

    colorbar = figure.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label(colorbar_label)
    axis.set_xticks(
        [value - 0.5 for value in range(1, len(column_labels))],
        minor=True,
    )
    axis.set_yticks(
        [value - 0.5 for value in range(1, len(row_labels))],
        minor=True,
    )
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)

    for extension in ("png", "pdf"):
        figure.savefig(
            output_stem.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def render_grouped_bar(
    values: list[list[float | None]],
    model_labels: list[str],
    classifier_labels: list[str],
    title: str,
    y_label: str,
    output_stem: Path,
    logarithmic: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axis = plt.subplots(figsize=(13.5, 6.5), constrained_layout=True)
    x_values = np.arange(len(classifier_labels), dtype=float)
    group_width = 0.82
    bar_width = group_width / max(1, len(model_labels))
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")
    positive_values: list[float] = []

    for model_index, (model_label, row_values) in enumerate(
        zip(model_labels, values)
    ):
        offsets = (
            x_values
            - group_width / 2
            + bar_width / 2
            + model_index * bar_width
        )
        heights = [
            float("nan") if value is None else float(value)
            for value in row_values
        ]
        positive_values.extend(
            value for value in heights if math.isfinite(value) and value > 0
        )
        axis.bar(
            offsets,
            heights,
            width=bar_width * 0.92,
            label=model_label,
            color=colors[model_index % len(colors)],
            edgecolor="white",
            linewidth=0.5,
        )

    if not positive_values:
        plt.close(figure)
        return

    if logarithmic and min(positive_values) > 0:
        axis.set_yscale("log")

    axis.set_xticks(x_values, classifier_labels, rotation=30, ha="right")
    axis.set_xlabel("Classifier")
    axis.set_ylabel(y_label)
    axis.set_title(
        title + "\nStrict chapter-held-out protocol",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend(
        title="Embedding model",
        ncols=min(4, len(model_labels)),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        frameon=False,
    )

    for extension in ("png", "pdf"):
        figure.savefig(
            output_stem.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def build_best_classifier_rows(
    runs: list[ValidatedRun],
    model_ids: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for model_id in model_ids:
        candidates = [run for run in runs if run.model_id == model_id]
        if not candidates:
            continue

        candidates.sort(
            key=lambda run: (
                -float(run.aggregates["accuracy"]["mean"]),
                -float(run.aggregates["f1_macro"]["mean"]),
                run.classifier_order,
            )
        )
        best = candidates[0]
        rows.append(
            {
                "model_order": best.model_order,
                "model_id": best.model_id,
                "model_display_name": best.model_display_name,
                "best_classifier_id": best.classifier_id,
                "best_classifier_display_name": best.classifier_display_name,
                "accuracy_mean": best.aggregates["accuracy"]["mean"],
                "accuracy_mean_percent": (
                    float(best.aggregates["accuracy"]["mean"]) * 100.0
                ),
                "accuracy_fold_sd": best.aggregates["accuracy"]["std"],
                "f1_macro_mean": best.aggregates["f1_macro"]["mean"],
                "f1_macro_mean_percent": (
                    float(best.aggregates["f1_macro"]["mean"]) * 100.0
                ),
                "classifier_fit_seconds_mean": best.aggregates[
                    "classifier_fit_seconds"
                ]["mean"],
                "prediction_ms_per_sample_mean": best.aggregates[
                    "prediction_ms_per_sample"
                ]["mean"],
                "selection_rule": (
                    "highest mean accuracy; ties broken by mean macro-F1, "
                    "then deterministic classifier order"
                ),
            }
        )

    return rows


def main() -> int:
    arguments = parse_arguments()
    tag = str(arguments.tag).strip()
    if not tag:
        raise ValueError("--tag cannot be empty.")

    protocol = load_protocol_context(arguments.protocol_file)
    model_ids = load_model_ids(arguments.config.resolve())
    baseline_root = arguments.baseline_root.resolve()
    strict_root = (
        baseline_root / "protocols" / protocol.protocol_id
    )

    if not strict_root.is_dir():
        raise FileNotFoundError(
            "No strict benchmark results exist for the supplied protocol:\n"
            f"{strict_root}"
        )

    if arguments.output_root is None:
        output_directory = (
            strict_root / "summary" / "classifier_benchmark" / tag
        )
    else:
        output_directory = arguments.output_root.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    runs, missing_runs, classifier_ids = discover_and_validate_runs(
        strict_root=strict_root,
        model_ids=model_ids,
        tag=tag,
        protocol=protocol,
        allow_missing=arguments.allow_missing,
    )

    long_rows = build_long_rows(runs, protocol)
    atomic_csv(
        output_directory / "classifier_benchmark_long.csv",
        LONG_CSV_FIELDS,
        long_rows,
    )

    missing_lookup = {
        (item["model_id"], item["classifier_id"]): item["reason"]
        for item in missing_runs
    }
    run_summaries = []
    for run in sorted(
        runs,
        key=lambda item: (item.model_order, item.classifier_order),
    ):
        run_summaries.append(
            {
                "model_order": run.model_order,
                "model_id": run.model_id,
                "model_display_name": run.model_display_name,
                "provider": run.provider,
                "architecture": run.architecture,
                "embedding_dimension": run.embedding_dimension,
                "classifier_order": run.classifier_order,
                "classifier_id": run.classifier_id,
                "classifier_display_name": run.classifier_display_name,
                "embedding_tag": run.tag,
                "aggregates": run.aggregates,
                "fold_metrics": run.fold_rows,
                "embedding_artifact_sha256": run.embedding_hashes,
                "run_directory": str(run.run_directory),
            }
        )

    atomic_json(
        output_directory / "classifier_benchmark_long.json",
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "status": "partial" if missing_runs else "complete",
            "protocol_label": STRICT_PROTOCOL_LABEL,
            "protocol": {
                "path": str(protocol.path),
                "protocol_id": protocol.protocol_id,
                "sha256": protocol.sha256,
                "ordered_assignment_sha256": (
                    protocol.ordered_assignment_sha256
                ),
                "rows": protocol.row_count,
                "speakers": protocol.speakers,
                "source_groups": protocol.source_groups,
                "fold_values": list(protocol.fold_values),
                "fold_counts": protocol.fold_counts,
            },
            "model_order": model_ids,
            "required_classifier_order": list(REQUIRED_CLASSIFIERS),
            "included_classifier_order": classifier_ids,
            "allow_missing": bool(arguments.allow_missing),
            "missing_runs": missing_runs,
            "runs": run_summaries,
        },
    )

    accuracy_values = matrix_values(
        runs,
        model_ids,
        classifier_ids,
        "accuracy",
        multiplier=100.0,
    )
    f1_values = matrix_values(
        runs,
        model_ids,
        classifier_ids,
        "f1_macro",
        multiplier=100.0,
    )
    training_values = matrix_values(
        runs,
        model_ids,
        classifier_ids,
        "classifier_fit_seconds",
    )
    prediction_values = matrix_values(
        runs,
        model_ids,
        classifier_ids,
        "prediction_ms_per_sample",
    )

    matrix_artifacts = (
        ("accuracy_matrix.csv", accuracy_values),
        ("macro_f1_matrix.csv", f1_values),
        ("training_time_seconds_matrix.csv", training_values),
        ("prediction_time_ms_per_sample_matrix.csv", prediction_values),
    )
    for filename, values in matrix_artifacts:
        fields, rows = matrix_rows(model_ids, classifier_ids, values)
        atomic_csv(output_directory / filename, fields, rows)

    model_labels = [MODEL_LABELS.get(model_id, model_id) for model_id in model_ids]
    classifier_labels = [
        CLASSIFIER_LABELS[classifier_id] for classifier_id in classifier_ids
    ]

    render_heatmap(
        accuracy_values,
        model_labels,
        classifier_labels,
        "Recognition accuracy by embedding model and classifier",
        "Mean accuracy (%)",
        output_directory / "accuracy_heatmap",
        ".2f",
        fixed_range=(0.0, 100.0),
    )
    render_heatmap(
        f1_values,
        model_labels,
        classifier_labels,
        "Macro-F1 by embedding model and classifier",
        "Mean macro-F1 (%)",
        output_directory / "macro_f1_heatmap",
        ".2f",
        fixed_range=(0.0, 100.0),
    )
    render_heatmap(
        training_values,
        model_labels,
        classifier_labels,
        "Classifier training time by embedding model",
        "Mean classifier fit time (s, log scale)",
        output_directory / "training_time_heatmap",
        ".3g",
        logarithmic=True,
    )
    render_heatmap(
        prediction_values,
        model_labels,
        classifier_labels,
        "Classifier prediction latency by embedding model",
        "Mean prediction time (ms/sample, log scale)",
        output_directory / "prediction_time_heatmap",
        ".3g",
        logarithmic=True,
    )
    render_grouped_bar(
        training_values,
        model_labels,
        classifier_labels,
        "Classifier training-time comparison",
        "Mean classifier fit time (seconds, log scale)",
        output_directory / "training_time_bar",
        logarithmic=True,
    )
    render_grouped_bar(
        prediction_values,
        model_labels,
        classifier_labels,
        "Classifier prediction-latency comparison",
        "Mean prediction time (ms/sample, log scale)",
        output_directory / "prediction_time_bar",
        logarithmic=True,
    )

    best_rows = build_best_classifier_rows(runs, model_ids)
    best_fields = (
        "model_order",
        "model_id",
        "model_display_name",
        "best_classifier_id",
        "best_classifier_display_name",
        "accuracy_mean",
        "accuracy_mean_percent",
        "accuracy_fold_sd",
        "f1_macro_mean",
        "f1_macro_mean_percent",
        "classifier_fit_seconds_mean",
        "prediction_ms_per_sample_mean",
        "selection_rule",
    )
    atomic_csv(
        output_directory / "best_classifier_per_model.csv",
        best_fields,
        best_rows,
    )
    atomic_json(
        output_directory / "best_classifier_per_model.json",
        {
            "protocol_label": STRICT_PROTOCOL_LABEL,
            "status": "partial" if missing_runs else "complete",
            "selection_rule": (
                "highest mean accuracy; ties broken by mean macro-F1, then "
                "deterministic classifier order"
            ),
            "rows": best_rows,
        },
    )

    atomic_json(
        output_directory / "benchmark_metadata.json",
        {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "status": "partial" if missing_runs else "complete",
            "publication_ready_complete_grid": not missing_runs,
            "protocol_label": STRICT_PROTOCOL_LABEL,
            "protocol_id": protocol.protocol_id,
            "protocol_path": str(protocol.path),
            "protocol_sha256": protocol.sha256,
            "ordered_assignment_sha256": (
                protocol.ordered_assignment_sha256
            ),
            "models": model_ids,
            "required_classifiers": list(REQUIRED_CLASSIFIERS),
            "included_classifiers": classifier_ids,
            "complete_runs": len(runs),
            "expected_runs": len(model_ids) * len(classifier_ids),
            "missing_runs": missing_runs,
            "output_directory": str(output_directory.resolve()),
        },
    )

    print("=" * 78)
    print("STRICT CLASSIFIER BENCHMARK SUMMARY")
    print("=" * 78)
    print(f"Protocol      : {protocol.protocol_id}")
    print(f"Models        : {len(model_ids)}")
    print(f"Classifiers   : {len(classifier_ids)}")
    print(f"Complete runs : {len(runs)}")
    print(f"Missing runs  : {len(missing_runs)}")
    print(
        "Status        : "
        + ("PARTIAL (--allow-missing)" if missing_runs else "COMPLETE")
    )
    print(f"Artifacts     : {output_directory.resolve()}")
    print("=" * 78)

    if missing_runs:
        print("\nMissing/incomplete grid cells:")
        for model_id in model_ids:
            for classifier_id in classifier_ids:
                reason = missing_lookup.get((model_id, classifier_id))
                if reason:
                    print(f"- {model_id} / {classifier_id}: {reason}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nSummary interrupted.")
        raise SystemExit(130)
    except Exception as error:
        print("\n" + "=" * 78)
        print("CLASSIFIER BENCHMARK SUMMARY FAILED")
        print("=" * 78)
        print(str(error))
        print("=" * 78)
        raise SystemExit(1)
