"""Create publication-ready summaries of completed baseline CV experiments.

The script consolidates one classifier's results across the four embedding
models listed in ``configs/research_baseline.json``.  Before aggregating any
values, it verifies that every model used the same source manifest, persisted
fold identifiers, sample count, and speaker count.

Performance error bars in the figures are the sample standard deviation over
the five persisted folds.  Error bars are clipped to the physically meaningful
0--100% range only for plotting.  The unmodified means, standard deviations,
and Student-t confidence intervals remain available in the JSON and CSV
artifacts.
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
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_ROOT = PROJECT_ROOT / "research_results" / "baseline_cv"
DEFAULT_EMBEDDINGS_ROOT = (
    PROJECT_ROOT
    / "research_results"
    / "embeddings"
    / "librispeech_dev_clean"
)
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "research_baseline.json"
DEFAULT_CLASSIFIER = "linear_svm"
EXPECTED_METRICS = ("accuracy", "f1_macro")

CSV_FIELDS = (
    "rank_accuracy",
    "model_id",
    "model_display_name",
    "provider",
    "architecture",
    "pretrained_name",
    "classifier",
    "embedding_tag",
    "evaluation_protocol_id",
    "evaluation_protocol_sha256",
    "samples",
    "speakers",
    "folds",
    "embedding_dimension",
    "accuracy_mean",
    "accuracy_fold_sd",
    "accuracy_ci95_half_width",
    "accuracy_ci95_lower_raw",
    "accuracy_ci95_upper_raw",
    "accuracy_mean_percent",
    "accuracy_fold_sd_percent",
    "f1_macro_mean",
    "f1_macro_fold_sd",
    "f1_macro_ci95_half_width",
    "f1_macro_ci95_lower_raw",
    "f1_macro_ci95_upper_raw",
    "f1_macro_mean_percent",
    "f1_macro_fold_sd_percent",
    "warm_extraction_median_ms",
    "warm_extraction_mean_ms",
    "warm_extraction_p95_ms",
    "cold_first_success_ms",
    "embedding_coverage_percent",
    "manifest_sha256",
)

MODEL_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
)


@dataclass(frozen=True)
class ProtocolIdentity:
    """Fields that must be identical for a fair cross-model comparison."""

    manifest_sha256: str
    evaluation_protocol_id: str | None
    evaluation_protocol_sha256: str | None
    ordered_assignment_sha256: str | None
    fold_values: tuple[int, ...]
    samples: int
    speakers: int


@dataclass(frozen=True)
class EvaluationProtocolContext:
    """Canonical identity derived from an optional external protocol CSV."""

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
class ModelSummary:
    """Validated model result used by the consolidated table and figures."""

    model_id: str
    display_name: str
    provider: str
    architecture: str
    pretrained_name: str
    classifier: str
    embedding_tag: str
    evaluation_protocol_id: str | None
    evaluation_protocol_sha256: str | None
    ordered_assignment_sha256: str | None
    samples: int
    speakers: int
    fold_values: list[int]
    embedding_dimension: int
    manifest_sha256: str
    accuracy: dict[str, Any]
    f1_macro: dict[str, Any]
    warm_timing_ms: dict[str, float]
    cold_first_success_ms: float
    coverage_percent: float
    run_metadata_path: str
    aggregate_metrics_path: str
    embedding_metadata_path: str
    fold_values_raw: dict[str, list[float]]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate the four fixed-fold embedding-model baselines into "
            "publication-ready tables and figures."
        )
    )
    parser.add_argument(
        "--classifier",
        default=DEFAULT_CLASSIFIER,
        help="Completed classifier result to summarize (default: linear_svm).",
    )
    parser.add_argument(
        "--tag",
        default="full",
        help=(
            "Embedding artifact tag used by the completed CV runs "
            "(default: full)."
        ),
    )
    parser.add_argument(
        "--protocol-file",
        type=Path,
        default=None,
        help=(
            "Optional external protocol CSV used by run_baseline_cv. When "
            "provided, results and summaries are resolved inside the same "
            "protocol-specific tree."
        ),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_BASELINE_ROOT,
        help="Root containing <model_id>/<classifier>/ results.",
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=DEFAULT_EMBEDDINGS_ROOT,
        help="Fallback root containing <model_id>/full/metadata.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Research config that declares the four embedding model IDs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional destination. The default is "
            "<baseline-root>/summary/<classifier>."
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
    """Mirror run_baseline_cv.safe_protocol_name exactly."""

    stem = path.stem.lower()

    if stem.endswith("_protocol"):
        stem = stem[: -len("_protocol")]

    stem = re.sub(r"[^a-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")

    if not stem:
        stem = "external_protocol"

    return f"{stem}__{protocol_hash[:12]}"


def load_protocol_context(path: Path) -> EvaluationProtocolContext:
    """Hash and validate an external protocol using the runner's rules."""

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
                    f"{fold_text!r}."
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

    fold_values = tuple(
        sorted({int(row["fold"]) for row in rows})
    )

    if len(fold_values) < 2:
        raise ValueError(
            "Evaluation protocol must contain at least two folds."
        )

    speakers = {
        str(row["speaker_label"])
        for row in rows
    }
    assignment_digest = hashlib.sha256()

    for row in rows:
        canonical = (
            f"{row['clip_id']}\t{row['speaker_label']}\t"
            f"{row['source_group']}\t{row['fold']}\n"
        )
        assignment_digest.update(canonical.encode("utf-8"))

    return EvaluationProtocolContext(
        path=protocol_path,
        sha256=protocol_hash,
        protocol_id=safe_protocol_name(
            protocol_path,
            protocol_hash,
        ),
        ordered_assignment_sha256=assignment_digest.hexdigest(),
        row_count=len(rows),
        fold_values=fold_values,
        fold_counts={
            str(fold): sum(
                int(row["fold"]) == fold for row in rows
            )
            for fold in fold_values
        },
        speakers=len(speakers),
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
    fieldnames: Iterable[str],
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


def require_mapping(
    parent: dict[str, Any],
    key: str,
    source: Path,
) -> dict[str, Any]:
    value = parent.get(key)

    if not isinstance(value, dict):
        raise ValueError(f"Missing object '{key}' in:\n{source}")

    return value


def require_number(
    parent: dict[str, Any],
    key: str,
    source: Path,
) -> float:
    value = parent.get(key)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric '{key}' in:\n{source}")

    numeric = float(value)

    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite '{key}' in:\n{source}")

    return numeric


def load_model_ids(config_path: Path) -> list[str]:
    config = read_json(config_path)
    model_ids = config.get("embedding_models")

    if not isinstance(model_ids, list) or not model_ids:
        raise ValueError(
            "Research config must contain a non-empty "
            f"'embedding_models' list:\n{config_path}"
        )

    cleaned: list[str] = []

    for model_id in model_ids:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(
                f"Invalid model ID in research config:\n{config_path}"
            )

        if model_id in cleaned:
            raise ValueError(
                f"Duplicate model ID '{model_id}' in:\n{config_path}"
            )

        cleaned.append(model_id)

    if len(cleaned) != 4:
        raise ValueError(
            "This baseline comparison requires exactly four embedding models; "
            f"the config declares {len(cleaned)}:\n{config_path}"
        )

    return cleaned


def load_fold_metrics(
    path: Path,
    expected_folds: tuple[int, ...],
) -> dict[str, list[float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Fold metrics were not found:\n{path}")

    metric_values = {name: [] for name in EXPECTED_METRICS}
    observed_folds: list[int] = []

    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError(f"Fold metrics CSV has no header:\n{path}")

        required = {"fold", *EXPECTED_METRICS}
        missing = sorted(required.difference(reader.fieldnames))

        if missing:
            raise ValueError(
                f"Fold metrics CSV is missing {missing}:\n{path}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                fold = int(row["fold"])
                values = {
                    name: float(row[name])
                    for name in EXPECTED_METRICS
                }
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid value on row {row_number} of:\n{path}"
                ) from error

            if any(not math.isfinite(value) for value in values.values()):
                raise ValueError(
                    f"Non-finite metric on row {row_number} of:\n{path}"
                )

            observed_folds.append(fold)

            for name, value in values.items():
                metric_values[name].append(value)

    if tuple(observed_folds) != expected_folds:
        raise ValueError(
            "Fold order in fold_metrics.csv does not match run metadata.\n"
            f"Expected: {list(expected_folds)}\n"
            f"Observed: {observed_folds}\n"
            f"File: {path}"
        )

    return metric_values


def verify_aggregate_metric(
    metric_name: str,
    aggregate: dict[str, Any],
    fold_values: list[float],
    source: Path,
) -> None:
    expected_n = len(fold_values)
    observed_n = int(require_number(aggregate, "n", source))

    if observed_n != expected_n:
        raise ValueError(
            f"{metric_name} aggregate n={observed_n}, expected {expected_n}:"
            f"\n{source}"
        )

    recomputed_mean = statistics.fmean(fold_values)
    recomputed_sd = statistics.stdev(fold_values) if expected_n > 1 else 0.0
    stored_mean = require_number(aggregate, "mean", source)
    stored_sd = require_number(aggregate, "std", source)

    if not math.isclose(
        stored_mean,
        recomputed_mean,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Stored {metric_name} mean does not match fold_metrics.csv:"
            f"\n{source}"
        )

    if not math.isclose(
        stored_sd,
        recomputed_sd,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Stored {metric_name} SD does not match fold_metrics.csv:"
            f"\n{source}"
        )

    for key in (
        "ci95_half_width",
        "ci95_lower",
        "ci95_upper",
        "minimum",
        "maximum",
    ):
        require_number(aggregate, key, source)


def resolve_embedding_metadata(
    run_metadata: dict[str, Any],
    model_id: str,
    tag: str,
    embeddings_root: Path,
    run_metadata_path: Path,
) -> tuple[Path, dict[str, Any]]:
    embedding_artifacts = require_mapping(
        run_metadata,
        "embedding_artifacts",
        run_metadata_path,
    )
    embedded_source_metadata = require_mapping(
        embedding_artifacts,
        "source_metadata",
        run_metadata_path,
    )

    if embedded_source_metadata.get("tag") != tag:
        raise ValueError(
            f"CV run embedding tag mismatch for {model_id}. Expected "
            f"'{tag}' in:\n{run_metadata_path}"
        )

    directory_value = embedding_artifacts.get("directory")
    candidates: list[Path] = []

    if isinstance(directory_value, str) and directory_value.strip():
        candidates.append(Path(directory_value) / "metadata.json")

    candidates.append(
        embeddings_root / model_id / tag / "metadata.json"
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(), read_json(candidate)

    formatted = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Embedding metadata could not be located. Checked:\n"
        f"{formatted}"
    )


def load_model_summary(
    model_id: str,
    classifier: str,
    tag: str,
    baseline_root: Path,
    embeddings_root: Path,
    evaluation_protocol: EvaluationProtocolContext | None,
) -> tuple[ModelSummary, ProtocolIdentity]:
    if evaluation_protocol is None:
        result_directory = baseline_root / model_id / classifier / tag
    else:
        result_directory = (
            baseline_root
            / "protocols"
            / evaluation_protocol.protocol_id
            / model_id
            / classifier
            / tag
        )
    run_metadata_path = result_directory / "run_metadata.json"
    aggregate_path = result_directory / "aggregate_metrics.json"
    fold_metrics_path = result_directory / "fold_metrics.csv"

    run_metadata = read_json(run_metadata_path)
    aggregate_metadata = read_json(aggregate_path)

    if run_metadata.get("model_id") != model_id:
        raise ValueError(
            f"Model ID mismatch in:\n{run_metadata_path}"
        )

    classifier_metadata = require_mapping(
        run_metadata,
        "classifier",
        run_metadata_path,
    )

    if classifier_metadata.get("name") != classifier:
        raise ValueError(
            f"Classifier mismatch in:\n{run_metadata_path}"
        )

    model_metadata = require_mapping(
        run_metadata,
        "model",
        run_metadata_path,
    )
    protocol = require_mapping(
        run_metadata,
        "protocol",
        run_metadata_path,
    )
    manifest = require_mapping(
        run_metadata,
        "manifest",
        run_metadata_path,
    )

    if evaluation_protocol is None:
        if protocol.get("mode") == "external_protocol_file":
            raise ValueError(
                "A protocol-specific result cannot be summarized without "
                f"--protocol-file:\n{run_metadata_path}"
            )
    else:
        expected_protocol_values = {
            "mode": "external_protocol_file",
            "protocol_id": evaluation_protocol.protocol_id,
            "sha256": evaluation_protocol.sha256,
            "ordered_assignment_sha256": (
                evaluation_protocol.ordered_assignment_sha256
            ),
            "row_count": evaluation_protocol.row_count,
            "fold_values": list(evaluation_protocol.fold_values),
            "fold_counts": evaluation_protocol.fold_counts,
            "speakers": evaluation_protocol.speakers,
            "source_groups": evaluation_protocol.source_groups,
        }

        for key, expected_value in expected_protocol_values.items():
            if protocol.get(key) != expected_value:
                raise ValueError(
                    f"External protocol metadata '{key}' does not match "
                    f"the supplied protocol file for {model_id}.\n"
                    f"Expected: {expected_value!r}\n"
                    f"Observed: {protocol.get(key)!r}\n"
                    f"Run: {run_metadata_path}"
                )

    fold_value_source = protocol.get("fold_values")

    if (
        not isinstance(fold_value_source, list)
        or not fold_value_source
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in fold_value_source
        )
    ):
        raise ValueError(
            f"Invalid protocol.fold_values in:\n{run_metadata_path}"
        )

    fold_values = tuple(int(value) for value in fold_value_source)
    fold_count = int(
        require_number(protocol, "number_of_folds", run_metadata_path)
    )

    if fold_count != len(fold_values):
        raise ValueError(
            "number_of_folds does not match fold_values in:\n"
            f"{run_metadata_path}"
        )

    samples = int(require_number(protocol, "samples", run_metadata_path))
    speakers = int(require_number(protocol, "speakers", run_metadata_path))
    manifest_rows = int(
        require_number(manifest, "row_count", run_metadata_path)
    )
    manifest_sha = str(manifest.get("sha256", "")).strip().lower()

    if not manifest_sha:
        raise ValueError(
            f"Missing manifest SHA-256 in:\n{run_metadata_path}"
        )

    if evaluation_protocol is None and manifest_rows != samples:
        raise ValueError(
            "Manifest row_count does not match protocol samples in:\n"
            f"{run_metadata_path}"
        )

    if (
        evaluation_protocol is not None
        and samples != evaluation_protocol.row_count
    ):
        raise ValueError(
            f"Protocol sample count differs for {model_id}: "
            f"{samples} != {evaluation_protocol.row_count}"
        )

    fold_metrics = load_fold_metrics(fold_metrics_path, fold_values)
    aggregate_values = require_mapping(
        aggregate_metadata,
        "values",
        aggregate_path,
    )

    for metric_name in EXPECTED_METRICS:
        metric_aggregate = require_mapping(
            aggregate_values,
            metric_name,
            aggregate_path,
        )
        verify_aggregate_metric(
            metric_name,
            metric_aggregate,
            fold_metrics[metric_name],
            aggregate_path,
        )

    embedding_metadata_path, embedding_metadata = (
        resolve_embedding_metadata(
            run_metadata,
            model_id,
            tag,
            embeddings_root,
            run_metadata_path,
        )
    )

    if embedding_metadata.get("model_id") != model_id:
        raise ValueError(
            f"Embedding model ID mismatch in:\n{embedding_metadata_path}"
        )

    if embedding_metadata.get("tag") != tag:
        raise ValueError(
            f"Embedding tag mismatch for {model_id}. Expected '{tag}' in:\n"
            f"{embedding_metadata_path}"
        )

    embedding_manifest_sha = str(
        embedding_metadata.get("manifest_sha256", "")
    ).strip().lower()

    if embedding_manifest_sha != manifest_sha:
        raise ValueError(
            "Embedding and baseline manifest hashes differ for "
            f"{model_id}.\nEmbedding: {embedding_metadata_path}\n"
            f"Baseline: {run_metadata_path}"
        )

    successful_rows = int(
        require_number(
            embedding_metadata,
            "successful_rows",
            embedding_metadata_path,
        )
    )

    if evaluation_protocol is None:
        if successful_rows != samples:
            raise ValueError(
                f"{model_id} has {successful_rows} successful embeddings but "
                f"the baseline reports {samples} samples."
            )
    else:
        available_samples = int(
            require_number(
                protocol,
                "available_artifact_samples",
                run_metadata_path,
            )
        )

        if successful_rows != available_samples:
            raise ValueError(
                f"{model_id} embedding metadata reports {successful_rows} "
                f"successful rows but the protocol run reports "
                f"{available_samples} available artifact rows."
            )

        if successful_rows < samples:
            raise ValueError(
                f"{model_id} does not contain enough embeddings for the "
                f"external protocol: {successful_rows} < {samples}."
            )

    embedding_folds = int(
        require_number(
            embedding_metadata,
            "number_of_folds",
            embedding_metadata_path,
        )
    )

    if evaluation_protocol is None and embedding_folds != fold_count:
        raise ValueError(
            f"Embedding fold count differs for {model_id}: "
            f"{embedding_folds} != {fold_count}"
        )

    warm_timing = require_mapping(
        embedding_metadata,
        "warm_success_timing_ms",
        embedding_metadata_path,
    )
    validated_warm_timing = {
        key: require_number(warm_timing, key, embedding_metadata_path)
        for key in (
            "minimum",
            "median",
            "mean",
            "p95",
            "maximum",
            "total",
        )
    }

    identity = ProtocolIdentity(
        manifest_sha256=manifest_sha,
        evaluation_protocol_id=(
            evaluation_protocol.protocol_id
            if evaluation_protocol is not None
            else None
        ),
        evaluation_protocol_sha256=(
            evaluation_protocol.sha256
            if evaluation_protocol is not None
            else None
        ),
        ordered_assignment_sha256=(
            evaluation_protocol.ordered_assignment_sha256
            if evaluation_protocol is not None
            else None
        ),
        fold_values=fold_values,
        samples=samples,
        speakers=speakers,
    )

    summary = ModelSummary(
        model_id=model_id,
        display_name=str(
            model_metadata.get("display_name") or model_id
        ),
        provider=str(model_metadata.get("provider") or ""),
        architecture=str(model_metadata.get("architecture") or ""),
        pretrained_name=str(
            model_metadata.get("pretrained_name")
            or model_metadata.get("source")
            or ""
        ),
        classifier=classifier,
        embedding_tag=tag,
        evaluation_protocol_id=identity.evaluation_protocol_id,
        evaluation_protocol_sha256=(
            identity.evaluation_protocol_sha256
        ),
        ordered_assignment_sha256=(
            identity.ordered_assignment_sha256
        ),
        samples=samples,
        speakers=speakers,
        fold_values=list(fold_values),
        embedding_dimension=int(
            require_number(
                protocol,
                "embedding_dimension",
                run_metadata_path,
            )
        ),
        manifest_sha256=manifest_sha,
        accuracy=dict(aggregate_values["accuracy"]),
        f1_macro=dict(aggregate_values["f1_macro"]),
        warm_timing_ms=validated_warm_timing,
        cold_first_success_ms=require_number(
            embedding_metadata,
            "cold_first_success_ms",
            embedding_metadata_path,
        ),
        coverage_percent=require_number(
            embedding_metadata,
            "coverage_percent",
            embedding_metadata_path,
        ),
        run_metadata_path=str(run_metadata_path.resolve()),
        aggregate_metrics_path=str(aggregate_path.resolve()),
        embedding_metadata_path=str(embedding_metadata_path),
        fold_values_raw=fold_metrics,
    )

    return summary, identity


def verify_common_protocol(
    identities: dict[str, ProtocolIdentity],
) -> ProtocolIdentity:
    if not identities:
        raise ValueError("No completed model results were loaded.")

    first_model_id = next(iter(identities))
    reference = identities[first_model_id]

    for model_id, identity in identities.items():
        differences: list[str] = []

        if identity.manifest_sha256 != reference.manifest_sha256:
            differences.append("manifest SHA-256")

        if (
            identity.evaluation_protocol_id
            != reference.evaluation_protocol_id
        ):
            differences.append("evaluation protocol ID")

        if (
            identity.evaluation_protocol_sha256
            != reference.evaluation_protocol_sha256
        ):
            differences.append("evaluation protocol SHA-256")

        if (
            identity.ordered_assignment_sha256
            != reference.ordered_assignment_sha256
        ):
            differences.append("ordered protocol assignment SHA-256")

        if identity.fold_values != reference.fold_values:
            differences.append("fold values")

        if identity.samples != reference.samples:
            differences.append("sample count")

        if identity.speakers != reference.speakers:
            differences.append("speaker count")

        if differences:
            raise ValueError(
                f"Protocol mismatch between {first_model_id} and {model_id}: "
                + ", ".join(differences)
            )

    return reference


def model_to_json(summary: ModelSummary, rank: int) -> dict[str, Any]:
    return {
        "accuracy_rank": rank,
        "model_id": summary.model_id,
        "display_name": summary.display_name,
        "provider": summary.provider,
        "architecture": summary.architecture,
        "pretrained_name": summary.pretrained_name,
        "classifier": summary.classifier,
        "embedding_tag": summary.embedding_tag,
        "evaluation_protocol": (
            {
                "protocol_id": summary.evaluation_protocol_id,
                "sha256": summary.evaluation_protocol_sha256,
                "ordered_assignment_sha256": (
                    summary.ordered_assignment_sha256
                ),
            }
            if summary.evaluation_protocol_id is not None
            else None
        ),
        "protocol": {
            "samples": summary.samples,
            "speakers": summary.speakers,
            "fold_values": summary.fold_values,
            "embedding_dimension": summary.embedding_dimension,
            "manifest_sha256": summary.manifest_sha256,
        },
        "performance": {
            "accuracy": summary.accuracy,
            "f1_macro": summary.f1_macro,
            "fold_values": summary.fold_values_raw,
        },
        "embedding_extraction_timing_ms": {
            "cold_first_success": summary.cold_first_success_ms,
            "warm_success": summary.warm_timing_ms,
            "coverage_percent": summary.coverage_percent,
        },
        "sources": {
            "run_metadata": summary.run_metadata_path,
            "aggregate_metrics": summary.aggregate_metrics_path,
            "embedding_metadata": summary.embedding_metadata_path,
        },
    }


def model_to_csv(summary: ModelSummary, rank: int) -> dict[str, object]:
    accuracy = summary.accuracy
    f1_macro = summary.f1_macro

    return {
        "rank_accuracy": rank,
        "model_id": summary.model_id,
        "model_display_name": summary.display_name,
        "provider": summary.provider,
        "architecture": summary.architecture,
        "pretrained_name": summary.pretrained_name,
        "classifier": summary.classifier,
        "embedding_tag": summary.embedding_tag,
        "evaluation_protocol_id": (
            summary.evaluation_protocol_id or ""
        ),
        "evaluation_protocol_sha256": (
            summary.evaluation_protocol_sha256 or ""
        ),
        "samples": summary.samples,
        "speakers": summary.speakers,
        "folds": len(summary.fold_values),
        "embedding_dimension": summary.embedding_dimension,
        "accuracy_mean": accuracy["mean"],
        "accuracy_fold_sd": accuracy["std"],
        "accuracy_ci95_half_width": accuracy["ci95_half_width"],
        "accuracy_ci95_lower_raw": accuracy["ci95_lower"],
        "accuracy_ci95_upper_raw": accuracy["ci95_upper"],
        "accuracy_mean_percent": float(accuracy["mean"]) * 100.0,
        "accuracy_fold_sd_percent": float(accuracy["std"]) * 100.0,
        "f1_macro_mean": f1_macro["mean"],
        "f1_macro_fold_sd": f1_macro["std"],
        "f1_macro_ci95_half_width": f1_macro["ci95_half_width"],
        "f1_macro_ci95_lower_raw": f1_macro["ci95_lower"],
        "f1_macro_ci95_upper_raw": f1_macro["ci95_upper"],
        "f1_macro_mean_percent": float(f1_macro["mean"]) * 100.0,
        "f1_macro_fold_sd_percent": float(f1_macro["std"]) * 100.0,
        "warm_extraction_median_ms": summary.warm_timing_ms["median"],
        "warm_extraction_mean_ms": summary.warm_timing_ms["mean"],
        "warm_extraction_p95_ms": summary.warm_timing_ms["p95"],
        "cold_first_success_ms": summary.cold_first_success_ms,
        "embedding_coverage_percent": summary.coverage_percent,
        "manifest_sha256": summary.manifest_sha256,
    }


def import_plotting() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as error:
        raise RuntimeError(
            "Matplotlib is required to generate publication figures. "
            "Install it in the project environment and rerun this script."
        ) from error

    return plt, PercentFormatter


def configure_plot_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def short_model_names(summaries: list[ModelSummary]) -> list[str]:
    names: list[str] = []

    for summary in summaries:
        if summary.model_id == "speechbrain_ecapa":
            names.append("ECAPA-TDNN")
        elif summary.model_id == "speechbrain_xvector":
            names.append("X-Vector")
        elif summary.model_id == "wavlm_base_plus_sv":
            names.append("WavLM")
        elif summary.model_id == "unispeech_sat_base_plus_sv":
            names.append("UniSpeech-SAT")
        else:
            names.append(summary.display_name)

    return names


def clipped_sd_errors(
    means: list[float],
    standard_deviations: list[float],
) -> list[list[float]]:
    lower = [
        min(max(sd, 0.0), max(mean, 0.0))
        for mean, sd in zip(means, standard_deviations)
    ]
    upper = [
        min(max(sd, 0.0), max(1.0 - mean, 0.0))
        for mean, sd in zip(means, standard_deviations)
    ]
    return [lower, upper]


def save_performance_chart(
    plt: Any,
    PercentFormatter: Any,
    summaries: list[ModelSummary],
    metric_key: str,
    title: str,
    ylabel: str,
    output_stem: Path,
    fold_count: int,
) -> None:
    names = short_model_names(summaries)
    means = [
        float(getattr(summary, metric_key)["mean"])
        for summary in summaries
    ]
    standard_deviations = [
        float(getattr(summary, metric_key)["std"])
        for summary in summaries
    ]
    positions = list(range(len(summaries)))

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    bars = axis.bar(
        positions,
        means,
        width=0.66,
        color=MODEL_COLORS[: len(summaries)],
        edgecolor="#222222",
        linewidth=0.7,
        yerr=clipped_sd_errors(means, standard_deviations),
        capsize=5,
        error_kw={
            "elinewidth": 1.1,
            "capthick": 1.1,
            "ecolor": "#222222",
        },
    )
    axis.set_title(title, pad=12)
    axis.set_ylabel(ylabel)
    axis.set_xticks(positions, names)
    axis.set_ylim(0.0, 1.035)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    for bar, mean in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(mean + 0.012, 1.018),
            f"{mean * 100.0:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    figure.text(
        0.5,
        0.012,
        (
            f"Bars show {fold_count}-fold means; error bars show fold SD. "
            "Visual error bounds are clipped to 0–100%."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    figure.tight_layout(rect=(0.02, 0.055, 0.98, 0.98))
    figure.savefig(output_stem.with_suffix(".png"), dpi=300)
    figure.savefig(output_stem.with_suffix(".pdf"))
    plt.close(figure)


def save_timing_chart(
    plt: Any,
    summaries: list[ModelSummary],
    output_stem: Path,
) -> None:
    names = short_model_names(summaries)
    medians = [
        summary.warm_timing_ms["median"]
        for summary in summaries
    ]
    means = [
        summary.warm_timing_ms["mean"]
        for summary in summaries
    ]
    positions = list(range(len(summaries)))
    width = 0.36

    figure, axis = plt.subplots(figsize=(7.6, 4.9))
    median_bars = axis.bar(
        [position - width / 2 for position in positions],
        medians,
        width=width,
        label="Warm median",
        color="#56B4E9",
        edgecolor="#222222",
        linewidth=0.7,
    )
    mean_bars = axis.bar(
        [position + width / 2 for position in positions],
        means,
        width=width,
        label="Warm mean",
        color="#D55E00",
        edgecolor="#222222",
        linewidth=0.7,
        hatch="//",
    )
    axis.set_title("Warm Speaker-Embedding Extraction Time", pad=12)
    axis.set_ylabel("Time per utterance (ms)")
    axis.set_xticks(positions, names)
    axis.set_ylim(0.0, max(means) * 1.20)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="upper left")

    for bars in (median_bars, mean_bars):
        for bar in bars:
            value = float(bar.get_height())
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(means) * 0.018,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    figure.text(
        0.5,
        0.012,
        (
            "Timing values come from each model's full embedding-extraction "
            "metadata; the first cold-start success is excluded."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    figure.tight_layout(rect=(0.02, 0.055, 0.98, 0.98))
    figure.savefig(output_stem.with_suffix(".png"), dpi=300)
    figure.savefig(output_stem.with_suffix(".pdf"))
    plt.close(figure)


def print_summary(
    summaries: list[ModelSummary],
    output_directory: Path,
) -> None:
    print()
    print("=" * 78)
    print("BASELINE MODEL COMPARISON")
    print("=" * 78)
    print(
        f"{'Model':<18}"
        f"{'Accuracy':>14}"
        f"{'Macro F1':>14}"
        f"{'Warm med.':>14}"
        f"{'Warm mean':>14}"
    )
    print("-" * 78)

    for summary in summaries:
        name = short_model_names([summary])[0]
        print(
            f"{name:<18}"
            f"{float(summary.accuracy['mean']) * 100:>13.2f}%"
            f"{float(summary.f1_macro['mean']) * 100:>13.2f}%"
            f"{summary.warm_timing_ms['median']:>11.2f} ms"
            f"{summary.warm_timing_ms['mean']:>11.2f} ms"
        )

    print("=" * 78)
    print(f"Artifacts: {output_directory}")
    print("=" * 78)


def main() -> int:
    arguments = parse_arguments()
    model_ids = load_model_ids(arguments.config.resolve())
    baseline_root = arguments.baseline_root.resolve()
    embeddings_root = arguments.embeddings_root.resolve()
    evaluation_protocol = (
        load_protocol_context(arguments.protocol_file)
        if arguments.protocol_file is not None
        else None
    )

    if evaluation_protocol is None:
        default_output_directory = (
            baseline_root
            / "summary"
            / arguments.classifier
            / arguments.tag
        )
    else:
        default_output_directory = (
            baseline_root
            / "protocols"
            / evaluation_protocol.protocol_id
            / "summary"
            / arguments.classifier
            / arguments.tag
        )

    output_directory = (
        arguments.output_root.resolve()
        if arguments.output_root is not None
        else default_output_directory
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    summaries: list[ModelSummary] = []
    identities: dict[str, ProtocolIdentity] = {}

    for model_id in model_ids:
        summary, identity = load_model_summary(
            model_id=model_id,
            classifier=arguments.classifier,
            tag=arguments.tag,
            baseline_root=baseline_root,
            embeddings_root=embeddings_root,
            evaluation_protocol=evaluation_protocol,
        )
        summaries.append(summary)
        identities[model_id] = identity

    common_protocol = verify_common_protocol(identities)
    ranked = sorted(
        summaries,
        key=lambda item: (
            -float(item.accuracy["mean"]),
            -float(item.f1_macro["mean"]),
            item.model_id,
        ),
    )

    csv_rows = [
        model_to_csv(summary, rank)
        for rank, summary in enumerate(ranked, start=1)
    ]
    json_models = [
        model_to_json(summary, rank)
        for rank, summary in enumerate(ranked, start=1)
    ]

    atomic_csv(
        output_directory / "model_comparison.csv",
        CSV_FIELDS,
        csv_rows,
    )

    summary_json = {
        "schema_version": 1,
        "experiment": (
            "four_model_external_protocol_baseline_summary"
            if evaluation_protocol is not None
            else "four_model_fixed_fold_baseline_summary"
        ),
        "generated_at_utc": utc_now(),
        "classifier": arguments.classifier,
        "embedding_tag": arguments.tag,
        "evaluation_protocol": (
            {
                "path": str(evaluation_protocol.path),
                "protocol_id": evaluation_protocol.protocol_id,
                "sha256": evaluation_protocol.sha256,
                "ordered_assignment_sha256": (
                    evaluation_protocol.ordered_assignment_sha256
                ),
                "row_count": evaluation_protocol.row_count,
                "fold_values": list(evaluation_protocol.fold_values),
                "fold_counts": evaluation_protocol.fold_counts,
                "speakers": evaluation_protocol.speakers,
                "source_groups": evaluation_protocol.source_groups,
            }
            if evaluation_protocol is not None
            else None
        ),
        "model_count": len(ranked),
        "ranking_rule": (
            "Descending five-fold mean accuracy, then descending macro F1, "
            "then model_id."
        ),
        "common_protocol_validation": {
            "passed": True,
            "manifest_sha256": common_protocol.manifest_sha256,
            "evaluation_protocol_id": (
                common_protocol.evaluation_protocol_id
            ),
            "evaluation_protocol_sha256": (
                common_protocol.evaluation_protocol_sha256
            ),
            "ordered_assignment_sha256": (
                common_protocol.ordered_assignment_sha256
            ),
            "fold_values": list(common_protocol.fold_values),
            "number_of_folds": len(common_protocol.fold_values),
            "samples": common_protocol.samples,
            "speakers": common_protocol.speakers,
            "validated_fields": [
                "manifest_sha256",
                "evaluation_protocol_id",
                "evaluation_protocol_sha256",
                "ordered_assignment_sha256",
                "fold_values",
                "sample_count",
                "speaker_count",
            ],
        },
        "statistical_reporting": {
            "performance_central_value": "arithmetic mean over folds",
            "performance_error_bar": "sample standard deviation over folds",
            "confidence_interval": (
                "raw two-sided 95% Student-t intervals copied from each "
                "aggregate_metrics.json"
            ),
            "chart_physical_bounds": (
                "performance error bars clipped to [0, 1] for display only; "
                "raw statistics are retained in this JSON"
            ),
            "timing_source": (
                "warm_success_timing_ms in each full embedding metadata.json"
            ),
        },
        "models": json_models,
        "artifacts": {
            "csv": str(
                (output_directory / "model_comparison.csv").resolve()
            ),
            "accuracy_png": str(
                (output_directory / "accuracy_bar.png").resolve()
            ),
            "accuracy_pdf": str(
                (output_directory / "accuracy_bar.pdf").resolve()
            ),
            "macro_f1_png": str(
                (output_directory / "macro_f1_bar.png").resolve()
            ),
            "macro_f1_pdf": str(
                (output_directory / "macro_f1_bar.pdf").resolve()
            ),
            "warm_timing_png": str(
                (
                    output_directory
                    / "warm_extraction_timing_bar.png"
                ).resolve()
            ),
            "warm_timing_pdf": str(
                (
                    output_directory
                    / "warm_extraction_timing_bar.pdf"
                ).resolve()
            ),
        },
    }
    atomic_json(
        output_directory / "model_comparison.json",
        summary_json,
    )

    plt, PercentFormatter = import_plotting()
    configure_plot_style(plt)
    save_performance_chart(
        plt=plt,
        PercentFormatter=PercentFormatter,
        summaries=ranked,
        metric_key="accuracy",
        title=(
            f"{len(common_protocol.fold_values)}-Fold "
            "Speaker Identification Accuracy"
        ),
        ylabel="Accuracy",
        output_stem=output_directory / "accuracy_bar",
        fold_count=len(common_protocol.fold_values),
    )
    save_performance_chart(
        plt=plt,
        PercentFormatter=PercentFormatter,
        summaries=ranked,
        metric_key="f1_macro",
        title=f"{len(common_protocol.fold_values)}-Fold Macro F1 Score",
        ylabel="Macro F1",
        output_stem=output_directory / "macro_f1_bar",
        fold_count=len(common_protocol.fold_values),
    )
    save_timing_chart(
        plt=plt,
        summaries=ranked,
        output_stem=output_directory / "warm_extraction_timing_bar",
    )

    print_summary(ranked, output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
