"""Leakage-safe nested SVM tuning for strict speaker protocols.

The outer folds come from an explicit source-group-held-out protocol and are
never used during hyperparameter selection. Within each outer-training
partition, a deterministic shuffled StratifiedKFold over utterances is used by
GridSearchCV to select between linear and RBF SVM configurations using
macro-F1. The inner split is intentionally utterance-stratified rather than
chapter-grouped: after an outer chapter is held out, some speakers have only
one chapter left and therefore cannot participate in a chapter-grouped inner
cross-validation split. This limitation is recorded in every run.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from model_config import MODEL_CONFIGS, get_model_config
from research.run_baseline_cv import (
    apply_evaluation_protocol,
    compute_metrics,
    file_sha256,
    load_and_validate_artifacts,
    load_evaluation_protocol,
    save_confusion_matrix,
    summarize_values,
    validate_cross_model_alignment,
)


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
    / "hyperparameter_tuning"
)
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_GAMMA_GRID: tuple[str | float, ...] = (
    "scale",
    "auto",
    0.0001,
    0.001,
    0.01,
)
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
)
TIMING_NAMES = (
    "grid_search_seconds",
    "classifier_fit_seconds",
    "prediction_seconds",
    "prediction_ms_per_sample",
    "fold_total_seconds",
)
PREDICTION_FIELDS = (
    "outer_fold",
    "sample_index",
    "clip_id",
    "source_group",
    "true_label",
    "predicted_label",
    "correct",
    "decision_score",
    "score_type",
)
FOLD_METRIC_FIELDS = (
    "outer_fold",
    "train_samples",
    "test_samples",
    "inner_folds",
    "best_inner_macro_f1",
    "best_kernel",
    "best_C",
    "best_gamma",
    *METRIC_NAMES,
    *TIMING_NAMES,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run nested linear/RBF SVM hyperparameter tuning with an "
            "untouched source-group-held-out outer protocol."
        )
    )
    parser.add_argument(
        "--protocol-file",
        type=Path,
        required=True,
        help=(
            "Required CSV containing clip_id, speaker_label, source_group "
            "and fold."
        ),
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
        help="Tune every configured embedding model.",
    )
    parser.add_argument(
        "--tag",
        default="full",
        help="Embedding artifact tag (default: full).",
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
        "--seed",
        type=int,
        default=42,
        help="Deterministic inner-CV and SVC seed (default: 42).",
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=3,
        help="Utterance-stratified inner folds (default: 3).",
    )
    parser.add_argument(
        "--c-grid",
        nargs="+",
        default=[str(value) for value in DEFAULT_C_GRID],
        metavar="C",
        help=(
            "Positive C values. Linear tests C; RBF tests C x gamma. "
            "Default: 0.01 0.1 1 10 100."
        ),
    )
    parser.add_argument(
        "--gamma-grid",
        nargs="+",
        default=[str(value) for value in DEFAULT_GAMMA_GRID],
        metavar="GAMMA",
        help=(
            "RBF gamma values: scale, auto, or positive numbers. "
            "Default: scale auto 0.0001 0.001 0.01."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "GridSearchCV workers (default: 1 for reproducibility and "
            "bounded Windows memory use; -1 uses all processors)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing nested-tuning result directory.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    fieldnames: Sequence[str],
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
            fieldnames=list(fieldnames),
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_c_grid(values: Sequence[str]) -> list[float]:
    parsed: list[float] = []
    for raw_value in values:
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(
                f"Invalid --c-grid value: {raw_value!r}"
            ) from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                "--c-grid values must be positive and finite."
            )
        if value not in parsed:
            parsed.append(value)
    if not parsed:
        raise ValueError("--c-grid must contain at least one value.")
    return parsed


def parse_gamma_grid(
    values: Sequence[str],
) -> list[str | float]:
    parsed: list[str | float] = []
    for raw_value in values:
        normalized = str(raw_value).strip().lower()
        if normalized in {"scale", "auto"}:
            value: str | float = normalized
        else:
            try:
                value = float(normalized)
            except ValueError as error:
                raise ValueError(
                    f"Invalid --gamma-grid value: {raw_value!r}"
                ) from error
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "--gamma-grid numeric values must be positive and "
                    "finite."
                )
        if value not in parsed:
            parsed.append(value)
    if not parsed:
        raise ValueError(
            "--gamma-grid must contain at least one value."
        )
    return parsed


def prepare_output(
    output_directory: Path,
    overwrite: bool,
) -> None:
    if output_directory.exists():
        existing = list(output_directory.iterdir())
        if existing and not overwrite:
            raise FileExistsError(
                f"Result directory is not empty:\n{output_directory}\n"
                "Use --overwrite or choose another --output-root/tag."
            )
        if existing and overwrite:
            for child in existing:
                if child.is_dir():
                    if not child.name.startswith("fold_"):
                        raise RuntimeError(
                            "Refusing to remove unexpected directory from "
                            f"result path:\n{child}"
                        )
                    shutil.rmtree(child)
                elif child.is_file():
                    child.unlink()
                else:
                    raise RuntimeError(
                        f"Cannot safely replace unusual path:\n{child}"
                    )
    output_directory.mkdir(parents=True, exist_ok=True)


def json_value(value: object) -> object:
    """Convert NumPy/masked values into strict JSON-compatible values."""

    if np.ma.is_masked(value):
        return None
    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {
            str(key): json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value
    return str(value)


def cv_results_rows(
    cv_results: dict[str, object],
) -> tuple[list[str], list[dict[str, object]]]:
    fieldnames = list(cv_results)
    if not fieldnames:
        raise RuntimeError("GridSearchCV produced no cv_results_.")
    lengths = {
        len(np.asarray(cv_results[fieldname]))
        for fieldname in fieldnames
    }
    if len(lengths) != 1:
        raise RuntimeError(
            "GridSearchCV cv_results_ columns have inconsistent lengths."
        )
    row_count = lengths.pop()
    rows: list[dict[str, object]] = []
    for row_index in range(row_count):
        row: dict[str, object] = {}
        for fieldname in fieldnames:
            column = cv_results[fieldname]
            item = column[row_index]  # type: ignore[index]
            converted = json_value(item)
            if isinstance(converted, (dict, list)):
                row[fieldname] = json.dumps(
                    converted,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            elif converted is None:
                row[fieldname] = ""
            else:
                row[fieldname] = converted
        rows.append(row)
    return fieldnames, rows


def canonical_parameter_value(
    value: object | None,
) -> str | float | int | None:
    converted = json_value(value)
    if converted is None:
        return None
    if isinstance(converted, (str, float, int)):
        return converted
    return str(converted)


def prediction_strength(
    estimator: object,
    X: np.ndarray,
) -> tuple[np.ndarray, str, np.ndarray]:
    decision_function = getattr(estimator, "decision_function", None)
    if not callable(decision_function):
        return (
            np.zeros(len(X), dtype=np.float64),
            "unavailable",
            np.empty((len(X), 0), dtype=np.float64),
        )
    raw_scores = np.asarray(
        decision_function(X),
        dtype=np.float64,
    )
    if raw_scores.ndim == 1:
        scalar_scores = np.abs(raw_scores)
        score_matrix = raw_scores.reshape(-1, 1)
    elif raw_scores.ndim == 2:
        scalar_scores = np.max(raw_scores, axis=1)
        score_matrix = raw_scores
    else:
        raise RuntimeError(
            "Unexpected SVM decision_function shape: "
            f"{raw_scores.shape}"
        )
    if (
        not np.isfinite(scalar_scores).all()
        or not np.isfinite(score_matrix).all()
    ):
        raise RuntimeError(
            "SVM decision_function returned non-finite values."
        )
    return scalar_scores, "decision_function_max", score_matrix


def software_versions() -> dict[str, str | None]:
    packages = (
        "numpy",
        "scikit-learn",
        "scipy",
        "joblib",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def validate_inner_cv(
    labels: np.ndarray,
    source_groups: np.ndarray,
    inner_folds: int,
    outer_fold: int,
) -> dict[str, object]:
    if inner_folds < 2:
        raise ValueError("--inner-folds must be at least two.")
    counts = Counter(labels.tolist())
    insufficient = {
        str(label): int(count)
        for label, count in counts.items()
        if count < inner_folds
    }
    if insufficient:
        examples = ", ".join(
            f"{label}={count}"
            for label, count in list(insufficient.items())[:10]
        )
        raise ValueError(
            f"Outer fold {outer_fold} cannot support "
            f"{inner_folds}-fold stratified inner CV. "
            f"Speaker sample counts below {inner_folds}: {examples}"
        )
    groups_per_speaker: dict[str, int] = {}
    for label in sorted(counts):
        mask = labels == label
        groups_per_speaker[str(label)] = len(
            set(source_groups[mask].tolist())
        )
    one_group_speakers = sorted(
        label
        for label, count in groups_per_speaker.items()
        if count == 1
    )
    return {
        "minimum_speaker_utterances": int(min(counts.values())),
        "maximum_speaker_utterances": int(max(counts.values())),
        "speakers_with_one_outer_training_source_group": (
            one_group_speakers
        ),
        "speaker_source_group_counts": groups_per_speaker,
        "grouped_inner_cv_feasible_for_every_speaker": (
            len(one_group_speakers) == 0
        ),
    }


def build_parameter_grid(
    c_values: list[float],
    gamma_values: list[str | float],
) -> list[dict[str, list[object]]]:
    return [
        {
            "svc__kernel": ["linear"],
            "svc__C": list(c_values),
        },
        {
            "svc__kernel": ["rbf"],
            "svc__C": list(c_values),
            "svc__gamma": list(gamma_values),
        },
    ]


def run_model(
    *,
    model_id: str,
    tag: str,
    embeddings_root: Path,
    output_root: Path,
    protocol: dict[str, Any],
    seed: int,
    inner_folds: int,
    c_values: list[float],
    gamma_values: list[str | float],
    n_jobs: int,
    overwrite: bool,
) -> dict[str, object]:
    from sklearn import __version__ as sklearn_version
    from sklearn.metrics import confusion_matrix, f1_score, make_scorer
    from sklearn.model_selection import (
        GridSearchCV,
        StratifiedKFold,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    run_started = time.perf_counter()
    artifact_directory = (
        embeddings_root.resolve() / model_id / tag
    )
    artifacts = load_and_validate_artifacts(
        artifact_directory,
        model_id,
    )
    artifacts = apply_evaluation_protocol(
        artifacts,
        protocol,
        model_id,
    )
    X = np.asarray(artifacts["X"], dtype=np.float64)
    y = np.asarray(artifacts["y"]).astype(str)
    clip_ids = np.asarray(artifacts["clip_ids"]).astype(str)
    folds = np.asarray(artifacts["folds"], dtype=np.int64)
    source_groups = np.asarray(
        artifacts["source_groups"]
    ).astype(str)
    unique_folds = np.unique(folds)
    class_labels = np.unique(y)
    if not np.isfinite(X).all():
        raise ValueError(f"{model_id} embeddings contain non-finite values.")
    if len(unique_folds) < 2:
        raise ValueError("At least two outer folds are required.")

    output_directory = (
        output_root.resolve()
        / "protocols"
        / str(protocol["protocol_id"])
        / "svm"
        / model_id
        / tag
    )
    prepare_output(output_directory, overwrite)
    parameter_grid = build_parameter_grid(c_values, gamma_values)
    candidate_count = len(c_values) * (
        1 + len(gamma_values)
    )

    print()
    print("=" * 78)
    print("NESTED SVM HYPERPARAMETER TUNING")
    print("=" * 78)
    print(f"Model          : {get_model_config(model_id)['display_name']}")
    print(f"Outer protocol : {protocol['protocol_id']}")
    print(f"Outer folds    : {len(unique_folds)} (chapter-held-out)")
    print(
        f"Inner folds    : {inner_folds} "
        "(shuffled utterance-stratified)"
    )
    print(f"Candidates     : {candidate_count}")
    print(f"Samples        : {len(X)}")
    print(f"Speakers       : {len(class_labels)}")
    print(f"Output         : {output_directory}")
    print("=" * 78)

    fold_results: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    all_prediction_rows: list[dict[str, object]] = []
    combined_grid_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    tested_indices: list[int] = []
    aggregate_confusion = np.zeros(
        (len(class_labels), len(class_labels)),
        dtype=np.int64,
    )

    for fold_value in unique_folds:
        outer_fold = int(fold_value)
        fold_started = time.perf_counter()
        test_mask = folds == fold_value
        train_mask = ~test_mask
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)
        if len(train_indices) == 0 or len(test_indices) == 0:
            raise ValueError(
                f"Outer fold {outer_fold} has an empty partition."
            )

        train_clip_ids = set(clip_ids[train_mask].tolist())
        test_clip_ids = set(clip_ids[test_mask].tolist())
        clip_overlap = train_clip_ids & test_clip_ids
        train_groups = set(source_groups[train_mask].tolist())
        test_groups = set(source_groups[test_mask].tolist())
        source_group_overlap = train_groups & test_groups
        train_speakers = set(y[train_mask].tolist())
        test_speakers = set(y[test_mask].tolist())
        missing_in_train = test_speakers - train_speakers
        missing_in_test = train_speakers - test_speakers
        if clip_overlap:
            raise RuntimeError(
                f"Outer fold {outer_fold} has clip leakage."
            )
        if source_group_overlap:
            raise RuntimeError(
                f"Outer fold {outer_fold} has source-group leakage: "
                + ", ".join(sorted(source_group_overlap)[:10])
            )
        if missing_in_train or missing_in_test:
            raise ValueError(
                f"Outer fold {outer_fold} does not contain every speaker "
                "in both train and test. "
                f"test-only={sorted(missing_in_train)[:10]}, "
                f"train-only={sorted(missing_in_test)[:10]}"
            )

        inner_audit = validate_inner_cv(
            y[train_mask],
            source_groups[train_mask],
            inner_folds,
            outer_fold,
        )
        inner_splitter = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed,
        )
        pipeline = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(
                        copy=True,
                        with_mean=True,
                        with_std=True,
                    ),
                ),
                (
                    "svc",
                    SVC(
                        probability=False,
                        decision_function_shape="ovr",
                        random_state=seed,
                        cache_size=2048,
                    ),
                ),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring=make_scorer(
                f1_score,
                average="macro",
                pos_label=None,
                zero_division=0,
            ),
            n_jobs=n_jobs,
            refit=True,
            cv=inner_splitter,
            return_train_score=True,
            error_score="raise",
        )
        search_started = time.perf_counter()
        search.fit(X[train_mask], y[train_mask])
        grid_search_seconds = time.perf_counter() - search_started
        classifier_fit_seconds = float(
            getattr(search, "refit_time_", 0.0)
        )

        best_params = dict(search.best_params_)
        best_kernel = str(best_params["svc__kernel"])
        best_c = float(best_params["svc__C"])
        best_gamma = (
            canonical_parameter_value(best_params.get("svc__gamma"))
            if best_kernel == "rbf"
            else None
        )
        best_inner_macro_f1 = float(search.best_score_)
        best_index = int(search.best_index_)
        best_rank = int(
            np.asarray(search.cv_results_["rank_test_score"])[
                best_index
            ]
        )

        prediction_started = time.perf_counter()
        y_pred = np.asarray(
            search.best_estimator_.predict(X[test_mask])
        ).astype(str)
        prediction_seconds = time.perf_counter() - prediction_started
        estimator_classes = np.asarray(
            search.best_estimator_.classes_
        ).astype(str)
        if not np.array_equal(estimator_classes, class_labels):
            raise RuntimeError(
                "Best SVM class order differs from the persisted global "
                "class label order."
            )
        decision_scores, score_type, decision_matrix = (
            prediction_strength(
                search.best_estimator_,
                X[test_mask],
            )
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
            "grid_search_seconds": grid_search_seconds,
            "classifier_fit_seconds": classifier_fit_seconds,
            "prediction_seconds": prediction_seconds,
            "prediction_ms_per_sample": (
                1000.0 * prediction_seconds / len(test_indices)
            ),
            "fold_total_seconds": fold_total_seconds,
        }
        fold_directory = (
            output_directory / f"fold_{outer_fold:02d}"
        )
        fold_directory.mkdir(parents=True, exist_ok=False)
        fold_prediction_rows: list[dict[str, object]] = []
        for local_index, sample_index in enumerate(test_indices):
            row = {
                "outer_fold": outer_fold,
                "sample_index": int(sample_index),
                "clip_id": str(clip_ids[sample_index]),
                "source_group": str(source_groups[sample_index]),
                "true_label": str(y[sample_index]),
                "predicted_label": str(y_pred[local_index]),
                "correct": int(
                    str(y[sample_index]) == str(y_pred[local_index])
                ),
                "decision_score": float(
                    decision_scores[local_index]
                ),
                "score_type": score_type,
            }
            fold_prediction_rows.append(row)
            all_prediction_rows.append(row)
        write_csv(
            fold_directory / "predictions.csv",
            PREDICTION_FIELDS,
            fold_prediction_rows,
        )
        atomic_npy(
            fold_directory / "decision_scores.npy",
            decision_matrix,
        )
        save_confusion_matrix(
            fold_directory,
            matrix,
            class_labels,
        )

        cv_payload = {
            str(key): json_value(value)
            for key, value in search.cv_results_.items()
        }
        atomic_json(
            fold_directory / "cv_results.json",
            cv_payload,
        )
        cv_fieldnames, current_cv_rows = cv_results_rows(
            search.cv_results_
        )
        write_csv(
            fold_directory / "cv_results.csv",
            cv_fieldnames,
            current_cv_rows,
        )
        for candidate_index, result_row in enumerate(current_cv_rows):
            params = search.cv_results_["params"][candidate_index]
            candidate_params = dict(params)
            combined_grid_rows.append(
                {
                    "outer_fold": outer_fold,
                    "candidate_index": candidate_index,
                    "params_json": json.dumps(
                        json_value(candidate_params),
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                    "kernel": candidate_params.get(
                        "svc__kernel",
                        "",
                    ),
                    "C": candidate_params.get("svc__C", ""),
                    "gamma": candidate_params.get(
                        "svc__gamma",
                        "",
                    ),
                    "mean_test_score": result_row[
                        "mean_test_score"
                    ],
                    "std_test_score": result_row[
                        "std_test_score"
                    ],
                    "rank_test_score": result_row[
                        "rank_test_score"
                    ],
                    "mean_fit_time": result_row["mean_fit_time"],
                    "std_fit_time": result_row["std_fit_time"],
                    "mean_score_time": result_row[
                        "mean_score_time"
                    ],
                    "std_score_time": result_row[
                        "std_score_time"
                    ],
                }
            )

        best_payload = {
            "outer_fold": outer_fold,
            "selection_metric": "f1_macro",
            "selection_scorer": {
                "function": "sklearn.metrics.f1_score",
                "average": "macro",
                "pos_label": None,
                "zero_division": 0,
            },
            "best_index": best_index,
            "best_rank": best_rank,
            "best_inner_macro_f1": best_inner_macro_f1,
            "best_parameters": {
                "kernel": best_kernel,
                "C": best_c,
                "gamma": best_gamma,
            },
            "raw_pipeline_parameters": json_value(best_params),
        }
        atomic_json(
            fold_directory / "best_parameters.json",
            best_payload,
        )
        fold_result = {
            "outer_fold": outer_fold,
            "train_samples": int(len(train_indices)),
            "test_samples": int(len(test_indices)),
            "inner_folds": inner_folds,
            "best_inner_macro_f1": best_inner_macro_f1,
            "best_parameters": best_payload["best_parameters"],
            "metrics": metrics,
            "timing": timing,
            "leakage_audit": {
                "clip_overlap_count": 0,
                "source_group_overlap_count": 0,
                "test_only_speaker_count": 0,
                "train_only_speaker_count": 0,
                "outer_test_used_in_tuning": False,
                "inner_cv_partition": "outer_training_only",
                "inner_cv_type": (
                    "deterministic_shuffled_utterance_stratified"
                ),
                "inner_source_group_isolation_enforced": False,
                "inner_cv_limitation": (
                    "Some outer-training speakers have only one remaining "
                    "chapter; grouped inner CV cannot preserve every "
                    "speaker in every inner fold. The untouched outer "
                    "chapter-held-out test remains the final evaluation."
                ),
                "inner_cv_feasibility": inner_audit,
            },
        }
        atomic_json(
            fold_directory / "metrics.json",
            fold_result,
        )
        fold_results.append(fold_result)
        fold_row: dict[str, object] = {
            "outer_fold": outer_fold,
            "train_samples": len(train_indices),
            "test_samples": len(test_indices),
            "inner_folds": inner_folds,
            "best_inner_macro_f1": best_inner_macro_f1,
            "best_kernel": best_kernel,
            "best_C": best_c,
            "best_gamma": best_gamma if best_gamma is not None else "",
        }
        fold_row.update(metrics)
        fold_row.update(timing)
        fold_rows.append(fold_row)
        selected_rows.append(
            {
                "outer_fold": outer_fold,
                "best_kernel": best_kernel,
                "best_C": best_c,
                "best_gamma": (
                    best_gamma if best_gamma is not None else ""
                ),
                "best_inner_macro_f1": best_inner_macro_f1,
                "rank_test_score": best_rank,
            }
        )
        leakage_rows.append(
            {
                "outer_fold": outer_fold,
                "train_samples": len(train_indices),
                "test_samples": len(test_indices),
                "clip_overlap_count": 0,
                "source_group_overlap_count": 0,
                "test_only_speaker_count": 0,
                "train_only_speaker_count": 0,
                "inner_cv_partition": "outer_training_only",
                "outer_test_used_in_tuning": False,
                "inner_cv_type": (
                    "deterministic_shuffled_utterance_stratified"
                ),
                "inner_source_group_isolation_enforced": False,
                "inner_folds": inner_folds,
                "one_group_speaker_count": len(
                    inner_audit[
                        "speakers_with_one_outer_training_source_group"
                    ]
                ),
            }
        )
        print(
            f"Outer fold {outer_fold}: "
            f"kernel={best_kernel:<6} C={best_c:g} "
            f"gamma={best_gamma if best_gamma is not None else '-'}  "
            f"inner macro-F1={best_inner_macro_f1:.4f}  "
            f"test accuracy={metrics['accuracy'] * 100:.2f}%  "
            f"test macro-F1={metrics['f1_macro']:.4f}"
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
            "Outer protocol did not test every selected sample exactly "
            "once."
        )

    all_prediction_rows.sort(
        key=lambda row: int(row["sample_index"])
    )
    write_csv(
        output_directory / "all_predictions.csv",
        PREDICTION_FIELDS,
        all_prediction_rows,
    )
    save_confusion_matrix(
        output_directory,
        aggregate_confusion,
        class_labels,
    )
    atomic_npy(
        output_directory / "class_labels.npy",
        np.asarray(class_labels, dtype=str),
    )
    write_csv(
        output_directory / "fold_metrics.csv",
        FOLD_METRIC_FIELDS,
        fold_rows,
    )
    write_csv(
        output_directory / "selected_parameters.csv",
        (
            "outer_fold",
            "best_kernel",
            "best_C",
            "best_gamma",
            "best_inner_macro_f1",
            "rank_test_score",
        ),
        selected_rows,
    )
    write_csv(
        output_directory / "leakage_audit.csv",
        (
            "outer_fold",
            "train_samples",
            "test_samples",
            "clip_overlap_count",
            "source_group_overlap_count",
            "test_only_speaker_count",
            "train_only_speaker_count",
            "inner_cv_partition",
            "outer_test_used_in_tuning",
            "inner_cv_type",
            "inner_source_group_isolation_enforced",
            "inner_folds",
            "one_group_speaker_count",
        ),
        leakage_rows,
    )
    write_csv(
        output_directory / "grid_search_results.csv",
        (
            "outer_fold",
            "candidate_index",
            "params_json",
            "kernel",
            "C",
            "gamma",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
            "std_score_time",
        ),
        combined_grid_rows,
    )
    aggregate_values: dict[str, dict[str, float | int]] = {}
    aggregate_rows: list[dict[str, object]] = []
    for category, names in (
        ("metric", METRIC_NAMES),
        ("timing", TIMING_NAMES),
        ("selection", ("best_inner_macro_f1",)),
    ):
        for name in names:
            if category == "metric":
                values = [
                    float(result["metrics"][name])  # type: ignore[index]
                    for result in fold_results
                ]
            elif category == "timing":
                values = [
                    float(result["timing"][name])  # type: ignore[index]
                    for result in fold_results
                ]
            else:
                values = [
                    float(result[name])
                    for result in fold_results
                ]
            summary = summarize_values(values)
            aggregate_values[name] = summary
            aggregate_rows.append(
                {
                    "category": category,
                    "name": name,
                    **summary,
                }
            )
    write_csv(
        output_directory / "aggregate_metrics.csv",
        (
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
        ),
        aggregate_rows,
    )
    aggregate_payload = {
        "confidence_interval": (
            "two-sided 95% Student-t interval over outer-fold values"
        ),
        "confidence_interval_caution": (
            "Only two strict outer folds are available. Means and sample "
            "standard deviations are descriptive; the two-fold Student-t "
            "interval is not suitable as a primary uncertainty claim."
        ),
        "values": aggregate_values,
    }
    atomic_json(
        output_directory / "aggregate_metrics.json",
        aggregate_payload,
    )
    config = get_model_config(model_id)
    run_metadata = {
        "schema_version": 1,
        "experiment": "nested_svm_hyperparameter_tuning",
        "completed_at_utc": utc_now(),
        "model_id": model_id,
        "embedding_tag": tag,
        "model": {
            "display_name": config.get("display_name"),
            "provider": config.get("provider"),
            "architecture": config.get("architecture"),
            "pretrained_name": config.get("pretrained_name"),
            "source": config.get("source"),
            "embedding_dimension": int(X.shape[1]),
        },
        "protocol": {
            "mode": "external_protocol_file",
            "protocol_id": protocol["protocol_id"],
            "path": protocol["path"],
            "sha256": protocol["sha256"],
            "ordered_assignment_sha256": (
                protocol["ordered_assignment_sha256"]
            ),
            "row_count": protocol["row_count"],
            "samples": protocol["row_count"],
            "fold_values": [
                int(value) for value in unique_folds
            ],
            "fold_counts": protocol["fold_counts"],
            "number_of_folds": len(unique_folds),
            "speaker_count": protocol["speaker_count"],
            "speakers": protocol["speaker_count"],
            "source_group_count": protocol["source_group_count"],
            "source_groups": protocol["source_group_count"],
            "embedding_dimension": int(X.shape[1]),
            "outer_split": "source_group_chapter_held_out",
            "source_group_disjoint_enforced": True,
            "every_speaker_in_outer_train_and_test": True,
            "each_sample_tested_once": True,
            "exact_clip_leakage_detected": False,
        },
        "tuning": {
            "selection_metric": "f1_macro",
            "refit_best_estimator": True,
            "outer_test_used_in_tuning": False,
            "inner_cv": {
                "type": (
                    "deterministic_shuffled_utterance_stratified"
                ),
                "folds": inner_folds,
                "shuffle": True,
                "random_state": seed,
                "partition": "outer_training_only",
                "source_group_disjoint_enforced": False,
                "reason": (
                    "Some outer-training speakers have only one remaining "
                    "chapter, so chapter-grouped inner CV cannot retain "
                    "every speaker in every inner fold. Inner selection "
                    "never accesses the untouched outer test partition."
                ),
            },
            "pipeline": [
                {
                    "name": "StandardScaler",
                    "fit_partition": (
                        "each GridSearchCV inner-training partition"
                    ),
                    "with_mean": True,
                    "with_std": True,
                },
                {
                    "name": "SVC",
                    "probability": False,
                    "decision_function_shape": "ovr",
                },
            ],
            "parameter_grid": {
                "linear": {
                    "kernel": "linear",
                    "C": c_values,
                },
                "rbf": {
                    "kernel": "rbf",
                    "C": c_values,
                    "gamma": gamma_values,
                },
                "candidate_count_per_outer_fold": candidate_count,
            },
            "n_jobs": n_jobs,
            "seed": seed,
        },
        "embedding_artifacts": {
            "directory": str(artifact_directory.resolve()),
            "available_samples": artifacts[
                "available_artifact_samples"
            ],
            "selected_protocol_samples": len(X),
            "excluded_samples": artifacts[
                "excluded_artifact_samples"
            ],
            "manifest": artifacts["manifest"],
            "sha256": artifacts["artifact_hashes"],
            "source_metadata": artifacts["metadata"],
        },
        "manifest": artifacts["manifest"],
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": software_versions(),
            "scikit_learn": sklearn_version,
        },
        "code": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": file_sha256(Path(__file__).resolve()),
        },
        "results": {
            "outer_fold_count": len(fold_results),
            "aggregate_metrics_file": "aggregate_metrics.json",
            "fold_metrics_file": "fold_metrics.csv",
            "selected_parameters_file": "selected_parameters.csv",
            "grid_search_results_file": "grid_search_results.csv",
            "all_predictions_file": "all_predictions.csv",
            "leakage_audit_file": "leakage_audit.csv",
        },
        "wall_seconds": time.perf_counter() - run_started,
        "output_directory": str(output_directory.resolve()),
    }
    atomic_json(
        output_directory / "run_metadata.json",
        run_metadata,
    )

    accuracy = aggregate_values["accuracy"]
    macro_f1 = aggregate_values["f1_macro"]
    print("-" * 78)
    print(
        f"Outer accuracy : {float(accuracy['mean']) * 100:.2f}% "
        f"+/- {float(accuracy['std']) * 100:.2f}% SD"
    )
    print(
        f"Outer macro-F1 : {float(macro_f1['mean']):.4f} "
        f"+/- {float(macro_f1['std']):.4f} SD"
    )
    print(f"Artifacts      : {output_directory}")
    print("=" * 78)
    return {
        "model_id": model_id,
        "output_directory": str(output_directory),
        "accuracy_mean": float(accuracy["mean"]),
        "f1_macro_mean": float(macro_f1["mean"]),
    }


def main() -> int:
    arguments = parse_arguments()
    c_values = parse_c_grid(arguments.c_grid)
    gamma_values = parse_gamma_grid(arguments.gamma_grid)
    if arguments.inner_folds < 2:
        raise ValueError("--inner-folds must be at least two.")
    if arguments.n_jobs == 0:
        raise ValueError("--n-jobs cannot be zero.")
    protocol = load_evaluation_protocol(
        arguments.protocol_file
    )
    model_ids = (
        list(MODEL_CONFIGS)
        if arguments.all_models
        else [arguments.model]
    )
    embeddings_root = arguments.embeddings_root.resolve()
    if len(model_ids) > 1:
        validate_cross_model_alignment(
            embeddings_root,
            model_ids,
            arguments.tag,
            protocol,
        )

    failures: list[tuple[str, str]] = []
    for model_id in model_ids:
        try:
            run_model(
                model_id=str(model_id),
                tag=arguments.tag,
                embeddings_root=embeddings_root,
                output_root=arguments.output_root.resolve(),
                protocol=protocol,
                seed=arguments.seed,
                inner_folds=arguments.inner_folds,
                c_values=c_values,
                gamma_values=gamma_values,
                n_jobs=arguments.n_jobs,
                overwrite=arguments.overwrite,
            )
        except Exception as error:
            failures.append((str(model_id), str(error)))
            print(
                f"\nFAILED: {model_id}\n{error}",
                file=sys.stderr,
            )

    if failures:
        print(
            "\nNested SVM tuning finished with failures:",
            file=sys.stderr,
        )
        for model_id, message in failures:
            print(f"- {model_id}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
