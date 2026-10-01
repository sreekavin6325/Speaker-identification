"""Evaluate clean-trained classifiers on noisy and short-duration speech.

For each strict outer fold, a StandardScaler and LogisticRegression classifier
are fitted only on clean training embeddings.  The same fitted pipeline is
then evaluated on the clean held-out clips and on condition-specific embeddings
for those exact held-out clips.  This isolates embedding robustness without
training-time augmentation or outer-test leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from model_config import MODEL_CONFIGS, get_model_config
from research.extract_robustness_embeddings import (
    CONDITIONS,
    load_protocol_and_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "manifests" / "librispeech_dev_clean_manifest.csv"
)
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "manifests"
    / "librispeech_dev_clean_chapter_heldout_2fold_protocol.csv"
)
DEFAULT_CLEAN_ROOT = (
    PROJECT_ROOT / "research_results" / "embeddings" / "librispeech_dev_clean"
)
DEFAULT_CONDITION_ROOT = PROJECT_ROOT / "research_results" / "robustness_embeddings"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "research_results" / "robustness_benchmark"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train on clean outer-fold embeddings and evaluate the same "
            "held-out clips under short-duration and noise conditions."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model", choices=list(MODEL_CONFIGS))
    selection.add_argument("--all-models", action="store_true")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITIONS),
        default=list(CONDITIONS),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-file", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--condition-root", type=Path, default=DEFAULT_CONDITION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="full")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}.")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output is not empty:\n{path}\nUse --overwrite or another tag."
        )
    path.mkdir(parents=True, exist_ok=True)


def load_array_artifacts(path: Path) -> dict[str, Any]:
    required = ["X.npy", "y.npy", "clip_ids.npy", "folds.npy", "metadata.json"]
    missing = [filename for filename in required if not (path / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts in {path}: " + ", ".join(missing)
        )
    X = np.load(path / "X.npy").astype(np.float32)
    y = np.load(path / "y.npy").astype(str)
    clip_ids = np.load(path / "clip_ids.npy").astype(str)
    folds = np.load(path / "folds.npy").astype(int)
    with (path / "metadata.json").open("r", encoding="utf-8") as input_file:
        metadata = json.load(input_file)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError(f"Invalid X.npy in {path}")
    if len({len(X), len(y), len(clip_ids), len(folds)}) != 1:
        raise ValueError(f"Artifact lengths differ in {path}")
    if len(set(clip_ids.tolist())) != len(clip_ids):
        raise ValueError(f"Duplicate clip IDs in {path}")
    return {
        "X": X,
        "y": y,
        "clip_ids": clip_ids,
        "folds": folds,
        "metadata": metadata,
        "hashes": {filename: file_sha256(path / filename) for filename in required},
        "path": str(path.resolve()),
    }


def align_to_protocol(
    artifacts: dict[str, Any],
    protocol_rows: list[dict[str, str]],
    model_id: str,
    label: str,
    enforce_artifact_folds: bool = True,
) -> dict[str, Any]:
    index = {clip_id: i for i, clip_id in enumerate(artifacts["clip_ids"])}
    missing = [row["clip_id"] for row in protocol_rows if row["clip_id"] not in index]
    if missing:
        raise ValueError(
            f"{model_id}/{label} is missing {len(missing)} protocol clips. "
            "No model-specific intersection is allowed. Examples: "
            + ", ".join(missing[:10])
        )
    selected = np.asarray([index[row["clip_id"]] for row in protocol_rows], dtype=int)
    X = artifacts["X"][selected]
    y = artifacts["y"][selected]
    clip_ids = artifacts["clip_ids"][selected]
    folds = artifacts["folds"][selected]
    for position, row in enumerate(protocol_rows):
        if clip_ids[position] != row["clip_id"]:
            raise RuntimeError("Clip alignment failed.")
        if y[position] != row["speaker_label"]:
            raise ValueError(f"Label mismatch for {row['clip_id']} in {label}.")
        if enforce_artifact_folds and int(folds[position]) != int(row["fold"]):
            raise ValueError(f"Fold mismatch for {row['clip_id']} in {label}.")
    # Clean research embeddings carry the original five-fold manifest values.
    # The external strict protocol intentionally replaces those assignments.
    protocol_folds = np.asarray(
        [int(row["fold"]) for row in protocol_rows], dtype=int
    )
    result = dict(artifacts)
    result.update(
        {"X": X, "y": y, "clip_ids": clip_ids, "folds": protocol_folds}
    )
    return result


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def aggregate(values: list[float]) -> dict[str, float | int]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "minimum": min(clean),
        "maximum": max(clean),
    }


def run_model(
    model_id: str,
    conditions: list[str],
    protocol_rows: list[dict[str, str]],
    protocol: dict[str, Any],
    clean_root: Path,
    condition_root: Path,
    output_root: Path,
    tag: str,
    c_value: float,
    max_iter: int,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    if c_value <= 0 or not math.isfinite(c_value):
        raise ValueError("--c must be a positive finite number.")
    if max_iter < 1:
        raise ValueError("--max-iter must be positive.")

    output_directory = (
        output_root.resolve()
        / "protocols"
        / str(protocol["protocol_id"])
        / model_id
        / "logistic_regression"
        / tag
    )
    prepare_output(output_directory, overwrite)

    clean = align_to_protocol(
        load_array_artifacts(clean_root.resolve() / model_id / tag),
        protocol_rows,
        model_id,
        "clean_full",
        enforce_artifact_folds=False,
    )
    transformed: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        artifact_path = (
            condition_root.resolve()
            / str(protocol["protocol_id"])
            / model_id
            / condition
            / tag
        )
        item = align_to_protocol(
            load_array_artifacts(artifact_path),
            protocol_rows,
            model_id,
            condition,
        )
        metadata = item["metadata"]
        if metadata.get("condition") != condition:
            raise ValueError(f"Condition metadata mismatch for {condition}.")
        condition_protocol = metadata.get("protocol", {})
        if condition_protocol.get("sha256") != protocol["sha256"]:
            raise ValueError(f"Protocol hash mismatch for {model_id}/{condition}.")
        if item["X"].shape[1] != clean["X"].shape[1]:
            raise ValueError(f"Embedding dimension mismatch for {condition}.")
        if not np.array_equal(item["clip_ids"], clean["clip_ids"]):
            raise ValueError(f"Clip order mismatch for {condition}.")
        if not np.array_equal(item["y"], clean["y"]):
            raise ValueError(f"Label order mismatch for {condition}.")
        if not np.array_equal(item["folds"], clean["folds"]):
            raise ValueError(f"Fold order mismatch for {condition}.")
        transformed[condition] = item

    X_clean = clean["X"]
    y = clean["y"]
    clip_ids = clean["clip_ids"]
    folds = clean["folds"]
    source_groups = np.asarray([row["source_group"] for row in protocol_rows], dtype=str)
    fold_values = sorted(np.unique(folds).tolist())
    condition_order = ["clean_full", *conditions]
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    print()
    print("=" * 72)
    print("ROBUSTNESS BENCHMARK")
    print("=" * 72)
    print(f"Model      : {get_model_config(model_id)['display_name']}")
    print("Classifier : StandardScaler + LogisticRegression")
    print(f"Conditions : {len(condition_order)}")
    print(f"Samples    : {len(y)}")
    print(f"Output     : {output_directory}")
    print("=" * 72)

    for fold in fold_values:
        test_mask = folds == fold
        train_mask = ~test_mask
        train_ids = set(clip_ids[train_mask].tolist())
        test_ids = set(clip_ids[test_mask].tolist())
        train_groups = set(source_groups[train_mask].tolist())
        test_groups = set(source_groups[test_mask].tolist())
        train_speakers = set(y[train_mask].tolist())
        test_speakers = set(y[test_mask].tolist())
        audit = {
            "model_id": model_id,
            "outer_fold": int(fold),
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "clip_overlap_count": len(train_ids & test_ids),
            "source_group_overlap_count": len(train_groups & test_groups),
            "test_only_speaker_count": len(test_speakers - train_speakers),
            "classifier_fit_on_clean_train_only": True,
            "transformed_outer_test_used_for_fit": False,
        }
        if any(
            audit[key] != 0
            for key in (
                "clip_overlap_count",
                "source_group_overlap_count",
                "test_only_speaker_count",
            )
        ):
            raise RuntimeError(f"Leakage audit failed for outer fold {fold}.")
        leakage_rows.append(audit)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_clean[train_mask])
        classifier = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=seed,
            n_jobs=1,
        )
        classifier.fit(X_train, y[train_mask])

        clean_accuracy: float | None = None
        for condition in condition_order:
            X_condition = X_clean if condition == "clean_full" else transformed[condition]["X"]
            prediction_started = time.perf_counter()
            prediction = classifier.predict(scaler.transform(X_condition[test_mask]))
            prediction_seconds = time.perf_counter() - prediction_started
            current_metrics = metrics(y[test_mask], prediction)
            if condition == "clean_full":
                clean_accuracy = current_metrics["accuracy"]
            if clean_accuracy is None:
                raise RuntimeError("Clean condition must be evaluated first.")
            definition = (
                {"family": "clean_reference"}
                if condition == "clean_full"
                else CONDITIONS[condition]
            )
            row = {
                "model_id": model_id,
                "outer_fold": int(fold),
                "condition": condition,
                "condition_family": definition["family"],
                "duration_seconds": definition.get("duration_seconds", ""),
                "snr_db": definition.get("snr_db", ""),
                "target_sample_rate": definition.get("target_sample_rate", ""),
                "train_samples": int(train_mask.sum()),
                "test_samples": int(test_mask.sum()),
                **current_metrics,
                "accuracy_drop_from_clean": clean_accuracy - current_metrics["accuracy"],
                "prediction_seconds": prediction_seconds,
                "prediction_ms_per_sample": (
                    1000.0 * prediction_seconds / int(test_mask.sum())
                ),
            }
            fold_rows.append(row)
            for clip_id, truth, predicted in zip(
                clip_ids[test_mask], y[test_mask], prediction, strict=True
            ):
                prediction_rows.append(
                    {
                        "model_id": model_id,
                        "outer_fold": int(fold),
                        "condition": condition,
                        "clip_id": str(clip_id),
                        "true_speaker": str(truth),
                        "predicted_speaker": str(predicted),
                        "correct": bool(truth == predicted),
                    }
                )

    tested_clean = [row for row in prediction_rows if row["condition"] == "clean_full"]
    if len(tested_clean) != len(y):
        raise RuntimeError("Every protocol clip must be tested exactly once clean.")
    if len({row["clip_id"] for row in tested_clean}) != len(y):
        raise RuntimeError("Clean outer-test clip coverage is not one-to-one.")

    aggregate_rows: list[dict[str, Any]] = []
    for condition in condition_order:
        selected = [row for row in fold_rows if row["condition"] == condition]
        definition = (
            {"family": "clean_reference"}
            if condition == "clean_full"
            else CONDITIONS[condition]
        )
        aggregate_rows.append(
            {
                "model_id": model_id,
                "model_display_name": get_model_config(model_id)["display_name"],
                "condition": condition,
                "condition_family": definition["family"],
                "duration_seconds": definition.get("duration_seconds", ""),
                "snr_db": definition.get("snr_db", ""),
                "target_sample_rate": definition.get("target_sample_rate", ""),
                "outer_folds": len(selected),
                "test_samples_total": sum(int(row["test_samples"]) for row in selected),
                "accuracy_mean": aggregate([row["accuracy"] for row in selected])["mean"],
                "accuracy_fold_sd": aggregate([row["accuracy"] for row in selected])["std"],
                "f1_macro_mean": aggregate([row["f1_macro"] for row in selected])["mean"],
                "f1_macro_fold_sd": aggregate([row["f1_macro"] for row in selected])["std"],
                "accuracy_drop_from_clean_mean": aggregate(
                    [row["accuracy_drop_from_clean"] for row in selected]
                )["mean"],
                "prediction_ms_per_sample_mean": aggregate(
                    [row["prediction_ms_per_sample"] for row in selected]
                )["mean"],
            }
        )

    write_csv(output_directory / "fold_metrics.csv", fold_rows)
    write_csv(output_directory / "aggregate_metrics.csv", aggregate_rows)
    write_csv(output_directory / "all_predictions.csv", prediction_rows)
    write_csv(output_directory / "leakage_audit.csv", leakage_rows)
    atomic_json(output_directory / "aggregate_metrics.json", aggregate_rows)
    metadata = {
        "schema_version": 1,
        "experiment": "clean_train_transformed_test_robustness",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": model_id,
        "model": {
            "display_name": get_model_config(model_id)["display_name"],
            "provider": get_model_config(model_id)["provider"],
            "architecture": get_model_config(model_id)["architecture"],
            "pretrained_name": (
                get_model_config(model_id).get("pretrained_name")
                or get_model_config(model_id).get("source")
            ),
            "embedding_dimension": int(X_clean.shape[1]),
        },
        "classifier": {
            "name": "logistic_regression",
            "C": c_value,
            "solver": "lbfgs",
            "max_iter": max_iter,
            "seed": seed,
            "standard_scaler_fit_partition": "clean_outer_training_only",
        },
        "protocol": protocol,
        "conditions": condition_order,
        "condition_definitions": {
            condition: CONDITIONS[condition] for condition in conditions
        },
        "method": {
            "training_domain": "clean full-duration embeddings only",
            "evaluation_domain": "same held-out clips transformed per condition",
            "outer_test_used_for_fit": False,
            "training_augmentation": False,
        },
        "alignment_checks": {
            "exact_clip_order_across_conditions": True,
            "exact_labels_across_conditions": True,
            "exact_outer_folds_across_conditions": True,
            "outer_clip_overlap_count": 0,
            "outer_source_group_overlap_count": 0,
        },
        "clean_artifacts": clean["path"],
        "condition_artifacts": {
            condition: transformed[condition]["path"] for condition in conditions
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "wall_seconds": time.perf_counter() - started,
        "statistical_caution": (
            "The two strict outer folds are correlated. Means and fold standard "
            "deviations are descriptive; no significance claim is made."
        ),
        "output_directory": str(output_directory),
    }
    atomic_json(output_directory / "run_metadata.json", metadata)
    print(f"Completed  : {model_id}")
    print("=" * 72)
    return metadata


def create_summary(
    model_ids: list[str],
    conditions: list[str],
    protocol: dict[str, Any],
    output_root: Path,
    tag: str,
) -> Path:
    summary_directory = (
        output_root.resolve()
        / "protocols"
        / str(protocol["protocol_id"])
        / "summary"
        / "logistic_regression"
        / tag
    )
    summary_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for order, model_id in enumerate(model_ids, start=1):
        run_directory = (
            output_root.resolve()
            / "protocols"
            / str(protocol["protocol_id"])
            / model_id
            / "logistic_regression"
            / tag
        )
        with (run_directory / "aggregate_metrics.json").open(
            "r", encoding="utf-8"
        ) as input_file:
            model_rows = json.load(input_file)
        for row in model_rows:
            rows.append({"model_order": order, **row})
    write_csv(summary_directory / "model_condition_summary.csv", rows)
    atomic_json(summary_directory / "model_condition_summary.json", rows)

    import matplotlib.pyplot as plt

    labels = {
        model_id: get_model_config(model_id)["architecture"] for model_id in model_ids
    }
    clean_by_model = {
        model_id: next(
            float(row["accuracy_mean"])
            for row in rows
            if row["model_id"] == model_id and row["condition"] == "clean_full"
        )
        for model_id in model_ids
    }

    def plot_family(
        family: str,
        x_key: str,
        xlabel: str,
        title: str,
        filename: str,
        reverse_x: bool = False,
    ) -> None:
        figure, axis = plt.subplots(figsize=(10.5, 6.2), dpi=180)
        for model_id in model_ids:
            selected = [
                row
                for row in rows
                if row["model_id"] == model_id
                and row["condition_family"] == family
            ]
            selected.sort(key=lambda row: float(row[x_key]))
            x = [float(row[x_key]) for row in selected]
            y = [100.0 * float(row["accuracy_mean"]) for row in selected]
            axis.plot(x, y, marker="o", linewidth=2.2, label=labels[model_id])
            axis.axhline(
                100.0 * clean_by_model[model_id],
                linewidth=0.8,
                alpha=0.12,
            )
        if reverse_x:
            axis.invert_xaxis()
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Mean outer-fold accuracy (%)")
        axis.set_title(title, weight="bold")
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(frameon=False, ncol=2)
        figure.text(
            0.5,
            0.01,
            "Clean-trained classifier; transformed held-out test clips only. "
            "Two correlated outer folds; descriptive comparison.",
            ha="center",
            fontsize=9,
            color="#555555",
        )
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        figure.savefig(summary_directory / f"{filename}.png", bbox_inches="tight")
        figure.savefig(summary_directory / f"{filename}.pdf", bbox_inches="tight")
        plt.close(figure)

    if any(CONDITIONS[name]["family"] == "short_duration" for name in conditions):
        plot_family(
            "short_duration",
            "duration_seconds",
            "Test segment duration (seconds)",
            "Speaker recognition under short-duration speech",
            "short_duration_accuracy",
        )
    if any(CONDITIONS[name]["family"] == "additive_white_noise" for name in conditions):
        plot_family(
            "additive_white_noise",
            "snr_db",
            "Signal-to-noise ratio (dB; harder to the right)",
            "Speaker recognition under additive white noise",
            "noise_accuracy",
            reverse_x=True,
        )

    if any(
        CONDITIONS[name]["family"] == "additive_environmental_noise"
        for name in conditions
    ):
        selected_conditions = [
            name
            for name in conditions
            if CONDITIONS[name]["family"] == "additive_environmental_noise"
        ]
        noise_labels = [str(CONDITIONS[name]["noise_type"]).title() for name in selected_conditions]
        figure, axis = plt.subplots(figsize=(11.0, 6.2), dpi=180)
        positions = np.arange(len(selected_conditions))
        width = 0.19
        for model_order, model_id in enumerate(model_ids):
            values = []
            for condition in selected_conditions:
                values.append(
                    100.0
                    * float(
                        next(
                            row["accuracy_mean"]
                            for row in rows
                            if row["model_id"] == model_id
                            and row["condition"] == condition
                        )
                    )
                )
            axis.bar(
                positions + (model_order - 1.5) * width,
                values,
                width=width,
                label=labels[model_id],
            )
        axis.set_xticks(positions, noise_labels)
        axis.set_ylabel("Mean outer-fold accuracy (%)")
        axis.set_title("Speaker Recognition Across Controlled Noise Types", weight="bold")
        axis.grid(True, axis="y", linestyle="--", alpha=0.35)
        axis.legend(frameon=False, ncol=2)
        figure.text(
            0.5,
            0.01,
            "Deterministic controlled noise proxies mixed at 10 dB SNR; "
            "clean-trained classifier and identical held-out clips.",
            ha="center",
            fontsize=9,
            color="#555555",
        )
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        figure.savefig(summary_directory / "environmental_noise_accuracy.png", bbox_inches="tight")
        figure.savefig(summary_directory / "environmental_noise_accuracy.pdf", bbox_inches="tight")
        plt.close(figure)

    if any(
        CONDITIONS[name]["family"] == "sample_rate_ablation"
        for name in conditions
    ):
        plot_family(
            "sample_rate_ablation",
            "target_sample_rate",
            "Intermediate audio sample rate (Hz)",
            "Sample-rate and resampling ablation",
            "sample_rate_accuracy",
        )

    atomic_json(
        summary_directory / "summary_metadata.json",
        {
            "schema_version": 1,
            "experiment": "robustness_summary",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "protocol": protocol,
            "models": model_ids,
            "conditions": ["clean_full", *conditions],
            "classifier": "logistic_regression",
            "training_domain": "clean full-duration",
            "test_domain": "strict held-out transformed clips",
            "significance_test_performed": False,
            "output_directory": str(summary_directory),
        },
    )
    return summary_directory


def main() -> None:
    arguments = parse_arguments()
    rows, protocol = load_protocol_and_manifest(
        arguments.protocol_file,
        arguments.manifest,
        limit=None,
    )
    model_ids = list(MODEL_CONFIGS) if arguments.all_models else [arguments.model]
    conditions = list(dict.fromkeys(arguments.conditions))
    for model_id in model_ids:
        run_model(
            model_id,
            conditions,
            rows,
            protocol,
            arguments.clean_root,
            arguments.condition_root,
            arguments.output_root,
            arguments.tag,
            arguments.c,
            arguments.max_iter,
            arguments.seed,
            arguments.overwrite,
        )
    summary = create_summary(
        model_ids,
        conditions,
        protocol,
        arguments.output_root,
        arguments.tag,
    )
    print()
    print("Robustness summary:")
    print(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print("ROBUSTNESS BENCHMARK FAILED")
        print("=" * 72)
        print(error)
        raise
