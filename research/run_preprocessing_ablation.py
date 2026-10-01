"""Run a strict 2^3 preprocessing ablation on speaker embeddings.

Each preprocessing variant is applied consistently to the outer-training and
outer-test clips.  Within every strict chapter-held-out fold, StandardScaler
and LogisticRegression are fitted only on that variant's outer-training
embeddings.  The raw condition reuses the validated clean research embeddings.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from model_config import MODEL_CONFIGS, get_model_config
from research.extract_preprocessing_embeddings import (
    PREPROCESSING_PARAMETERS,
    PREPROCESSING_VARIANTS,
)
from research.extract_robustness_embeddings import load_protocol_and_manifest
from research.run_robustness_benchmark import (
    aggregate,
    align_to_protocol,
    atomic_json,
    load_array_artifacts,
    metrics,
    prepare_output,
    write_csv,
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
DEFAULT_VARIANT_ROOT = (
    PROJECT_ROOT / "research_results" / "preprocessing_embeddings"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "research_results" / "preprocessing_benchmark"
)

RAW_FACTORS = {"trim": False, "denoise": False, "normalize": False}
VARIANT_FACTORS: dict[str, dict[str, bool]] = {
    "raw": RAW_FACTORS,
    **PREPROCESSING_VARIANTS,
}
VARIANT_ORDER = list(VARIANT_FACTORS)
FACTOR_ORDER = ["trim", "denoise", "normalize"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate raw plus seven preprocessing combinations under the "
            "strict chapter-held-out protocol."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model", choices=list(MODEL_CONFIGS))
    selection.add_argument("--all-models", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_ORDER,
        default=VARIANT_ORDER,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-file", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--variant-root", type=Path, default=DEFAULT_VARIANT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clean-artifact-tag", default="full")
    parser.add_argument("--variant-artifact-tag", default="full")
    parser.add_argument("--output-tag", default="full")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if not math.isfinite(arguments.c) or arguments.c <= 0:
        raise ValueError("--c must be a positive finite number.")
    if arguments.max_iter < 1:
        raise ValueError("--max-iter must be positive.")
    if arguments.limit is not None and arguments.limit < 1:
        raise ValueError("--limit must be positive.")


def artifact_directory(
    model_id: str,
    variant: str,
    protocol_id: str,
    clean_root: Path,
    variant_root: Path,
    clean_tag: str,
    variant_tag: str,
) -> Path:
    if variant == "raw":
        return clean_root.resolve() / model_id / clean_tag
    return (
        variant_root.resolve()
        / protocol_id
        / model_id
        / variant
        / variant_tag
    )


def load_variants(
    model_id: str,
    variants: list[str],
    protocol_rows: list[dict[str, str]],
    protocol: dict[str, Any],
    clean_root: Path,
    variant_root: Path,
    clean_tag: str,
    variant_tag: str,
) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for variant in variants:
        path = artifact_directory(
            model_id,
            variant,
            str(protocol["protocol_id"]),
            clean_root,
            variant_root,
            clean_tag,
            variant_tag,
        )
        item = align_to_protocol(
            load_array_artifacts(path),
            protocol_rows,
            model_id,
            variant,
            enforce_artifact_folds=(variant != "raw"),
        )
        if variant != "raw":
            metadata = item["metadata"]
            if metadata.get("variant") != variant:
                raise ValueError(f"Variant metadata mismatch for {model_id}/{variant}.")
            if metadata.get("factors") != VARIANT_FACTORS[variant]:
                raise ValueError(f"Factor metadata mismatch for {model_id}/{variant}.")
            if metadata.get("protocol", {}).get("sha256") != protocol["sha256"]:
                raise ValueError(f"Protocol hash mismatch for {model_id}/{variant}.")
        loaded[variant] = item

    reference = loaded[variants[0]]
    for variant, item in loaded.items():
        if not np.array_equal(item["clip_ids"], reference["clip_ids"]):
            raise ValueError(f"Clip order mismatch for {model_id}/{variant}.")
        if not np.array_equal(item["y"], reference["y"]):
            raise ValueError(f"Label order mismatch for {model_id}/{variant}.")
        if not np.array_equal(item["folds"], reference["folds"]):
            raise ValueError(f"Fold order mismatch for {model_id}/{variant}.")
    return loaded


def factorial_contrasts(
    model_id: str,
    fold_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Estimate balanced coded-level contrasts for a complete 2^3 design."""
    present = {str(row["variant"]) for row in fold_rows}
    if present != set(VARIANT_ORDER):
        return []
    results: list[dict[str, Any]] = []
    contrast_definitions = [
        ("main", "trim", ("trim",)),
        ("main", "denoise", ("denoise",)),
        ("main", "normalize", ("normalize",)),
        ("two_way_interaction", "trim:denoise", ("trim", "denoise")),
        ("two_way_interaction", "trim:normalize", ("trim", "normalize")),
        (
            "two_way_interaction",
            "denoise:normalize",
            ("denoise", "normalize"),
        ),
        (
            "three_way_interaction",
            "trim:denoise:normalize",
            ("trim", "denoise", "normalize"),
        ),
    ]
    for fold in sorted({int(row["outer_fold"]) for row in fold_rows}):
        selected = [row for row in fold_rows if int(row["outer_fold"]) == fold]
        for effect_type, effect_name, factors in contrast_definitions:
            positive: list[dict[str, Any]] = []
            negative: list[dict[str, Any]] = []
            for row in selected:
                levels = [1 if VARIANT_FACTORS[str(row["variant"])][f] else -1 for f in factors]
                sign = int(np.prod(levels))
                (positive if sign > 0 else negative).append(row)
            if len(positive) != 4 or len(negative) != 4:
                raise RuntimeError(f"Unbalanced factorial contrast: {effect_name}.")
            result: dict[str, Any] = {
                "model_id": model_id,
                "outer_fold": fold,
                "effect_type": effect_type,
                "effect_name": effect_name,
                "positive_variants": "|".join(str(row["variant"]) for row in positive),
                "negative_variants": "|".join(str(row["variant"]) for row in negative),
            }
            for metric_name in ("accuracy", "balanced_accuracy", "f1_macro"):
                on_mean = statistics.fmean(float(row[metric_name]) for row in positive)
                off_mean = statistics.fmean(float(row[metric_name]) for row in negative)
                result[f"{metric_name}_positive_mean"] = on_mean
                result[f"{metric_name}_negative_mean"] = off_mean
                result[f"{metric_name}_effect"] = on_mean - off_mean
            results.append(result)
    return results


def aggregate_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted(
        {(str(row["model_id"]), str(row["effect_type"]), str(row["effect_name"])) for row in rows}
    )
    for model_id, effect_type, effect_name in keys:
        selected = [
            row
            for row in rows
            if row["model_id"] == model_id and row["effect_name"] == effect_name
        ]
        output.append(
            {
                "model_id": model_id,
                "model_display_name": get_model_config(model_id)["display_name"],
                "effect_type": effect_type,
                "effect_name": effect_name,
                "outer_folds": len(selected),
                "accuracy_effect_mean": aggregate(
                    [row["accuracy_effect"] for row in selected]
                )["mean"],
                "accuracy_effect_fold_sd": aggregate(
                    [row["accuracy_effect"] for row in selected]
                )["std"],
                "balanced_accuracy_effect_mean": aggregate(
                    [row["balanced_accuracy_effect"] for row in selected]
                )["mean"],
                "f1_macro_effect_mean": aggregate(
                    [row["f1_macro_effect"] for row in selected]
                )["mean"],
            }
        )
    return output


def run_model(
    model_id: str,
    variants: list[str],
    protocol_rows: list[dict[str, str]],
    protocol: dict[str, Any],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    output_directory = (
        arguments.output_root.resolve()
        / "protocols"
        / str(protocol["protocol_id"])
        / model_id
        / "logistic_regression"
        / arguments.output_tag
    )
    prepare_output(output_directory, arguments.overwrite)
    artifacts = load_variants(
        model_id,
        variants,
        protocol_rows,
        protocol,
        arguments.clean_root,
        arguments.variant_root,
        arguments.clean_artifact_tag,
        arguments.variant_artifact_tag,
    )
    reference = artifacts[variants[0]]
    y = reference["y"]
    clip_ids = reference["clip_ids"]
    folds = reference["folds"]
    source_groups = np.asarray(
        [row["source_group"] for row in protocol_rows], dtype=str
    )
    fold_values = sorted(np.unique(folds).tolist())
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    print()
    print("=" * 72)
    print("PREPROCESSING FACTORIAL BENCHMARK")
    print("=" * 72)
    print(f"Model      : {get_model_config(model_id)['display_name']}")
    print("Classifier : StandardScaler + LogisticRegression")
    print(f"Variants   : {len(variants)}")
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
            "variant_train_and_test_preprocessing_matched": True,
            "outer_test_used_for_fit": False,
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

        raw_accuracy: float | None = None
        for variant in variants:
            X = artifacts[variant]["X"]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_mask])
            classifier = LogisticRegression(
                C=arguments.c,
                solver="lbfgs",
                max_iter=arguments.max_iter,
                random_state=arguments.seed,
                n_jobs=1,
            )
            fit_started = time.perf_counter()
            classifier.fit(X_train, y[train_mask])
            fit_seconds = time.perf_counter() - fit_started
            prediction_started = time.perf_counter()
            prediction = classifier.predict(scaler.transform(X[test_mask]))
            prediction_seconds = time.perf_counter() - prediction_started
            current_metrics = metrics(y[test_mask], prediction)
            if variant == "raw":
                raw_accuracy = current_metrics["accuracy"]
            if "raw" in variants and raw_accuracy is None:
                raise RuntimeError("Raw variant must be evaluated first.")
            factors = VARIANT_FACTORS[variant]
            fold_rows.append(
                {
                    "model_id": model_id,
                    "outer_fold": int(fold),
                    "variant": variant,
                    "trim": factors["trim"],
                    "denoise": factors["denoise"],
                    "normalize": factors["normalize"],
                    "train_samples": int(train_mask.sum()),
                    "test_samples": int(test_mask.sum()),
                    **current_metrics,
                    "accuracy_delta_from_raw": (
                        current_metrics["accuracy"] - raw_accuracy
                        if raw_accuracy is not None
                        else ""
                    ),
                    "fit_seconds": fit_seconds,
                    "prediction_seconds": prediction_seconds,
                }
            )
            for clip_id, truth, predicted in zip(
                clip_ids[test_mask], y[test_mask], prediction, strict=True
            ):
                prediction_rows.append(
                    {
                        "model_id": model_id,
                        "outer_fold": int(fold),
                        "variant": variant,
                        "clip_id": str(clip_id),
                        "true_speaker": str(truth),
                        "predicted_speaker": str(predicted),
                        "correct": bool(truth == predicted),
                    }
                )

    aggregate_rows: list[dict[str, Any]] = []
    for variant in variants:
        selected = [row for row in fold_rows if row["variant"] == variant]
        factors = VARIANT_FACTORS[variant]
        aggregate_rows.append(
            {
                "model_id": model_id,
                "model_display_name": get_model_config(model_id)["display_name"],
                "variant": variant,
                "trim": factors["trim"],
                "denoise": factors["denoise"],
                "normalize": factors["normalize"],
                "outer_folds": len(selected),
                "test_samples_total": sum(int(row["test_samples"]) for row in selected),
                "accuracy_mean": aggregate([row["accuracy"] for row in selected])["mean"],
                "accuracy_fold_sd": aggregate([row["accuracy"] for row in selected])["std"],
                "balanced_accuracy_mean": aggregate(
                    [row["balanced_accuracy"] for row in selected]
                )["mean"],
                "f1_macro_mean": aggregate([row["f1_macro"] for row in selected])["mean"],
                "f1_macro_fold_sd": aggregate([row["f1_macro"] for row in selected])["std"],
                "accuracy_delta_from_raw_mean": aggregate(
                    [row["accuracy_delta_from_raw"] for row in selected]
                )["mean"],
                "fit_seconds_mean": aggregate([row["fit_seconds"] for row in selected])["mean"],
            }
        )

    effect_rows = factorial_contrasts(model_id, fold_rows)
    effect_aggregate_rows = aggregate_effects(effect_rows)
    write_csv(output_directory / "fold_metrics.csv", fold_rows)
    write_csv(output_directory / "aggregate_metrics.csv", aggregate_rows)
    write_csv(output_directory / "all_predictions.csv", prediction_rows)
    write_csv(output_directory / "leakage_audit.csv", leakage_rows)
    atomic_json(output_directory / "aggregate_metrics.json", aggregate_rows)
    if effect_rows:
        write_csv(output_directory / "factorial_effects_fold.csv", effect_rows)
        write_csv(
            output_directory / "factorial_effects_aggregate.csv",
            effect_aggregate_rows,
        )
        atomic_json(
            output_directory / "factorial_effects_aggregate.json",
            effect_aggregate_rows,
        )

    metadata = {
        "schema_version": 1,
        "experiment": "preprocessing_2x2x2_factorial_ablation",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": model_id,
        "model": {
            "display_name": get_model_config(model_id)["display_name"],
            "provider": get_model_config(model_id)["provider"],
            "architecture": get_model_config(model_id)["architecture"],
            "embedding_dimension": int(reference["X"].shape[1]),
        },
        "classifier": {
            "name": "logistic_regression",
            "C": arguments.c,
            "solver": "lbfgs",
            "max_iter": arguments.max_iter,
            "seed": arguments.seed,
            "standard_scaler_fit_partition": "variant outer-training only",
        },
        "protocol": protocol,
        "variants": variants,
        "variant_factors": {name: VARIANT_FACTORS[name] for name in variants},
        "preprocessing_parameters": PREPROCESSING_PARAMETERS,
        "method": {
            "training_and_test_preprocessing_matched": True,
            "outer_test_used_for_fit": False,
            "raw_reference_reuses_validated_clean_embeddings": True,
            "factor_effect_estimator": (
                "balanced coded-level contrast: mean(response | product code +1) "
                "minus mean(response | product code -1)"
            ),
        },
        "artifact_directories": {
            variant: artifacts[variant]["path"] for variant in variants
        },
        "alignment_checks": {
            "exact_clip_order_across_variants": True,
            "exact_labels_across_variants": True,
            "exact_outer_folds_across_variants": True,
            "outer_clip_overlap_count": 0,
            "outer_source_group_overlap_count": 0,
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


def create_figures(
    summary_directory: Path,
    model_ids: list[str],
    variants: list[str],
    aggregate_rows: list[dict[str, Any]],
    effect_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    architectures = [get_model_config(model_id)["architecture"] for model_id in model_ids]
    matrix = np.asarray(
        [
            [
                100.0
                * next(
                    float(row["accuracy_mean"])
                    for row in aggregate_rows
                    if row["model_id"] == model_id and row["variant"] == variant
                )
                for variant in variants
            ]
            for model_id in model_ids
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(13.2, 5.8), dpi=180)
    image = axis.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=matrix.min(), vmax=matrix.max())
    axis.set_xticks(range(len(variants)), [name.replace("_", "\n") for name in variants])
    axis.set_yticks(range(len(model_ids)), architectures)
    axis.set_xlabel("Preprocessing variant")
    axis.set_title("Preprocessing ablation: mean strict outer-fold accuracy", weight="bold")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            color = "white" if value > (matrix.min() + matrix.max()) / 2 else "black"
            axis.text(column_index, row_index, f"{value:.2f}%", ha="center", va="center", color=color, fontsize=8)
    figure.colorbar(image, ax=axis, label="Accuracy (%)")
    figure.tight_layout()
    figure.savefig(summary_directory / "preprocessing_accuracy_heatmap.png", bbox_inches="tight")
    figure.savefig(summary_directory / "preprocessing_accuracy_heatmap.pdf", bbox_inches="tight")
    plt.close(figure)

    main_effects = [row for row in effect_rows if row["effect_type"] == "main"]
    if main_effects:
        x = np.arange(len(FACTOR_ORDER), dtype=float)
        width = 0.8 / max(1, len(model_ids))
        figure, axis = plt.subplots(figsize=(11.5, 6.2), dpi=180)
        for index, model_id in enumerate(model_ids):
            values = [
                100.0
                * next(
                    float(row["accuracy_effect_mean"])
                    for row in main_effects
                    if row["model_id"] == model_id and row["effect_name"] == factor
                )
                for factor in FACTOR_ORDER
            ]
            axis.bar(
                x - 0.4 + width / 2 + index * width,
                values,
                width,
                label=get_model_config(model_id)["architecture"],
            )
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.set_xticks(x, ["Silence trim", "Noise reduction", "Peak normalization"])
        axis.set_ylabel("Factorial main effect on accuracy (percentage points)")
        axis.set_title("Average preprocessing main effects", weight="bold")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.legend(frameon=False, ncol=2)
        figure.text(
            0.5,
            0.01,
            "Positive values improve accuracy on average across the other two factors. "
            "Two correlated outer folds; descriptive comparison.",
            ha="center",
            fontsize=9,
            color="#555555",
        )
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        figure.savefig(summary_directory / "preprocessing_main_effects.png", bbox_inches="tight")
        figure.savefig(summary_directory / "preprocessing_main_effects.pdf", bbox_inches="tight")
        plt.close(figure)


def create_summary(
    model_ids: list[str],
    variants: list[str],
    protocol: dict[str, Any],
    arguments: argparse.Namespace,
) -> Path:
    summary_directory = (
        arguments.output_root.resolve()
        / "protocols"
        / str(protocol["protocol_id"])
        / "summary"
        / "logistic_regression"
        / arguments.output_tag
    )
    summary_directory.mkdir(parents=True, exist_ok=True)
    aggregate_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        run_directory = (
            arguments.output_root.resolve()
            / "protocols"
            / str(protocol["protocol_id"])
            / model_id
            / "logistic_regression"
            / arguments.output_tag
        )
        with (run_directory / "aggregate_metrics.json").open(
            "r", encoding="utf-8"
        ) as input_file:
            aggregate_rows.extend(json.load(input_file))
        effect_path = run_directory / "factorial_effects_aggregate.json"
        if effect_path.is_file():
            with effect_path.open("r", encoding="utf-8") as input_file:
                effect_rows.extend(json.load(input_file))

    write_csv(summary_directory / "model_variant_summary.csv", aggregate_rows)
    atomic_json(summary_directory / "model_variant_summary.json", aggregate_rows)
    if effect_rows:
        write_csv(summary_directory / "factorial_effects_summary.csv", effect_rows)
        atomic_json(summary_directory / "factorial_effects_summary.json", effect_rows)
    create_figures(summary_directory, model_ids, variants, aggregate_rows, effect_rows)
    atomic_json(
        summary_directory / "summary_metadata.json",
        {
            "schema_version": 1,
            "experiment": "preprocessing_2x2x2_factorial_ablation_summary",
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "protocol": protocol,
            "models": model_ids,
            "variants": variants,
            "variant_factors": {name: VARIANT_FACTORS[name] for name in variants},
            "classifier": "StandardScaler + LogisticRegression(C=1)",
            "significance_test_performed": False,
            "output_directory": str(summary_directory),
        },
    )
    return summary_directory


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)
    protocol_rows, protocol = load_protocol_and_manifest(
        arguments.protocol_file,
        arguments.manifest,
        limit=arguments.limit,
    )
    model_ids = list(MODEL_CONFIGS) if arguments.all_models else [arguments.model]
    requested = list(dict.fromkeys(arguments.variants))
    variants = [name for name in VARIANT_ORDER if name in requested]
    if "raw" in variants and variants[0] != "raw":
        variants.remove("raw")
        variants.insert(0, "raw")
    for model_id in model_ids:
        run_model(model_id, variants, protocol_rows, protocol, arguments)
    summary = create_summary(model_ids, variants, protocol, arguments)
    print()
    print("Preprocessing ablation summary:")
    print(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print("PREPROCESSING BENCHMARK FAILED")
        print("=" * 72)
        print(error)
        raise
