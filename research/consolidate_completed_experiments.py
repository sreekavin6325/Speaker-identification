"""Consolidate already-completed classifier, tuning and preprocessing studies."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "research_results" / "ieee_complete"
BASELINE_PROTOCOL_ROOT = next(
    (PROJECT_ROOT / "research_results" / "baseline_cv" / "protocols").iterdir()
)
TUNING_PROTOCOL_ROOT = next(
    (PROJECT_ROOT / "research_results" / "hyperparameter_tuning" / "protocols").iterdir()
)
PREPROCESS_PROTOCOL_ROOT = next(
    (PROJECT_ROOT / "research_results" / "preprocessing_benchmark" / "protocols").iterdir()
)
MODEL_ORDER = (
    "speechbrain_ecapa",
    "speechbrain_xvector",
    "wavlm_base_plus_sv",
    "unispeech_sat_base_plus_sv",
)
MODEL_NAMES = {
    "speechbrain_ecapa": "ECAPA-TDNN",
    "speechbrain_xvector": "X-Vector",
    "wavlm_base_plus_sv": "WavLM",
    "unispeech_sat_base_plus_sv": "UniSpeech-SAT",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plotting() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def consolidate_classifiers() -> None:
    source = (
        BASELINE_PROTOCOL_ROOT
        / "summary"
        / "classifier_benchmark"
        / "full"
        / "classifier_benchmark_long.csv"
    )
    rows = read_csv(source)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["classifier_id"])].append(row)

    summary: list[dict[str, Any]] = []
    for (model_id, classifier_id), selected in grouped.items():
        result: dict[str, Any] = {
            "model_id": model_id,
            "model": MODEL_NAMES[model_id],
            "classifier_id": classifier_id,
            "classifier": selected[0]["classifier_display_name"],
            "folds": len(selected),
        }
        for metric in (
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "classifier_fit_seconds",
            "prediction_ms_per_sample",
        ):
            values = np.asarray([float(row[metric]) for row in selected])
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_sd"] = float(np.std(values, ddof=1))
        summary.append(result)
    summary.sort(
        key=lambda row: (
            MODEL_ORDER.index(str(row["model_id"])),
            -float(row["accuracy_mean"]),
        )
    )
    output = OUTPUT_ROOT / "task_03_classifier_comparison"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "classifier_comparison_all_metrics.csv", summary)
    shutil.copy2(source, output / "classifier_fold_metrics.csv")

    plt = plotting()
    classifiers = sorted({str(row["classifier"]) for row in summary})
    for metric, ylabel, filename in (
        ("accuracy_mean", "Accuracy (%)", "classifier_accuracy_comparison"),
        ("precision_macro_mean", "Macro precision (%)", "classifier_precision_comparison"),
        ("recall_macro_mean", "Macro recall (%)", "classifier_recall_comparison"),
        ("f1_macro_mean", "Macro F1 (%)", "classifier_f1_comparison"),
    ):
        figure, axis = plt.subplots(figsize=(11.5, 6.0))
        positions = np.arange(len(classifiers))
        for model_id in MODEL_ORDER:
            lookup = {
                str(row["classifier"]): float(row[metric]) * 100.0
                for row in summary
                if row["model_id"] == model_id
            }
            axis.plot(
                positions,
                [lookup[item] for item in classifiers],
                marker="o",
                linewidth=2,
                label=MODEL_NAMES[model_id],
            )
        axis.set_xticks(positions, classifiers, rotation=25, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel.replace(" (%)", "") + " by Embedding Model and Classifier")
        axis.legend(ncol=2, frameon=False)
        axis.grid(axis="x", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure.savefig(output / f"{filename}.png", dpi=300)
        figure.savefig(output / f"{filename}.pdf")
        plt.close(figure)


def consolidate_primary_confusion_matrices() -> None:
    output = OUTPUT_ROOT / "tasks_01_02_model_cv" / "confusion_matrices"
    output.mkdir(parents=True, exist_ok=True)
    plt = plotting()
    for model_id in MODEL_ORDER:
        source = (
            PROJECT_ROOT
            / "research_results"
            / "baseline_cv"
            / model_id
            / "linear_svm"
            / "full"
            / "confusion_matrix.csv"
        )
        rows = read_csv(source)
        labels = [row["true_label"].replace("LibriSpeech_", "") for row in rows]
        columns = [column for column in rows[0] if column != "true_label"]
        matrix = np.asarray(
            [[int(row[column]) for column in columns] for row in rows], dtype=float
        )
        row_sums = np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
        normalized = matrix / row_sums
        shutil.copy2(source, output / f"{model_id}_confusion_matrix.csv")

        figure, axis = plt.subplots(figsize=(8.2, 7.2))
        image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        axis.set_title(f"{MODEL_NAMES[model_id]} — Row-Normalized Confusion Matrix")
        axis.set_xlabel("Predicted speaker")
        axis.set_ylabel("True speaker")
        axis.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=5)
        axis.set_yticks(np.arange(len(labels)), labels, fontsize=5)
        axis.grid(False)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Per-speaker recall")
        figure.tight_layout()
        figure.savefig(output / f"{model_id}_confusion_matrix.png", dpi=300)
        figure.savefig(output / f"{model_id}_confusion_matrix.pdf")
        plt.close(figure)


def consolidate_tuning() -> None:
    source_dir = TUNING_PROTOCOL_ROOT / "summary" / "svm" / "full"
    output = OUTPUT_ROOT / "task_09_hyperparameter_tuning"
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "model_level_comparison.csv",
        "fold_level_comparison.csv",
        "selected_parameter_frequency.csv",
        "accuracy_comparison.png",
        "accuracy_comparison.pdf",
        "delta_vs_default.png",
        "delta_vs_default.pdf",
        "macro_f1_comparison.png",
        "macro_f1_comparison.pdf",
        "summary_metadata.json",
    ):
        shutil.copy2(source_dir / name, output / name)


def consolidate_preprocessing() -> None:
    source_dir = PREPROCESS_PROTOCOL_ROOT / "summary" / "logistic_regression" / "full"
    output = OUTPUT_ROOT / "task_12_preprocessing_ablation"
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "model_variant_summary.csv",
        "model_variant_summary.json",
        "factorial_effects_summary.csv",
        "factorial_effects_summary.json",
        "summary_metadata.json",
    ):
        shutil.copy2(source_dir / name, output / name)


def main() -> int:
    consolidate_primary_confusion_matrices()
    consolidate_classifiers()
    consolidate_tuning()
    consolidate_preprocessing()
    print(f"Completed experiment summaries: {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
