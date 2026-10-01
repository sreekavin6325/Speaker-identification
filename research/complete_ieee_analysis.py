"""Complete the embedding-level experiments required by the IEEE checklist.

This script deliberately reuses the persisted embeddings and fold assignments so
that all model comparisons operate on identical samples.  Audio-domain
robustness and sample-rate experiments are handled by separate scripts because
they require fresh embedding extraction.

Outputs are written to ``research_results/ieee_complete`` and include:

* four-model 5-fold performance tables and separate metric figures;
* a second-dataset comparison using the common five-speaker clip intersection;
* paired fold-level significance tests with Holm correction;
* leakage-aware open-set evaluation with disjoint calibration/test identities;
* PCA, t-SNE and UMAP plots for every embedding model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "research_results" / "ieee_complete"
LIBRI_EMBED_ROOT = (
    PROJECT_ROOT / "research_results" / "embeddings" / "librispeech_dev_clean"
)
BASELINE_ROOT = PROJECT_ROOT / "research_results" / "baseline_cv"
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "librispeech_dev_clean_manifest.csv"
STRICT_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "librispeech_dev_clean_chapter_heldout_2fold_protocol.csv"
)

MODEL_IDS = (
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
MODEL_COLORS = {
    "speechbrain_ecapa": "#0072B2",
    "speechbrain_xvector": "#E69F00",
    "wavlm_base_plus_sv": "#009E73",
    "unispeech_sat_base_plus_sv": "#CC79A7",
}
METRICS = ("accuracy", "precision_macro", "recall_macro", "f1_macro")
RANDOM_SEED = 42


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip PCA/t-SNE/UMAP generation.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULT_ROOT,
        help="Destination directory.",
    )
    return parser.parse_args()


def ensure_plotting() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.28,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_primary_fold_rows(model_id: str) -> list[dict[str, str]]:
    path = BASELINE_ROOT / model_id / "linear_svm" / "full" / "fold_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing completed baseline: {path}")
    rows = read_csv(path)
    if len(rows) != 5:
        raise ValueError(f"Expected five folds for {model_id}; found {len(rows)}")
    return rows


def load_embedding_metadata(model_id: str) -> dict[str, Any]:
    path = LIBRI_EMBED_ROOT / model_id / "full" / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_primary_comparison(output_root: Path) -> list[dict[str, Any]]:
    """Task 1 and Task 2: common five-fold comparison and figures."""

    output = output_root / "tasks_01_02_model_cv"
    output.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for model_id in MODEL_IDS:
        rows = load_primary_fold_rows(model_id)
        metadata = load_embedding_metadata(model_id)
        for row in rows:
            fold_rows.append(
                {
                    "model_id": model_id,
                    "model": MODEL_NAMES[model_id],
                    "fold": int(row["fold"]),
                    **{metric: float(row[metric]) for metric in METRICS},
                }
            )

        values = {
            metric: [float(row[metric]) for row in rows] for metric in METRICS
        }
        summary_rows.append(
            {
                "model_id": model_id,
                "model": MODEL_NAMES[model_id],
                "folds": 5,
                "accuracy_mean": statistics.mean(values["accuracy"]),
                "accuracy_sd": statistics.stdev(values["accuracy"]),
                "precision_macro_mean": statistics.mean(values["precision_macro"]),
                "precision_macro_sd": statistics.stdev(values["precision_macro"]),
                "recall_macro_mean": statistics.mean(values["recall_macro"]),
                "recall_macro_sd": statistics.stdev(values["recall_macro"]),
                "f1_macro_mean": statistics.mean(values["f1_macro"]),
                "f1_macro_sd": statistics.stdev(values["f1_macro"]),
                "classifier_training_seconds_mean": statistics.mean(
                    float(row["classifier_fit_seconds"]) for row in rows
                ),
                "classifier_prediction_ms_per_sample_mean": statistics.mean(
                    float(row["prediction_ms_per_sample"]) for row in rows
                ),
                "embedding_extraction_ms_median": float(
                    metadata["warm_success_timing_ms"]["median"]
                ),
                "embedding_extraction_ms_mean": float(
                    metadata["warm_success_timing_ms"]["mean"]
                ),
            }
        )

    write_csv(output / "five_fold_metrics.csv", fold_rows)
    write_csv(output / "model_comparison.csv", summary_rows)

    plt = ensure_plotting()
    for metric, title in (
        ("accuracy", "Five-Fold Accuracy"),
        ("precision_macro", "Five-Fold Macro Precision"),
        ("recall_macro", "Five-Fold Macro Recall"),
        ("f1_macro", "Five-Fold Macro F1"),
    ):
        means = [float(row[f"{metric}_mean"]) * 100 for row in summary_rows]
        sds = [float(row[f"{metric}_sd"]) * 100 for row in summary_rows]
        fig, ax = plt.subplots(figsize=(7.4, 4.7))
        positions = np.arange(len(MODEL_IDS))
        bars = ax.bar(
            positions,
            means,
            yerr=sds,
            capsize=5,
            color=[MODEL_COLORS[item] for item in MODEL_IDS],
            edgecolor="#222222",
            linewidth=0.7,
        )
        ax.set_xticks(positions, [MODEL_NAMES[item] for item in MODEL_IDS])
        ax.set_ylabel("Score (%)")
        ax.set_title(title)
        lower = max(0.0, min(means) - max(3.0, max(sds) * 2.0))
        ax.set_ylim(lower, 100.6)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(100.2, value + 0.25),
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        fig.text(
            0.5,
            0.012,
            "Mean ± sample SD across the persisted five utterance-level folds.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(output / f"{metric}_comparison.png", dpi=300)
        fig.savefig(output / f"{metric}_comparison.pdf")
        plt.close(fig)

    return summary_rows


def normalized_relative_audio_path(value: str | Path) -> str:
    text = str(value).replace("/", "\\")
    marker = "processed_dataset\\"
    lower = text.lower()
    index = lower.find(marker)
    if index >= 0:
        text = text[index:]
    return text.lower()


def reconstruct_ecapa_paths(labels: np.ndarray) -> np.ndarray:
    paths: list[str] = []
    root = PROJECT_ROOT / "processed_dataset"
    skipped = {"_background_noise_", "other"}
    for speaker in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not speaker.is_dir() or speaker.name.lower() in skipped:
            continue
        for audio_path in sorted(speaker.glob("*.wav"), key=lambda item: item.name.lower()):
            paths.append(str(audio_path.relative_to(PROJECT_ROOT)))
    if len(paths) != len(labels):
        raise ValueError(
            "ECAPA path reconstruction count differs from its labels: "
            f"{len(paths)} versus {len(labels)}"
        )
    reconstructed_labels = np.asarray([Path(item).parent.name for item in paths])
    if not np.array_equal(reconstructed_labels.astype(str), labels.astype(str)):
        raise ValueError("ECAPA reconstructed path order does not match y.npy.")
    return np.asarray(paths)


def load_original_dataset_model(model_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directory = PROJECT_ROOT / "embeddings" / model_id
    x = np.load(directory / "X.npy", allow_pickle=False)
    y = np.load(directory / "y.npy", allow_pickle=True).astype(str)
    path_file = directory / "audio_paths.npy"
    paths = (
        np.load(path_file, allow_pickle=True).astype(str)
        if path_file.exists()
        else reconstruct_ecapa_paths(y)
    )
    if not (len(x) == len(y) == len(paths)):
        raise ValueError(f"Original dataset arrays are misaligned for {model_id}.")
    return x, y, paths


def evaluate_second_dataset(
    output_root: Path,
    librispeech_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Task 4: compare LibriSpeech with the original Kaggle-derived set."""

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    output = output_root / "task_04_multi_dataset"
    output.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        librispeech_speaker_count = len(
            {
                str(row["speaker_label"])
                for row in csv.DictReader(handle)
            }
        )

    def chance_normalized_accuracy(accuracy: float, speakers: int) -> float:
        chance = 1.0 / float(speakers)
        return (float(accuracy) - chance) / (1.0 - chance)

    loaded = {model_id: load_original_dataset_model(model_id) for model_id in MODEL_IDS}
    path_sets = [
        {normalized_relative_audio_path(item) for item in loaded[model_id][2]}
        for model_id in MODEL_IDS
    ]
    common_paths = set.intersection(*path_sets)
    if not common_paths:
        raise ValueError("No common clips exist across the four original-dataset embeddings.")

    common_sorted = sorted(common_paths)
    output_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for model_id in MODEL_IDS:
        x, y, paths = loaded[model_id]
        lookup = {
            normalized_relative_audio_path(path): index for index, path in enumerate(paths)
        }
        indices = np.asarray([lookup[path] for path in common_sorted], dtype=int)
        model_x = np.asarray(x[indices], dtype=np.float32)
        model_y = np.asarray(y[indices], dtype=str)

        fold_metrics: list[dict[str, float]] = []
        for fold, (train_index, test_index) in enumerate(
            splitter.split(model_x, model_y), start=1
        ):
            classifier = make_pipeline(
                StandardScaler(),
                LinearSVC(C=1.0, dual="auto", random_state=RANDOM_SEED),
            )
            classifier.fit(model_x[train_index], model_y[train_index])
            prediction = classifier.predict(model_x[test_index])
            row = {
                "accuracy": accuracy_score(model_y[test_index], prediction),
                "precision_macro": precision_score(
                    model_y[test_index], prediction, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    model_y[test_index], prediction, average="macro", zero_division=0
                ),
                "f1_macro": f1_score(
                    model_y[test_index], prediction, average="macro", zero_division=0
                ),
            }
            fold_metrics.append(row)
            fold_rows.append(
                {
                    "dataset": "Original Kaggle-derived five-speaker set",
                    "model_id": model_id,
                    "model": MODEL_NAMES[model_id],
                    "fold": fold,
                    **row,
                }
            )

        output_rows.append(
            {
                "dataset": "Original Kaggle-derived five-speaker set",
                "model_id": model_id,
                "model": MODEL_NAMES[model_id],
                "common_samples": len(model_y),
                "speakers": len(np.unique(model_y)),
                **{
                    f"{metric}_mean": statistics.mean(item[metric] for item in fold_metrics)
                    for metric in METRICS
                },
                **{
                    f"{metric}_sd": statistics.stdev(item[metric] for item in fold_metrics)
                    for metric in METRICS
                },
            }
        )

    libri_lookup = {row["model_id"]: row for row in librispeech_summary}
    comparison_rows: list[dict[str, Any]] = []
    for original in output_rows:
        libri = libri_lookup[original["model_id"]]
        comparison_rows.append(
            {
                "model_id": original["model_id"],
                "model": original["model"],
                "librispeech_dev_clean_accuracy": libri["accuracy_mean"],
                "librispeech_dev_clean_accuracy_sd": libri["accuracy_sd"],
                "librispeech_speakers": librispeech_speaker_count,
                "librispeech_chance_normalized_accuracy": (
                    chance_normalized_accuracy(
                        libri["accuracy_mean"],
                        librispeech_speaker_count,
                    )
                ),
                "original_dataset_accuracy": original["accuracy_mean"],
                "original_dataset_accuracy_sd": original["accuracy_sd"],
                "original_dataset_speakers": int(original["speakers"]),
                "original_chance_normalized_accuracy": (
                    chance_normalized_accuracy(
                        original["accuracy_mean"],
                        int(original["speakers"]),
                    )
                ),
                "difference_original_minus_librispeech": (
                    original["accuracy_mean"] - libri["accuracy_mean"]
                ),
                "comparison_scope": (
                    "descriptive_only_different_speaker_counts"
                ),
            }
        )

    write_csv(output / "original_dataset_fold_metrics.csv", fold_rows)
    write_csv(output / "original_dataset_summary.csv", output_rows)
    write_csv(output / "dataset_comparison.csv", comparison_rows)

    plt = ensure_plotting()
    labels = [MODEL_NAMES[item] for item in MODEL_IDS]
    libri_values = [libri_lookup[item]["accuracy_mean"] * 100 for item in MODEL_IDS]
    original_lookup = {item["model_id"]: item for item in output_rows}
    original_values = [original_lookup[item]["accuracy_mean"] * 100 for item in MODEL_IDS]
    positions = np.arange(len(MODEL_IDS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    ax.bar(positions - width / 2, libri_values, width, label="LibriSpeech dev-clean")
    ax.bar(
        positions + width / 2,
        original_values,
        width,
        label="Original five-speaker dataset",
    )
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Five-fold accuracy (%)")
    ax.set_title("Cross-Dataset Speaker Identification Accuracy")
    ax.set_ylim(max(0, min(libri_values + original_values) - 8), 100.6)
    ax.legend()
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "dataset_accuracy_comparison.png", dpi=300)
    fig.savefig(output / "dataset_accuracy_comparison.pdf")
    plt.close(fig)

    write_json(
        output / "protocol_notes.json",
        {
            "comparison_classifier": "StandardScaler + linear SVM",
            "folds": 5,
            "random_seed": RANDOM_SEED,
            "original_dataset_common_clip_count": len(common_sorted),
            "fairness_control": (
                "Only the exact clip intersection available for all four embedding "
                "models was evaluated."
            ),
            "limitation": (
                "The original Kaggle-derived recordings may contain adjacent "
                "fragments from shared source recordings. Source-session IDs were "
                "not available, so its utterance-level folds can be optimistic. "
                "LibriSpeech and the original corpus also have different speaker "
                "counts; raw cross-dataset accuracy differences are descriptive, "
                "not a controlled ranking. Chance-normalized values are reported."
            ),
        },
    )
    return comparison_rows


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def run_statistical_analysis(output_root: Path) -> list[dict[str, Any]]:
    """Task 11: paired tests over identical persisted folds."""

    from scipy.stats import ttest_rel, wilcoxon

    output = output_root / "task_11_statistics"
    output.mkdir(parents=True, exist_ok=True)
    scores = {
        model_id: np.asarray(
            [float(row["accuracy"]) for row in load_primary_fold_rows(model_id)]
        )
        for model_id in MODEL_IDS
    }
    rows: list[dict[str, Any]] = []
    raw_wilcoxon: list[float] = []
    raw_ttest: list[float] = []

    for first, second in combinations(MODEL_IDS, 2):
        difference = scores[first] - scores[second]
        t_result = ttest_rel(scores[first], scores[second])
        if np.allclose(difference, 0):
            w_statistic, w_pvalue = 0.0, 1.0
        else:
            w_result = wilcoxon(
                scores[first],
                scores[second],
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
            w_statistic, w_pvalue = float(w_result.statistic), float(w_result.pvalue)
        row = {
            "model_a": first,
            "model_a_name": MODEL_NAMES[first],
            "model_b": second,
            "model_b_name": MODEL_NAMES[second],
            "folds": 5,
            "mean_accuracy_a": float(np.mean(scores[first])),
            "mean_accuracy_b": float(np.mean(scores[second])),
            "mean_paired_difference": float(np.mean(difference)),
            "paired_t_statistic": float(t_result.statistic),
            "paired_t_p_raw": float(t_result.pvalue),
            "wilcoxon_statistic": w_statistic,
            "wilcoxon_p_raw": w_pvalue,
        }
        rows.append(row)
        raw_ttest.append(float(t_result.pvalue))
        raw_wilcoxon.append(w_pvalue)

    adjusted_t = holm_adjust(raw_ttest)
    adjusted_w = holm_adjust(raw_wilcoxon)
    for row, t_value, w_value in zip(rows, adjusted_t, adjusted_w):
        row["paired_t_p_holm"] = t_value
        row["paired_t_significant_0p05"] = t_value < 0.05
        row["wilcoxon_p_holm"] = w_value
        row["wilcoxon_significant_0p05"] = w_value < 0.05
        row["recommended_interpretation"] = (
            "significant"
            if w_value < 0.05
            else "not significant after Holm correction"
        )

    write_csv(output / "pairwise_significance_tests.csv", rows)
    write_json(
        output / "statistical_protocol.json",
        {
            "unit_of_pairing": "identical persisted five-fold accuracy scores",
            "primary_test": "two-sided Wilcoxon signed-rank",
            "secondary_test": "two-sided paired t-test",
            "multiple_comparison_correction": "Holm family-wise correction",
            "alpha": 0.05,
            "important_limitation": (
                "Only five paired folds are available; tests have low power and "
                "must be interpreted with effect sizes and confidence intervals."
            ),
        },
    )
    return rows


def load_libri_model_arrays(
    model_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directory = LIBRI_EMBED_ROOT / model_id / "full"
    x = np.load(directory / "X.npy", allow_pickle=False).astype(np.float32)
    y = np.load(directory / "y.npy", allow_pickle=True).astype(str)
    clip_ids = np.load(directory / "clip_ids.npy", allow_pickle=True).astype(str)
    folds = np.load(directory / "folds.npy", allow_pickle=False).astype(int)
    if not (len(x) == len(y) == len(clip_ids) == len(folds)):
        raise ValueError(f"LibriSpeech arrays are misaligned for {model_id}.")
    return x, y, clip_ids, folds


def choose_eer_threshold(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([known_scores, unknown_scores]))
    candidates = np.concatenate(([0.0], candidates, [1.0]))
    best = (math.inf, 0.5)
    for threshold in candidates:
        frr = float(np.mean(known_scores < threshold))
        far = float(np.mean(unknown_scores >= threshold))
        objective = abs(frr - far) + 1e-6 * (frr + far)
        if objective < best[0]:
            best = (objective, float(threshold))
    return best[1]


def evaluate_open_set(output_root: Path) -> list[dict[str, Any]]:
    """Task 7: identity-disjoint threshold calibration and open-set testing."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    output = output_root / "task_07_open_set"
    output.mkdir(parents=True, exist_ok=True)
    strict_rows = read_csv(STRICT_PROTOCOL_PATH)
    known_speakers = sorted({row["speaker_label"] for row in strict_rows})

    results: list[dict[str, Any]] = []
    curve_payload: dict[str, dict[str, list[float]]] = {}
    for model_id in MODEL_IDS:
        x, y, _, folds = load_libri_model_arrays(model_id)
        all_speakers = sorted(set(y))
        excluded = sorted(set(all_speakers) - set(known_speakers))
        if len(excluded) < 2:
            raise ValueError("Open-set protocol requires at least two excluded speakers.")
        rng = np.random.default_rng(RANDOM_SEED)
        shuffled = np.asarray(excluded, dtype=object)
        rng.shuffle(shuffled)
        split = max(1, len(shuffled) // 2)
        calibration_unknown = set(shuffled[:split].tolist())
        test_unknown = set(shuffled[split:].tolist())

        model_fold_rows: list[dict[str, Any]] = []
        all_known_scores: list[float] = []
        all_unknown_scores: list[float] = []
        all_known_correct: list[bool] = []

        for fold in range(1, 6):
            known_train_mask = np.isin(y, known_speakers) & (folds != fold)
            known_test_mask = np.isin(y, known_speakers) & (folds == fold)
            calibration_unknown_mask = np.isin(y, list(calibration_unknown))
            test_unknown_mask = np.isin(y, list(test_unknown))

            train_indices = np.flatnonzero(known_train_mask)
            class_counts = {
                label: int(np.sum(y[train_indices] == label)) for label in known_speakers
            }
            if min(class_counts.values()) < 2:
                raise ValueError("Too few known samples for threshold calibration.")
            fit_indices, calibration_known_indices = train_test_split(
                train_indices,
                test_size=0.20,
                random_state=RANDOM_SEED + fold,
                stratify=y[train_indices],
            )

            label_encoder = LabelEncoder().fit(y[fit_indices])
            scaler = StandardScaler().fit(x[fit_indices])
            classifier = LogisticRegression(
                max_iter=3000,
                C=1.0,
                solver="lbfgs",
                random_state=RANDOM_SEED,
            )
            classifier.fit(
                scaler.transform(x[fit_indices]),
                label_encoder.transform(y[fit_indices]),
            )

            calibration_known_scores = np.max(
                classifier.predict_proba(scaler.transform(x[calibration_known_indices])),
                axis=1,
            )
            calibration_unknown_scores = np.max(
                classifier.predict_proba(scaler.transform(x[calibration_unknown_mask])),
                axis=1,
            )
            threshold = choose_eer_threshold(
                calibration_known_scores,
                calibration_unknown_scores,
            )

            known_probabilities = classifier.predict_proba(
                scaler.transform(x[known_test_mask])
            )
            known_scores = np.max(known_probabilities, axis=1)
            known_prediction = label_encoder.inverse_transform(
                np.argmax(known_probabilities, axis=1)
            )
            known_correct = known_prediction == y[known_test_mask]

            unknown_scores = np.max(
                classifier.predict_proba(scaler.transform(x[test_unknown_mask])),
                axis=1,
            )
            known_accept = known_scores >= threshold
            unknown_reject = unknown_scores < threshold
            known_closed_accuracy = accuracy_score(y[known_test_mask], known_prediction)
            known_open_accuracy = float(np.mean(known_accept & known_correct))
            unknown_detection = float(np.mean(unknown_reject))
            far = float(np.mean(~unknown_reject))
            frr = float(np.mean(~known_accept))
            balanced_open_accuracy = (known_open_accuracy + unknown_detection) / 2.0

            row = {
                "model_id": model_id,
                "model": MODEL_NAMES[model_id],
                "fold": fold,
                "threshold": threshold,
                "known_test_samples": int(np.sum(known_test_mask)),
                "unknown_test_samples": int(np.sum(test_unknown_mask)),
                "known_closed_set_accuracy": known_closed_accuracy,
                "known_open_set_correct_accept_accuracy": known_open_accuracy,
                "unknown_detection_accuracy": unknown_detection,
                "false_acceptance_rate": far,
                "false_rejection_rate": frr,
                "balanced_open_set_accuracy": balanced_open_accuracy,
                "calibration_unknown_speakers": ";".join(sorted(calibration_unknown)),
                "test_unknown_speakers": ";".join(sorted(test_unknown)),
            }
            model_fold_rows.append(row)
            all_known_scores.extend(known_scores.tolist())
            all_unknown_scores.extend(unknown_scores.tolist())
            all_known_correct.extend(known_correct.tolist())

        write_csv(output / f"{model_id}_fold_metrics.csv", model_fold_rows)
        aggregate: dict[str, Any] = {
            "model_id": model_id,
            "model": MODEL_NAMES[model_id],
            "folds": 5,
            "known_speakers": len(known_speakers),
            "calibration_unknown_speakers": len(calibration_unknown),
            "test_unknown_speakers": len(test_unknown),
        }
        for key in (
            "threshold",
            "known_closed_set_accuracy",
            "known_open_set_correct_accept_accuracy",
            "unknown_detection_accuracy",
            "false_acceptance_rate",
            "false_rejection_rate",
            "balanced_open_set_accuracy",
        ):
            values = [float(row[key]) for row in model_fold_rows]
            aggregate[f"{key}_mean"] = statistics.mean(values)
            aggregate[f"{key}_sd"] = statistics.stdev(values)

        labels_binary = np.concatenate(
            [np.ones(len(all_known_scores)), np.zeros(len(all_unknown_scores))]
        )
        scores_binary = np.asarray(all_known_scores + all_unknown_scores)
        aggregate["known_vs_unknown_roc_auc"] = float(
            roc_auc_score(labels_binary, scores_binary)
        )
        false_positive, true_positive, thresholds = roc_curve(
            labels_binary, scores_binary
        )
        curve_payload[model_id] = {
            "false_positive_rate": false_positive.tolist(),
            "true_positive_rate": true_positive.tolist(),
            "thresholds": thresholds.tolist(),
        }
        results.append(aggregate)

    write_csv(output / "open_set_summary.csv", results)
    write_json(output / "roc_curves.json", curve_payload)
    write_json(
        output / "protocol.json",
        {
            "known_identity_source": str(STRICT_PROTOCOL_PATH),
            "known_speakers": len(known_speakers),
            "excluded_identity_partition": (
                "Nine identities excluded from the strict grouped protocol were "
                "shuffled with seed 42 and split into disjoint threshold-calibration "
                "and final unknown-test identities."
            ),
            "threshold_rule": (
                "Per fold, select the maximum-probability threshold minimizing "
                "absolute calibration FAR minus FRR."
            ),
            "important_limitations": [
                "The five test folds are utterance-level within known identities.",
                "The same disjoint unknown-test cohort is reused across folds.",
                "Maximum softmax probability is a baseline rejection score, not a "
                "guaranteed calibrated posterior for novel populations.",
            ],
        },
    )

    plt = ensure_plotting()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for model_id in MODEL_IDS:
        payload = curve_payload[model_id]
        auc = next(row["known_vs_unknown_roc_auc"] for row in results if row["model_id"] == model_id)
        ax.plot(
            payload["false_positive_rate"],
            payload["true_positive_rate"],
            color=MODEL_COLORS[model_id],
            linewidth=2,
            label=f"{MODEL_NAMES[model_id]} (AUC={auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
    ax.set_xlabel("False-positive rate (unknown accepted)")
    ax.set_ylabel("True-positive rate (known accepted)")
    ax.set_title("Known-vs-Unknown Rejection ROC")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "open_set_roc.png", dpi=300)
    fig.savefig(output / "open_set_roc.pdf")
    plt.close(fig)
    return results


def balanced_visualization_indices(y: np.ndarray, per_speaker: int = 20) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    selected: list[int] = []
    for speaker in sorted(set(y)):
        indices = np.flatnonzero(y == speaker)
        take = min(per_speaker, len(indices))
        selected.extend(rng.choice(indices, size=take, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def save_embedding_scatter(
    output: Path,
    coordinates: np.ndarray,
    labels: np.ndarray,
    title: str,
) -> None:
    plt = ensure_plotting()
    speakers = sorted(set(labels))
    cmap = plt.get_cmap("gist_ncar", len(speakers))
    fig, ax = plt.subplots(figsize=(8.2, 6.5))
    for index, speaker in enumerate(speakers):
        mask = labels == speaker
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=14,
            alpha=0.72,
            color=cmap(index),
            label=speaker.replace("LibriSpeech_", ""),
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        title="Speaker ID",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=6,
        title_fontsize=7,
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def generate_embedding_visualizations(output_root: Path) -> list[dict[str, Any]]:
    """Task 8: common balanced-sample PCA, t-SNE and UMAP plots."""

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP

    output = output_root / "task_08_embedding_visualization"
    output.mkdir(parents=True, exist_ok=True)
    reference_x, reference_y, reference_ids, _ = load_libri_model_arrays(MODEL_IDS[0])
    selected_reference = balanced_visualization_indices(reference_y, per_speaker=20)
    selected_ids = reference_ids[selected_reference]
    selected_labels = reference_y[selected_reference]
    results: list[dict[str, Any]] = []

    for model_id in MODEL_IDS:
        x, y, clip_ids, _ = load_libri_model_arrays(model_id)
        lookup = {clip_id: index for index, clip_id in enumerate(clip_ids)}
        indices = np.asarray([lookup[clip_id] for clip_id in selected_ids], dtype=int)
        if not np.array_equal(y[indices], selected_labels):
            raise ValueError(f"Visualization labels differ for {model_id}.")
        scaled = StandardScaler().fit_transform(x[indices])
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        pca_coordinates = pca.fit_transform(scaled)
        tsne_coordinates = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            max_iter=1200,
            random_state=RANDOM_SEED,
        ).fit_transform(scaled)
        umap_coordinates = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=RANDOM_SEED,
            n_jobs=1,
        ).fit_transform(scaled)

        model_output = output / model_id
        model_output.mkdir(parents=True, exist_ok=True)
        for method, coordinates in (
            ("PCA", pca_coordinates),
            ("t-SNE", tsne_coordinates),
            ("UMAP", umap_coordinates),
        ):
            save_embedding_scatter(
                model_output / method.lower().replace("-", ""),
                coordinates,
                selected_labels,
                f"{MODEL_NAMES[model_id]} — {method} Speaker Embeddings",
            )
            results.append(
                {
                    "model_id": model_id,
                    "model": MODEL_NAMES[model_id],
                    "method": method,
                    "samples": len(indices),
                    "speakers": len(np.unique(selected_labels)),
                    "silhouette_score_2d": float(
                        silhouette_score(coordinates, selected_labels, metric="euclidean")
                    ),
                    "pca_explained_variance_2d": (
                        float(np.sum(pca.explained_variance_ratio_))
                        if method == "PCA"
                        else ""
                    ),
                }
            )

    write_csv(output / "embedding_visualization_metrics.csv", results)
    write_json(
        output / "protocol.json",
        {
            "selection": "20 deterministic clips per each of 40 speakers",
            "samples": len(selected_ids),
            "random_seed": RANDOM_SEED,
            "common_clip_ids_across_models": True,
            "standardization": "StandardScaler fit separately per embedding model",
            "parameters": {
                "PCA": "2 components",
                "t-SNE": "perplexity=30, init=PCA, 1200 iterations",
                "UMAP": "n_neighbors=15, min_dist=0.1, metric=cosine",
            },
            "interpretation_note": (
                "The two-dimensional silhouette score describes the plotted "
                "projection and is not a substitute for classification accuracy."
            ),
        },
    )
    return results


def write_completion_status(
    output_root: Path,
    completed_now: dict[str, str],
) -> None:
    rows = [
        {"task": key, "status": "COMPLETE", "evidence": value}
        for key, value in completed_now.items()
    ]
    write_csv(output_root / "embedding_level_completion_status.csv", rows)


def main() -> int:
    arguments = parse_arguments()
    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("[1/5] Consolidating five-fold model comparison...")
    primary = summarize_primary_comparison(output_root)
    print("[2/5] Evaluating the common original-dataset clip intersection...")
    evaluate_second_dataset(output_root, primary)
    print("[3/5] Running paired statistical significance tests...")
    run_statistical_analysis(output_root)
    print("[4/5] Running leakage-aware open-set evaluation...")
    evaluate_open_set(output_root)
    if arguments.skip_visualizations:
        print("[5/5] Visualizations skipped by request.")
    else:
        print("[5/5] Generating PCA, t-SNE and UMAP visualizations...")
        generate_embedding_visualizations(output_root)

    completed = {
        "Task 1 — Four-model comparison": "tasks_01_02_model_cv/model_comparison.csv",
        "Task 2 — Five-fold cross-validation": "tasks_01_02_model_cv/five_fold_metrics.csv",
        "Task 4 — Additional public dataset": "task_04_multi_dataset/dataset_comparison.csv",
        "Task 7 — Open-set recognition": "task_07_open_set/open_set_summary.csv",
        "Task 8 — PCA/t-SNE/UMAP": "task_08_embedding_visualization/embedding_visualization_metrics.csv",
        "Task 11 — Statistical analysis": "task_11_statistics/pairwise_significance_tests.csv",
    }
    if arguments.skip_visualizations:
        completed.pop("Task 8 — PCA/t-SNE/UMAP")
    write_completion_status(output_root, completed)
    print(f"Completed embedding-level analyses: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
