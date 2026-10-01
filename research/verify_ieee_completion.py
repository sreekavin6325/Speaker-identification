"""Verify all twelve IEEE checklist tasks against concrete result artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "research_results" / "ieee_complete"
EXTENDED_PROTOCOL = "librispeech_dev_clean_chapter_heldout_balanced_subset__90bb68c4fbc1"
EXTENDED_SUMMARY = (
    PROJECT_ROOT
    / "research_results"
    / "robustness_benchmark"
    / "protocols"
    / EXTENDED_PROTOCOL
    / "summary"
    / "logistic_regression"
    / "full"
    / "model_condition_summary.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def check_rows(path: Path, minimum: int) -> bool:
    return len(read_csv(require(path))) >= minimum


def verify_robustness(family: str, expected: set[str]) -> bool:
    rows = read_csv(require(EXTENDED_SUMMARY))
    present = {
        row["condition"]
        for row in rows
        if row.get("condition_family") == family
    }
    return expected.issubset(present)


def main() -> int:
    checks: list[dict[str, Any]] = []

    def add(
        number: int,
        title: str,
        evidence: list[Path],
        validator: Callable[[], bool],
        note: str,
    ) -> None:
        try:
            passed = bool(validator()) and all(require(path) for path in evidence)
            error = ""
        except Exception as exc:
            passed = False
            error = str(exc)
        checks.append(
            {
                "task_number": number,
                "task": title,
                "status": "COMPLETE" if passed else "INCOMPLETE",
                "evidence": "; ".join(str(path.resolve()) for path in evidence),
                "scientific_note": note,
                "verification_error": error,
            }
        )

    task12 = ROOT / "task_12_preprocessing_ablation"
    add(
        1,
        "Comparative analysis of four embedding models",
        [
            ROOT / "tasks_01_02_model_cv" / "model_comparison.csv",
            ROOT / "tasks_01_02_model_cv" / "accuracy_comparison.png",
            ROOT / "tasks_01_02_model_cv" / "precision_macro_comparison.png",
            ROOT / "tasks_01_02_model_cv" / "recall_macro_comparison.png",
            ROOT / "tasks_01_02_model_cv" / "f1_macro_comparison.png",
        ],
        lambda: check_rows(ROOT / "tasks_01_02_model_cv" / "model_comparison.csv", 4),
        "Same 2,703 LibriSpeech clips and persisted folds are used for every model; this utterance-level result is secondary to the strict protocol.",
    )
    add(
        2,
        "Five-fold cross-validation",
        [ROOT / "tasks_01_02_model_cv" / "five_fold_metrics.csv"],
        lambda: check_rows(ROOT / "tasks_01_02_model_cv" / "five_fold_metrics.csv", 20),
        "Five-fold estimates are utterance-level and optimistic; the chapter-held-out two-fold study is the primary leakage-controlled comparison.",
    )
    add(
        3,
        "Classifier comparison",
        [ROOT / "task_03_classifier_comparison" / "classifier_comparison_all_metrics.csv"],
        lambda: check_rows(
            ROOT / "task_03_classifier_comparison" / "classifier_comparison_all_metrics.csv", 32
        ),
        "Four embeddings by eight classifiers, exceeding the six requested classifier families.",
    )
    add(
        4,
        "Evaluation on an additional public dataset",
        [ROOT / "task_04_multi_dataset" / "dataset_comparison.csv"],
        lambda: check_rows(ROOT / "task_04_multi_dataset" / "dataset_comparison.csv", 4),
        "LibriSpeech (40 speakers) and the Kaggle-derived corpus (5 speakers) are reported descriptively with chance-normalized values; raw accuracies are not treated as a controlled ranking.",
    )
    noise_expected = {
        "noise_traffic_10db",
        "noise_office_10db",
        "noise_cafe_10db",
        "noise_rain_10db",
        "noise_fan_10db",
        "noise_street_10db",
    }
    add(
        5,
        "Noise robustness",
        [EXTENDED_SUMMARY, EXTENDED_SUMMARY.parent / "environmental_noise_accuracy.png"],
        lambda: verify_robustness("additive_environmental_noise", noise_expected)
        and verify_robustness("additive_white_noise", {"noise_white_10db"}),
        "Named environmental conditions are deterministic controlled proxies mixed at 10 dB SNR, not recordings from a real-noise corpus.",
    )
    duration_expected = {
        "short_0p5s",
        "short_1s",
        "short_2s",
        "short_3s",
        "short_5s",
        "short_10s",
    }
    add(
        6,
        "Short-duration evaluation",
        [EXTENDED_SUMMARY, EXTENDED_SUMMARY.parent / "short_duration_accuracy.png"],
        lambda: verify_robustness("short_duration", duration_expected),
        "Identical balanced held-out clips are center-cropped or zero-padded to each requested duration.",
    )
    add(
        7,
        "Open-set unknown-speaker recognition",
        [ROOT / "task_07_open_set" / "open_set_summary.csv", ROOT / "task_07_open_set" / "open_set_roc.png"],
        lambda: check_rows(ROOT / "task_07_open_set" / "open_set_summary.csv", 4),
        "Calibration and final unknown-test identities are disjoint; the desktop app applies the detector calibrated for the currently selected embedding model.",
    )
    add(
        8,
        "PCA, t-SNE and UMAP embedding visualization",
        [ROOT / "task_08_embedding_visualization" / "embedding_visualization_metrics.csv"],
        lambda: check_rows(
            ROOT / "task_08_embedding_visualization" / "embedding_visualization_metrics.csv", 12
        ),
        "All projections use the same deterministic balanced 800-clip subset.",
    )
    add(
        9,
        "Hyperparameter tuning",
        [ROOT / "task_09_hyperparameter_tuning" / "model_level_comparison.csv"],
        lambda: check_rows(ROOT / "task_09_hyperparameter_tuning" / "model_level_comparison.csv", 4),
        "Nested SVM tuning uses only outer-training data for selection.",
    )
    add(
        10,
        "Computational performance analysis",
        [ROOT / "task_10_computation" / "computational_performance.csv"],
        lambda: check_rows(ROOT / "task_10_computation" / "computational_performance.csv", 4),
        "Fresh sequential subprocess profiling reports cold/warm CPU latency, CPU time, peak RSS, model weights and GPU availability.",
    )
    add(
        11,
        "Statistical significance analysis",
        [ROOT / "task_11_statistics" / "pairwise_significance_tests.csv"],
        lambda: check_rows(ROOT / "task_11_statistics" / "pairwise_significance_tests.csv", 6),
        "Paired t-tests and Wilcoxon signed-rank tests use identical folds with Holm correction; n=5 remains low-power.",
    )
    sample_expected = {"sample_rate_8k", "sample_rate_16k", "sample_rate_22k05"}
    add(
        12,
        "Preprocessing and sample-rate ablation",
        [
            task12 / "model_variant_summary.csv",
            task12 / "factorial_effects_summary.csv",
            EXTENDED_SUMMARY.parent / "sample_rate_accuracy.png",
        ],
        lambda: check_rows(task12 / "model_variant_summary.csv", 32)
        and verify_robustness("sample_rate_ablation", sample_expected),
        "The 2^3 trim/denoise/normalize factorial is combined with 8, 16 and 22.05 kHz resampling conditions.",
    )

    ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = ROOT / "IEEE_12_TASK_COMPLETION_MATRIX.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    complete = sum(item["status"] == "COMPLETE" for item in checks)
    markdown = [
        "# IEEE Research Checklist Completion Report",
        "",
        f"**Verified status: {complete}/12 tasks complete.**",
        "",
        "| No. | Task | Status | Scientific note |",
        "|---:|---|:---:|---|",
    ]
    for item in checks:
        markdown.append(
            f"| {item['task_number']} | {item['task']} | **{item['status']}** | "
            f"{item['scientific_note']} |"
        )
    if complete != 12:
        markdown.extend(["", "## Outstanding verification errors", ""])
        for item in checks:
            if item["status"] != "COMPLETE":
                markdown.append(f"- Task {item['task_number']}: {item['verification_error']}")
    report_path = ROOT / "IEEE_12_TASK_COMPLETION_REPORT.md"
    report_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (ROOT / "completion_status.json").write_text(
        json.dumps({"complete": complete, "total": 12, "tasks": checks}, indent=2),
        encoding="utf-8",
    )
    print(f"Verified {complete}/12 tasks.")
    print(report_path)
    return 0 if complete == 12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
