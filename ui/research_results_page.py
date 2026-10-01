"""Read-only UI for the completed IEEE research experiments."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image

from ui.theme import (
    ACCENT,
    ACCENT_HOVER,
    BACKGROUND,
    BORDER,
    CARD,
    CARD_HOVER,
    ERROR,
    SUCCESS,
    TEXT,
    TEXT_SECONDARY,
    WARNING,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PROJECT_ROOT / "research_results"
IEEE_ROOT = RESEARCH_ROOT / "ieee_complete"
PAPER_ROOT = PROJECT_ROOT / "paper"


class ResearchResultsPage(ctk.CTkFrame):
    """Tabbed presentation of the canonical research CSV and figure files."""

    TAB_NAMES = (
        "Overview",
        "Models & CV",
        "Classifiers",
        "Robustness",
        "Open Set",
        "Efficiency",
        "Other Studies",
    )

    def __init__(self, parent):
        super().__init__(parent, fg_color=BACKGROUND, corner_radius=0)
        self._images = []
        self._tab_frames = {}
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._build_header()
        self._build_results()

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 8))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Research Results",
            font=("Segoe UI", 28, "bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text="IEEE multi-model speaker-recognition experiments",
            font=("Segoe UI", 13),
            text_color=TEXT_SECONDARY,
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.status_label = ctk.CTkLabel(
            header,
            text="Loading results...",
            font=("Segoe UI", 13, "bold"),
            text_color=WARNING,
        )
        self.status_label.grid(row=0, column=1, rowspan=2, padx=(12, 8))
        ctk.CTkButton(
            header,
            text="Refresh",
            width=92,
            command=self.refresh,
        ).grid(row=0, column=2, rowspan=2)

    def _build_results(self):
        self.tabview = ctk.CTkTabview(
            self,
            fg_color=BACKGROUND,
            segmented_button_fg_color=CARD,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color=ACCENT_HOVER,
            segmented_button_unselected_color=CARD,
            segmented_button_unselected_hover_color=CARD_HOVER,
        )
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
        self._tab_frames.clear()
        for name in self.TAB_NAMES:
            tab = self.tabview.add(name)
            tab.grid_rowconfigure(0, weight=1)
            tab.grid_columnconfigure(0, weight=1)
            scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
            scroll.grid(row=0, column=0, sticky="nsew")
            scroll.grid_columnconfigure(0, weight=1)
            self._tab_frames[name] = scroll

        try:
            self._populate_overview()
            self._populate_models()
            self._populate_classifiers()
            self._populate_robustness()
            self._populate_open_set()
            self._populate_efficiency()
            self._populate_other_studies()
            self.status_label.configure(
                text="12 / 12 tasks complete",
                text_color=SUCCESS,
            )
        except Exception as error:
            self.status_label.configure(text="Results unavailable", text_color=ERROR)
            self._error_card(
                self._tab_frames["Overview"],
                "Unable to load research results",
                str(error),
            )

    def _populate_overview(self):
        parent = self._tab_frames["Overview"]
        tasks = self._read_csv(IEEE_ROOT / "IEEE_12_TASK_COMPLETION_MATRIX.csv")
        summary = ctk.CTkFrame(parent, fg_color="transparent")
        summary.grid(row=0, column=0, sticky="ew", pady=(4, 12))
        for column in range(4):
            summary.grid_columnconfigure(column, weight=1)
        self._summary_card(summary, 0, "Research Tasks", f"{len(tasks)} / 12", SUCCESS)
        self._summary_card(summary, 1, "Embedding Models", "4", ACCENT)
        self._summary_card(summary, 2, "Classifiers", "8", "#8B5CF6")
        self._summary_card(
            summary, 3, "Paper Figures", str(len(list(IEEE_ROOT.rglob("*.png")))), "#38BDF8"
        )

        findings_card = self._section(parent, "Main research findings", row=1)
        ctk.CTkLabel(
            findings_card,
            text=(
                "Best leakage-controlled system: ECAPA-TDNN + Cosine Centroid (99.00%)\n"
                "Best classifier across embeddings: Logistic Regression (95.52% mean)\n"
                "Best robustness to noise and short speech: ECAPA-TDNN\n"
                "Best computational efficiency: X-Vector (25.7 ms warm median, 16.1 MB)\n"
                "Best accuracy-efficiency balance: X-Vector\n"
                "Utterance-level 5-fold scores are an optimistic sensitivity result"
            ),
            justify="left",
            anchor="w",
            font=("Segoe UI", 14),
            text_color=TEXT,
        ).pack(fill="x", padx=18, pady=(0, 14))
        buttons = ctk.CTkFrame(findings_card, fg_color="transparent")
        buttons.pack(fill="x", padx=14, pady=(0, 16))
        ctk.CTkButton(
            buttons,
            text="Open Results Folder",
            command=lambda: self._open_path(IEEE_ROOT),
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            buttons,
            text="Open IEEE Paper PDF",
            command=lambda: self._open_path(
                PAPER_ROOT / "Multi_Model_Speaker_Recognition_IEEE_Paper.pdf"
            ),
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            buttons,
            text="Open Completion Report",
            command=lambda: self._open_path(
                IEEE_ROOT / "IEEE_12_TASK_COMPLETION_REPORT.md"
            ),
        ).pack(side="left", padx=4)

        self._table(
            parent,
            "12-task implementation status",
            ["No.", "Experiment", "Status"],
            [[item["task_number"], item["task"], item["status"]] for item in tasks],
            row=2,
            widths=[55, 560, 120],
        )

    def _populate_models(self):
        parent = self._tab_frames["Models & CV"]
        five_fold_rows = self._read_csv(
            IEEE_ROOT / "tasks_01_02_model_cv" / "model_comparison.csv"
        )
        strict_rows = [
            item
            for item in self._read_csv(
                IEEE_ROOT
                / "task_03_classifier_comparison"
                / "classifier_comparison_all_metrics.csv"
            )
            if item["classifier_id"] == "linear_svm"
        ]
        strict_values = [
            [
                self._clean(item["model"]),
                self._percent(item["accuracy_mean"]),
                self._percent(item["accuracy_sd"]),
                self._percent(item["f1_macro_mean"]),
                item["folds"],
            ]
            for item in strict_rows
        ]
        self._table(
            parent,
            "Primary leakage-controlled chapter-held-out comparison",
            ["Model", "Accuracy", "SD", "Macro F1", "Folds"],
            strict_values,
            row=0,
            widths=[210, 120, 110, 120, 80],
        )

        five_fold_values = [
            [
                self._clean(item["model"]),
                self._percent(item["accuracy_mean"]),
                self._percent(item["precision_macro_mean"]),
                self._percent(item["recall_macro_mean"]),
                self._percent(item["f1_macro_mean"]),
                self._number(item["embedding_extraction_ms_median"], 2),
            ]
            for item in five_fold_rows
        ]
        self._table(
            parent,
            "Secondary utterance-level five-fold comparison (optimistic)",
            ["Model", "Accuracy", "Precision", "Recall", "Macro F1", "Embed ms"],
            five_fold_values,
            row=1,
            widths=[190, 105, 105, 105, 105, 105],
        )
        self._image_card(
            parent,
            "Accuracy comparison",
            IEEE_ROOT / "tasks_01_02_model_cv" / "accuracy_comparison.png",
            row=2,
        )
        self._note(
            parent,
            "Scientific note",
            "The chapter-held-out protocol is the primary scientific comparison. "
            "The five-fold split is utterance-level, has high chapter overlap, and is "
            "shown only as an optimistic sensitivity analysis.",
            row=3,
        )

    def _populate_classifiers(self):
        parent = self._tab_frames["Classifiers"]
        rows = self._read_csv(
            IEEE_ROOT
            / "task_03_classifier_comparison"
            / "classifier_comparison_all_metrics.csv"
        )
        grouped = defaultdict(list)
        for item in rows:
            grouped[self._clean(item["classifier"])].append(item)
        values = [
            [
                classifier,
                self._mean_percent(items, "accuracy_mean"),
                self._mean_percent(items, "precision_macro_mean"),
                self._mean_percent(items, "recall_macro_mean"),
                self._mean_percent(items, "f1_macro_mean"),
            ]
            for classifier, items in grouped.items()
        ]
        values.sort(key=lambda item: float(item[1].rstrip("%")), reverse=True)
        self._table(
            parent,
            "Average performance across four embeddings",
            ["Classifier", "Accuracy", "Precision", "Recall", "Macro F1"],
            values,
            row=0,
            widths=[220, 120, 120, 120, 120],
        )
        self._image_card(
            parent,
            "Classifier accuracy comparison",
            IEEE_ROOT
            / "task_03_classifier_comparison"
            / "classifier_accuracy_comparison.png",
            row=1,
        )

    def _populate_robustness(self):
        parent = self._tab_frames["Robustness"]
        rows = self._read_csv(self._find_robustness_summary())
        model_order = [
            "speechbrain_ecapa",
            "speechbrain_xvector",
            "wavlm_base_plus_sv",
            "unispeech_sat_base_plus_sv",
        ]
        names = ["ECAPA", "X-Vector", "WavLM", "UniSpeech"]
        lookup = {
            (item["condition"], item["model_id"]): self._percent(item["accuracy_mean"])
            for item in rows
        }

        def matrix(conditions):
            return [
                [label] + [lookup.get((condition, model), "--") for model in model_order]
                for condition, label in conditions
            ]

        columns = ["Condition"] + names
        widths = [170, 115, 115, 115, 115]
        self._table(
            parent,
            "Short-duration accuracy",
            columns,
            matrix([
                ("short_0p5s", "0.5 second"),
                ("short_1s", "1 second"),
                ("short_2s", "2 seconds"),
                ("short_3s", "3 seconds"),
                ("short_5s", "5 seconds"),
                ("short_10s", "10 seconds"),
            ]),
            row=0,
            widths=widths,
        )
        self._table(
            parent,
            "Environmental-noise accuracy at 10 dB SNR",
            columns,
            matrix([
                ("clean_full", "Clean"),
                ("noise_white_10db", "White"),
                ("noise_traffic_10db", "Traffic"),
                ("noise_office_10db", "Office"),
                ("noise_cafe_10db", "Cafe"),
                ("noise_rain_10db", "Rain"),
                ("noise_fan_10db", "Fan"),
                ("noise_street_10db", "Street"),
            ]),
            row=1,
            widths=widths,
        )
        self._table(
            parent,
            "Sample-rate accuracy",
            columns,
            matrix([
                ("sample_rate_8k", "8 kHz"),
                ("sample_rate_16k", "16 kHz"),
                ("sample_rate_22k05", "22.05 kHz"),
            ]),
            row=2,
            widths=widths,
        )
        self._note(
            parent,
            "Protocol note",
            "Environmental signals are deterministic controlled noise proxies. "
            "Short clips are center-cropped or zero-padded.",
            row=4,
        )

    def _populate_open_set(self):
        parent = self._tab_frames["Open Set"]
        rows = self._read_csv(IEEE_ROOT / "task_07_open_set" / "open_set_summary.csv")
        values = [
            [
                self._clean(item["model"]),
                self._percent(item["threshold_mean"]),
                self._percent(item["known_open_set_correct_accept_accuracy_mean"]),
                self._percent(item["unknown_detection_accuracy_mean"]),
                self._percent(item["false_acceptance_rate_mean"]),
                self._percent(item["false_rejection_rate_mean"]),
            ]
            for item in rows
        ]
        self._table(
            parent,
            "Unknown-speaker detection",
            ["Model", "Threshold", "Known Acc.", "Unknown Det.", "FAR", "FRR"],
            values,
            row=0,
            widths=[170, 110, 115, 120, 90, 90],
        )
        self._image_card(
            parent,
            "Known-versus-unknown ROC",
            IEEE_ROOT / "task_07_open_set" / "open_set_roc.png",
            row=1,
        )
        self._note(
            parent,
            "Deployment note",
            "Research-optimal thresholds differ by model. Calibrate the Settings threshold "
            "for the deployment data instead of assuming one universal value.",
            row=2,
        )

    def _populate_efficiency(self):
        parent = self._tab_frames["Efficiency"]
        rows = self._read_csv(
            IEEE_ROOT / "task_10_computation" / "computational_performance.csv"
        )
        values = [
            [
                self._clean(item["model"]),
                self._number(item["primary_weight_size_mb"], 1) + " MB",
                self._number(item["cold_wall_ms"], 1) + " ms",
                self._number(item["warm_wall_ms_median"], 1) + " ms",
                self._number(item["peak_rss_mb"], 1) + " MB",
                "Available" if item["gpu_available"].lower() == "true" else "CPU only",
            ]
            for item in rows
        ]
        self._table(
            parent,
            "Fresh-process resource profile",
            ["Model", "Weights", "Cold load", "Warm embed", "Peak RAM", "Device"],
            values,
            row=0,
            widths=[170, 110, 115, 115, 115, 105],
        )
        self._note(
            parent,
            "Interpretation",
            "X-Vector is the most efficient model on this Windows CPU host. "
            "GPU values are unavailable because no compatible GPU was present.",
            row=1,
        )

    def _populate_other_studies(self):
        parent = self._tab_frames["Other Studies"]
        datasets = self._read_csv(
            IEEE_ROOT / "task_04_multi_dataset" / "dataset_comparison.csv"
        )
        self._table(
            parent,
            "Cross-dataset comparison (descriptive; unequal speaker counts)",
            [
                "Model",
                "LibriSpeech raw (40)",
                "Original raw (5)",
                "LibriSpeech chance-norm.",
                "Original chance-norm.",
            ],
            [[
                self._clean(item["model"]),
                self._percent(item["librispeech_dev_clean_accuracy"]),
                self._percent(item["original_dataset_accuracy"]),
                self._percent(
                    item.get(
                        "librispeech_chance_normalized_accuracy",
                        item["librispeech_dev_clean_accuracy"],
                    )
                ),
                self._percent(
                    item.get(
                        "original_chance_normalized_accuracy",
                        item["original_dataset_accuracy"],
                    )
                ),
            ] for item in datasets],
            row=0,
            widths=[175, 125, 125, 145, 145],
        )
        self._note(
            parent,
            "Interpretation",
            "Raw accuracy differences are not a controlled ranking because "
            "LibriSpeech has 40 speakers while the original dataset has 5. "
            "Chance-normalized values are shown only as a descriptive aid.",
            row=1,
        )

        tuning = self._read_csv(
            IEEE_ROOT / "task_09_hyperparameter_tuning" / "model_level_comparison.csv"
        )
        self._table(
            parent,
            "Nested SVM hyperparameter tuning",
            ["Model", "Linear", "RBF", "Tuned", "Tuned vs linear"],
            [[
                self._clean(item["model_display_name"]),
                self._percent(item["linear_svm_accuracy_mean"]),
                self._percent(item["rbf_svm_accuracy_mean"]),
                self._percent(item["tuned_accuracy_mean"]),
                f"{float(item['tuned_minus_linear_accuracy']) * 100:+.2f} pp",
            ] for item in tuning],
            row=2,
            widths=[190, 110, 110, 110, 145],
        )

        visuals = self._read_csv(
            IEEE_ROOT
            / "task_08_embedding_visualization"
            / "embedding_visualization_metrics.csv"
        )
        self._table(
            parent,
            "Embedding visualization quality",
            ["Model", "Method", "2-D silhouette", "Samples", "Speakers"],
            [[
                self._clean(item["model"]),
                item["method"],
                self._number(item["silhouette_score_2d"], 4),
                item["samples"],
                item["speakers"],
            ] for item in visuals],
            row=3,
            widths=[170, 110, 145, 100, 100],
        )

        stats = self._read_csv(
            IEEE_ROOT / "task_11_statistics" / "pairwise_significance_tests.csv"
        )
        self._table(
            parent,
            "Pairwise significance tests (Holm corrected)",
            ["Comparison", "t-test p", "Wilcoxon p", "Wilcoxon significant"],
            [[
                f"{self._clean(item['model_a_name'])} vs {self._clean(item['model_b_name'])}",
                self._number(item["paired_t_p_holm"], 4),
                self._number(item["wilcoxon_p_holm"], 4),
                "Yes" if item["wilcoxon_significant_0p05"].lower() == "true" else "No",
            ] for item in stats],
            row=4,
            widths=[310, 120, 130, 180],
        )

        ablation = self._read_csv(
            IEEE_ROOT / "task_12_preprocessing_ablation" / "model_variant_summary.csv"
        )
        best = {}
        for item in ablation:
            model_id = item["model_id"]
            if model_id not in best or float(item["accuracy_mean"]) > float(
                best[model_id]["accuracy_mean"]
            ):
                best[model_id] = item
        self._table(
            parent,
            "Best preprocessing variant per model",
            ["Model", "Best variant", "Accuracy", "Gain vs raw"],
            [[
                self._clean(item["model_display_name"]),
                item["variant"].replace("_", " + ").title(),
                self._percent(item["accuracy_mean"]),
                f"{float(item['accuracy_delta_from_raw_mean']) * 100:+.2f} pp",
            ] for item in best.values()],
            row=5,
            widths=[190, 240, 120, 125],
        )

    def _summary_card(self, parent, column, title, value, color):
        card = ctk.CTkFrame(
            parent,
            fg_color=CARD,
            border_width=1,
            border_color=BORDER,
            corner_radius=14,
        )
        card.grid(row=0, column=column, sticky="nsew", padx=5)
        ctk.CTkLabel(
            card, text=title, font=("Segoe UI", 12), text_color=TEXT_SECONDARY
        ).pack(anchor="w", padx=16, pady=(14, 3))
        ctk.CTkLabel(
            card, text=value, font=("Segoe UI", 25, "bold"), text_color=color
        ).pack(anchor="w", padx=16, pady=(0, 14))

    def _section(self, parent, title, row):
        card = ctk.CTkFrame(
            parent,
            fg_color=CARD,
            border_width=1,
            border_color=BORDER,
            corner_radius=14,
        )
        card.grid(row=row, column=0, sticky="ew", pady=8)
        ctk.CTkLabel(
            card, text=title, font=("Segoe UI", 18, "bold"), text_color=TEXT
        ).pack(anchor="w", padx=18, pady=(16, 12))
        return card

    def _table(self, parent, title, columns, rows, row, widths):
        card = self._section(parent, title, row=row)
        table = ctk.CTkFrame(card, fg_color=BORDER, corner_radius=8)
        table.pack(fill="x", padx=14, pady=(0, 16))
        for column, width in enumerate(widths):
            table.grid_columnconfigure(column, weight=1, minsize=width)
        for column, heading in enumerate(columns):
            self._table_cell(table, heading, 0, column, True, widths[column])
        for row_index, values in enumerate(rows, start=1):
            for column, value in enumerate(values):
                self._table_cell(table, value, row_index, column, False, widths[column])
        return card

    def _table_cell(self, parent, text, row, column, header, width):
        background = CARD_HOVER if header else (CARD if row % 2 else BACKGROUND)
        ctk.CTkLabel(
            parent,
            text=self._clean(str(text)),
            height=34,
            width=width,
            anchor="w" if column == 0 else "center",
            justify="left" if column == 0 else "center",
            fg_color=background,
            text_color=TEXT if header or column == 0 else TEXT_SECONDARY,
            font=("Segoe UI", 12, "bold" if header else "normal"),
            corner_radius=0,
        ).grid(row=row, column=column, sticky="nsew", padx=1, pady=1)

    def _image_card(self, parent, title, image_path, row):
        card = self._section(parent, title, row=row)
        image_path = Path(image_path)
        if not image_path.exists():
            ctk.CTkLabel(
                card, text=f"Figure not found: {image_path.name}", text_color=ERROR
            ).pack(padx=18, pady=(0, 16))
            return
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.thumbnail((900, 430), Image.Resampling.LANCZOS)
            display = ctk.CTkImage(
                light_image=image, dark_image=image, size=image.size
            )
        self._images.append(display)
        ctk.CTkLabel(card, text="", image=display).pack(
            fill="x", padx=16, pady=(0, 16)
        )

    def _note(self, parent, title, text, row):
        card = ctk.CTkFrame(
            parent,
            fg_color=CARD,
            border_width=1,
            border_color=WARNING,
            corner_radius=12,
        )
        card.grid(row=row, column=0, sticky="ew", pady=8)
        ctk.CTkLabel(
            card, text=title, font=("Segoe UI", 14, "bold"), text_color=WARNING
        ).pack(anchor="w", padx=16, pady=(13, 4))
        ctk.CTkLabel(
            card,
            text=text,
            justify="left",
            anchor="w",
            wraplength=900,
            text_color=TEXT_SECONDARY,
        ).pack(fill="x", padx=16, pady=(0, 13))

    def _error_card(self, parent, title, message):
        card = ctk.CTkFrame(parent, fg_color=CARD, border_color=ERROR, border_width=1)
        card.grid(row=99, column=0, sticky="ew", pady=8)
        ctk.CTkLabel(
            card, text=title, font=("Segoe UI", 17, "bold"), text_color=ERROR
        ).pack(anchor="w", padx=16, pady=(14, 5))
        ctk.CTkLabel(
            card,
            text=message,
            justify="left",
            anchor="w",
            wraplength=900,
            text_color=TEXT_SECONDARY,
        ).pack(fill="x", padx=16, pady=(0, 14))

    def refresh(self):
        self._images.clear()
        self.tabview.destroy()
        self._build_results()

    @staticmethod
    def _read_csv(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Research result was not found:\n{path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _clean(value):
        return (
            str(value)
            .replace("â€”", "—")
            .replace("â€“", "–")
            .replace("â†’", "→")
        )

    @staticmethod
    def _number(value, decimals=2):
        try:
            return f"{float(value):.{decimals}f}"
        except (TypeError, ValueError):
            return "--"

    @staticmethod
    def _percent(value):
        try:
            number = float(value)
            if abs(number) <= 1.0:
                number *= 100.0
            return f"{number:.2f}%"
        except (TypeError, ValueError):
            return "--"

    @staticmethod
    def _mean_percent(rows, key):
        values = [float(item[key]) for item in rows]
        return f"{sum(values) * 100.0 / len(values):.2f}%" if values else "--"

    @staticmethod
    def _find_robustness_summary():
        candidates = sorted(
            RESEARCH_ROOT.glob(
                "robustness_benchmark/protocols/*/summary/"
                "logistic_regression/full/model_condition_summary.csv"
            )
        )
        if not candidates:
            raise FileNotFoundError("Robustness summary CSV was not found.")
        return candidates[-1]

    @staticmethod
    def _open_path(path):
        path = Path(path)
        if not path.exists():
            messagebox.showerror("Not Found", f"The requested file was not found:\n\n{path}")
            return
        try:
            os.startfile(str(path))
        except OSError as error:
            messagebox.showerror("Unable to Open", str(error))
