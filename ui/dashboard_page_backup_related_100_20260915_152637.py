# ==========================================================
# dashboard.py
# Professional Speaker Identification Dashboard
# Part 1
# ==========================================================

import os
import tkinter as tk
import customtkinter as ctk
import numpy as np
from pathlib import Path

try:
    import winsound
except ImportError:  # pragma: no cover - desktop target is Windows
    winsound = None

from tkinter import filedialog
from datetime import datetime
from tkinter import filedialog, messagebox
from ui.pdf_report import export_complete_report
from ui.waveform_page import WaveformPage
from ui.spectrogram_page import SpectrogramPage
from ui.settings_page import SettingsPage
from ui.research_results_page import ResearchResultsPage
from ui.theme import *
from ui.cards import (
    UploadCard,
    AudioInfoCard,
    PredictionCard,
)
from ui.sidebar import Sidebar
from ui.meter import ConfidenceMeter
from model_config import (
    DEFAULT_MODEL_ID,
    get_model_config,
)

from predict_multi import ( 
    CLASSIFIER_CONFIGS,
    DEFAULT_CLASSIFIER_ID,
    get_available_classifiers,
    get_speaker_classes,
    get_trained_models,
    predict_speaker,
)


class Dashboard(ctk.CTk):

    def __init__(self):

        super().__init__()

        # ---------------------------------------
        # Window
        # ---------------------------------------

        self.title("Speaker Identification System")

        self.geometry("1600x900")

        self.minsize(1200, 750)

        self.configure(
            fg_color=BACKGROUND
        )

        # Variables

        self.audio_path = None

        self.selected_model_id = DEFAULT_MODEL_ID
        self.selected_classifier_id = DEFAULT_CLASSIFIER_ID

        self.model_name_to_id = {}
        self.model_selector_var = None
        self.model_selector = None
        self.classifier_name_to_id = {}
        self.classifier_selector_var = None
        self.classifier_selector = None

        self.wave_plot = None

        self.spec_plot = None
        self.reference_audio_paths = []
        self.unknown_audio_history = []
        self.reference_embedding_cache = {}

        # Build UI
        self.last_speaker = None
        self.last_confidence = 0.0
        self.last_inference = 0.0
        self.last_probabilities = []
        self.last_prediction_diagnostics = {}

        self.create_layout()

    # =====================================================
    # Main Layout
    # =====================================================

    def create_layout(self):

        # Main window grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        # Sidebar
        self.sidebar = Sidebar(
            self,
            page_callback=self.show_page
        )

        self.sidebar.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        # Container holding every page
        self.page_container = ctk.CTkFrame(
            self,
            fg_color=BACKGROUND,
            corner_radius=0
        )

        self.page_container.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        self.page_container.grid_rowconfigure(0, weight=1)
        self.page_container.grid_columnconfigure(0, weight=1)

        # Dashboard page
        # Existing build_header() and build_dashboard()
        # methods create their widgets inside self.main.
        self.dashboard_page = ctk.CTkFrame(
            self.page_container,
            fg_color=BACKGROUND,
            corner_radius=0
        )

        # Keep this alias because build_header() may still use self.main.
        self.main = self.dashboard_page

        self.dashboard_page.grid_columnconfigure(0, weight=1)
        self.dashboard_page.grid_rowconfigure(1, weight=1)

        # Create the other pages
        self.waveform_page = WaveformPage(
            self.page_container
        )

        self.spectrogram_page = SpectrogramPage(
            self.page_container
        )

        self.settings_page = SettingsPage(
            self.page_container,
            on_settings_change=self.handle_settings_change
        )

        self.research_results_page = ResearchResultsPage(
            self.page_container
        )

        # Define pages BEFORE looping over them
        self.pages = {
            "dashboard": self.dashboard_page,
            "research": self.research_results_page,
            "waveform": self.waveform_page,
            "spectrogram": self.spectrogram_page,
            "settings": self.settings_page,
        }

        # Put all pages in the same position
        for page in self.pages.values():
            page.grid(
                row=0,
                column=0,
                sticky="nsew"
            )

        # Build dashboard content
        self.build_header()
        # ------------------------------------------------------
        # Model selector
        # ------------------------------------------------------

        trained_models = get_trained_models()

        if not trained_models:
            raise RuntimeError(
                "No trained speaker-identification models "
                "were found."
            )

        self.model_name_to_id = {
            model["display_name"]: model["model_id"]
            for model in trained_models
        }

        if self.selected_model_id not in {
            model["model_id"]
            for model in trained_models
        }:
            self.selected_model_id = (
                trained_models[0]["model_id"]
            )

        selected_config = get_model_config(
            self.selected_model_id
        )

        model_label = ctk.CTkLabel(
            self.header,
            text="Prediction Model:",
            font=("Segoe UI", 13, "bold"),
            text_color=TEXT_SECONDARY
        )

        model_label.grid(
            row=1,
            column=0,
            sticky="w",
            pady=(12, 0)
        )

        self.model_selector_var = tk.StringVar(
            value=selected_config["display_name"]
        )

        self.model_selector = ctk.CTkOptionMenu(
            self.header,
            variable=self.model_selector_var,
            values=list(self.model_name_to_id.keys()),
            width=260,
            command=self.change_prediction_model
        )

        self.model_selector.grid(
            row=1,
            column=1,
            sticky="w",
            padx=(8, 25),
            pady=(12, 0)
        )

        available_classifiers = get_available_classifiers(
            self.selected_model_id
        )
        if not available_classifiers:
            raise RuntimeError(
                "No trained classifier was found for the selected model."
            )

        self.classifier_name_to_id = {
            item["display_name"]: item["classifier_id"]
            for item in available_classifiers
        }
        available_classifier_ids = {
            item["classifier_id"]
            for item in available_classifiers
        }
        if self.selected_classifier_id not in available_classifier_ids:
            self.selected_classifier_id = available_classifiers[0][
                "classifier_id"
            ]

        classifier_label = ctk.CTkLabel(
            self.header,
            text="Classifier:",
            font=("Segoe UI", 13, "bold"),
            text_color=TEXT_SECONDARY,
        )
        classifier_label.grid(
            row=1,
            column=2,
            sticky="e",
            padx=(25, 8),
            pady=(12, 0),
        )

        classifier_display_name = CLASSIFIER_CONFIGS[
            self.selected_classifier_id
        ]["display_name"]
        self.classifier_selector_var = tk.StringVar(
            value=classifier_display_name
        )
        self.classifier_selector = ctk.CTkOptionMenu(
            self.header,
            variable=self.classifier_selector_var,
            values=list(self.classifier_name_to_id.keys()),
            width=210,
            command=self.change_prediction_classifier,
        )
        self.classifier_selector.grid(
            row=1,
            column=3,
            sticky="e",
            pady=(12, 0),
        )
        self.build_dashboard()
        self._update_prediction_model_label()

        # Open dashboard initially
        self.show_page("dashboard")
       
    # =====================================================
    # Header
    # =====================================================

    def build_header(self):

        self.header = ctk.CTkFrame(
            self.main,
            height=70,
            fg_color="transparent"
        )

        self.header.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=25,
            pady=(20, 10)
        )

        self.header.grid_columnconfigure(
            1,
            weight=1
        )

        heading = ctk.CTkLabel(
            self.header,
            text="Speaker Identification Dashboard",
            font=("Segoe UI", 28, "bold")
        )

        heading.grid(
            row=0,
            column=0,
            sticky="w"
        )

        status = ctk.CTkLabel(
            self.header,
            text="● Ready",
            text_color=SUCCESS,
            font=("Segoe UI", 15, "bold")
        )

        status.grid(
            row=0,
            column=1,
            sticky="e"
        )

    # =====================================================
    # Dashboard Grid
    # =====================================================

    def build_dashboard(self):
        
        speaker_classes = get_speaker_classes(
            model_id=self.selected_model_id,
            classifier_id=self.selected_classifier_id,
        )
        # Five enrolled speakers plus one fixed open-set decision row.
        number_of_speakers = len(speaker_classes) + 1
        
        self.dashboard_page.grid_rowconfigure(1, weight=1)
        self.dashboard_page.grid_columnconfigure(0, weight=1)

        self.content = ctk.CTkScrollableFrame(
            self.dashboard_page,
            fg_color="transparent",
            corner_radius=0,
            scrollbar_button_color="#475569",
            scrollbar_button_hover_color="#64748B",
        )

        self.content.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=25,
            pady=(0, 20)
        )

        # Grid

        # Three dashboard columns
        self.content.grid_columnconfigure(
            0,
            weight=2
        )

        self.content.grid_columnconfigure(
            1,
            weight=3
        )

        self.content.grid_columnconfigure(
            2,
            weight=2
        )

        # Top cards
        self.content.grid_rowconfigure(
            0,
            weight=0
        )

        # Probability distribution
        self.content.grid_rowconfigure(
            1,
            weight=0
        )

        # -------------------------
        # Upload Card
        # -------------------------

        self.upload_card = UploadCard(
            self.content,
            command=self.open_audio
        )

        self.upload_card.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=10,
            pady=10
        )

        # -------------------------
        # Audio Info
        # -------------------------

        self.audio_card = AudioInfoCard(
            self.content
        )

        self.audio_card.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=10,
            pady=10
        )

        # -------------------------
        # Prediction
        # -------------------------

        self.prediction_card = PredictionCard(
            self.content
        )

        self.prediction_card.grid(
            row=0,
            column=2,
            sticky="nsew",
            padx=10,
            pady=10
        )

        # Confidence Meter

        self.meter = ConfidenceMeter(
            self.prediction_card.body,
            size=180
        )

        self.meter.pack(
            pady=15
        )

        # =====================================================
        # Speaker Probability Distribution
        # =====================================================

        self.probability_card = ctk.CTkFrame(
            self.content,
            fg_color=CARD,
            corner_radius=18,
            border_width=1,
            border_color=BORDER
        )

        self.probability_card.grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="nsew",
            padx=10,
            pady=10
        )

        self.probability_card.grid_columnconfigure(
            0,
            weight=1
        )

        self.probability_card.grid_rowconfigure(
            2,
            weight=0
        )

        probability_title = ctk.CTkLabel(
            self.probability_card,
            text="📊 Speaker Probability Distribution",
            font=("Segoe UI", 20, "bold")
        )

        probability_title.grid(
            row=0,
            column=0,
            sticky="w",
            padx=20,
            pady=(18, 5)
        )

        # All speaker probabilities stay visible. The whole dashboard scrolls.
        self.probability_scroll = ctk.CTkFrame(
            self.probability_card,
            fg_color="transparent"
        )

        self.probability_scroll.grid(
            row=2,
            column=0,
            sticky="nsew",
            padx=(12, 8),
            pady=(5, 15)
        )

        self.probability_scroll.grid_columnconfigure(
            0,
            weight=1
        )

        self.probability_rows = []

        # Create one row for every trained speaker.
        # Your model currently has five speakers.

        for index in range(number_of_speakers):

            row = ctk.CTkFrame(
                self.probability_scroll,
                fg_color="#172033",
                corner_radius=10,
                height=44
            )

            row.grid(
                row=index,
                column=0,
                sticky="ew",
                padx=5,
                pady=3
            )

            row.grid_columnconfigure(1, weight=1)

            speaker_label = ctk.CTkLabel(
                row,
                text=(
                    f"{index + 1}.  Unknown Speaker"
                    if index == number_of_speakers - 1
                    else f"{index + 1}.  --"
                ),
                width=210,
                anchor="w",
                font=("Segoe UI", 14)
            )

            speaker_label.grid(
                row=0,
                column=0,
                sticky="w",
                padx=(14, 10),
                pady=8
            )

            probability_bar = ctk.CTkProgressBar(
                row,
                height=14,
                corner_radius=7,
                progress_color="#2563EB",
                fg_color="#334155"
            )

            probability_bar.grid(
                row=0,
                column=1,
                sticky="ew",
                padx=10,
                pady=8
            )

            probability_bar.set(0)

            probability_label = ctk.CTkLabel(
                row,
                text="0.00%",
                width=80,
                anchor="e",
                font=("Segoe UI", 14, "bold")
            )

            probability_label.grid(
                row=0,
                column=2,
                sticky="e",
                padx=(10, 14),
                pady=8
            )

            self.probability_rows.append(
                (
                    speaker_label,
                    probability_bar,
                    probability_label
                )
            )

        # =====================================================
        # Matched reference audio
        # =====================================================

        self.reference_audio_frame = ctk.CTkFrame(
            self.probability_card,
            fg_color="#172033",
            corner_radius=12,
            border_width=1,
            border_color=BORDER,
        )
        self.reference_audio_frame.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=17,
            pady=(0, 16),
        )
        self.reference_audio_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.reference_audio_frame,
            text="Related Audio Files",
            font=("Segoe UI", 15, "bold"),
            text_color=TEXT,
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(16, 10),
            pady=(13, 3),
        )

        self.reference_speaker_label = ctk.CTkLabel(
            self.reference_audio_frame,
            text="Speaker: --",
            font=("Segoe UI", 14, "bold"),
            text_color=TEXT,
            anchor="w",
        )
        self.reference_speaker_label.grid(
            row=0,
            column=1,
            sticky="e",
            padx=10,
            pady=(13, 3),
        )

        ctk.CTkButton(
            self.reference_audio_frame,
            text="Stop Audio",
            width=100,
            fg_color="#475569",
            hover_color="#334155",
            command=self.stop_reference_audio,
        ).grid(
            row=0,
            column=2,
            padx=(10, 14),
            pady=(10, 3),
        )

        self.reference_message_label = ctk.CTkLabel(
            self.reference_audio_frame,
            text="No related audio selected",
            font=("Segoe UI", 12),
            text_color=TEXT_SECONDARY,
            anchor="w",
        )
        self.reference_message_label.grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=16,
            pady=(4, 13),
        )

        self.reference_list_frame = ctk.CTkFrame(
            self.reference_audio_frame,
            fg_color="transparent",
        )
        self.reference_list_frame.grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=12,
            pady=(3, 12),
        )
        self.reference_list_frame.grid_columnconfigure(0, weight=1)

        self.reference_audio_rows = []
        for index in range(5):
            audio_row = ctk.CTkFrame(
                self.reference_list_frame,
                fg_color="#111827",
                corner_radius=9,
            )
            audio_row.grid_columnconfigure(0, weight=1)

            file_label = ctk.CTkLabel(
                audio_row,
                text=f"{index + 1}. --",
                anchor="w",
                font=("Segoe UI", 12),
                text_color=TEXT,
            )
            file_label.grid(
                row=0,
                column=0,
                sticky="ew",
                padx=(12, 8),
                pady=8,
            )

            play_button = ctk.CTkButton(
                audio_row,
                text="Play",
                width=72,
                command=lambda item=index: self.play_reference_audio(item),
            )
            play_button.grid(row=0, column=1, padx=5, pady=6)

            open_button = ctk.CTkButton(
                audio_row,
                text="Open File",
                width=90,
                fg_color="#475569",
                hover_color="#334155",
                command=lambda item=index: self.open_reference_audio(item),
            )
            open_button.grid(row=0, column=2, padx=(5, 8), pady=6)

            audio_row.grid_remove()
            self.reference_audio_rows.append(
                (audio_row, file_label, play_button, open_button)
            )
    # =====================================================
    # show page
    # =====================================================
    def show_page(self, page_name):

        # Export is a sidebar action, not a separate page.
        if page_name == "export_pdf":
            self.export_complete_pdf()
            return

        page = self.pages.get(page_name)

        if page is None:
            print(f"Unknown page: {page_name}")
            return

        page.tkraise()
        self.sidebar.set_active_button(page_name)

        print(f"Opened page: {page_name}")

    def handle_settings_change(self, settings):

        sample_rate = settings["sample_rate"]
        colormap = settings["spectrogram_colormap"]
        self.prediction_threshold = float(
            settings["prediction_threshold"]
        )

        self.waveform_page = self.pages["waveform"]
        self.spectrogram_page = self.pages["spectrogram"]

        self.waveform_page.default_sample_rate = sample_rate

        self.spectrogram_page.set_default_sample_rate(
            sample_rate
        )

        self.spectrogram_page.set_colormap(
            colormap
        )
# =====================================================
# Open Audio File
# =====================================================
    
    def open_audio(self):
        
        print("Open Audio Clicked")

        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Wave Files", "*.wav"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        self.audio_path = file_path

        self.load_audio(file_path)


# =====================================================
# Load Audio
# =====================================================

    def load_audio(self, audio_path):

        try:

            import librosa
            import soundfile as sf

            y, sr = librosa.load(audio_path, sr=None, mono=False)

            info = sf.info(audio_path)

            duration = info.duration
            sample_rate = info.samplerate
            channels = info.channels
            bit_depth = info.subtype
            file_size = os.path.getsize(audio_path) / 1024

            filename = os.path.basename(audio_path)

            # Update upload card

            self.upload_card.filename.configure(
                text=filename
            )

            # Update audio information

            self.audio_card.update_info(
                filename,
                duration,
                sample_rate,
                channels,
                bit_depth,
                file_size
            )

            waveform_page = self.pages.get("waveform")
            spectrogram_page = self.pages.get("spectrogram")
            auto_plot = (
                self.settings_page.get_settings().get("auto_plot", True)
                if hasattr(self, "settings_page")
                else True
            )

            if auto_plot and waveform_page is not None:
                waveform_page.load_audio(audio_path)

            if auto_plot and spectrogram_page is not None:
                spectrogram_page.load_audio(audio_path)

            # Predict

            self.run_prediction(audio_path)

        except Exception as e:

            self.show_error(str(e))

    # =====================================================
    # Change prediction model
    # =====================================================

    def change_prediction_model(
        self,
        selected_display_name
    ):

        if (
            selected_display_name
            not in self.model_name_to_id
        ):
            self.show_error(
                "The selected model is not available."
            )
            return

        self.selected_model_id = (
            self.model_name_to_id[
                selected_display_name
            ]
        )

        print(
            f"Selected prediction model: "
            f"{selected_display_name}"
        )

        self._refresh_classifier_selector()
        self._update_prediction_model_label()

        if self.audio_path:
            self.run_prediction(
                self.audio_path
            )

    def _refresh_classifier_selector(self):
        """Keep the classifier menu valid when the embedding model changes."""

        available = get_available_classifiers(
            self.selected_model_id
        )
        if not available:
            raise RuntimeError(
                "No trained classifier is available for the selected model."
            )

        self.classifier_name_to_id = {
            item["display_name"]: item["classifier_id"]
            for item in available
        }
        available_ids = {
            item["classifier_id"]
            for item in available
        }
        if self.selected_classifier_id not in available_ids:
            self.selected_classifier_id = available[0]["classifier_id"]

        selected_name = CLASSIFIER_CONFIGS[
            self.selected_classifier_id
        ]["display_name"]
        if self.classifier_selector is not None:
            self.classifier_selector.configure(
                values=list(self.classifier_name_to_id.keys())
            )
        if self.classifier_selector_var is not None:
            self.classifier_selector_var.set(selected_name)

    def change_prediction_classifier(self, selected_display_name):
        """Switch the identity classifier without re-extracting embeddings."""

        if selected_display_name not in self.classifier_name_to_id:
            self.show_error("The selected classifier is not available.")
            return

        self.selected_classifier_id = self.classifier_name_to_id[
            selected_display_name
        ]
        print(f"Selected classifier: {selected_display_name}")
        self._update_prediction_model_label()

        if self.audio_path:
            self.run_prediction(self.audio_path)

    def _update_prediction_model_label(self):
        if not (
            hasattr(self, "prediction_card")
            and hasattr(self.prediction_card, "model_label")
        ):
            return

        model_name = get_model_config(
            self.selected_model_id
        )["display_name"]
        classifier_name = CLASSIFIER_CONFIGS[
            self.selected_classifier_id
        ]["display_name"]
        self.prediction_card.model_label.configure(
            text=(
                f"Model: {model_name} | "
                f"Classifier: {classifier_name}"
            )
        )
    # =====================================================
    # Speaker prediction
    # =====================================================

    def run_prediction(self, audio_path):

        try:
            current_settings = (
                self.settings_page.get_settings()
                if hasattr(self, "settings_page")
                else {}
            )
            rejection_threshold = current_settings.get(
                "prediction_threshold",
                60.0,
            )

            (
                speaker,
                confidence,
                inference_time,
                probabilities,
                diagnostics,
            ) = predict_speaker(
                audio_path,
                model_id=self.selected_model_id,
                classifier_id=self.selected_classifier_id,
                rejection_threshold=rejection_threshold,
                return_diagnostics=True,
            )

            self.prediction_card.update_prediction(
                speaker,
                inference_time
            )

            displayed_probabilities = self.update_probability_distribution(
                probabilities,
                predicted_speaker=speaker,
                known_speaker_probability=diagnostics.get(
                    "known_speaker_probability"
                ),
            )

            display_confidence = next(
                (
                    probability
                    for name, probability in displayed_probabilities
                    if name == speaker
                ),
                confidence,
            )
            self.meter.set_value(display_confidence)

            self.update_reference_audio(
                speaker,
                audio_path,
                query_embedding=diagnostics.get("embedding"),
                model_id=self.selected_model_id,
            )

            selected_config = get_model_config(
                self.selected_model_id
            )

            self._update_prediction_model_label()

            # Save information for complete PDF export.
            self.last_speaker = speaker
            self.last_confidence = display_confidence
            self.last_inference = inference_time
            self.last_probabilities = displayed_probabilities
            self.last_prediction_diagnostics = diagnostics
            self.last_model_id = (
                self.selected_model_id
            )
            self.last_model_name = (
                selected_config["display_name"]
            )
            self.last_classifier_id = self.selected_classifier_id
            self.last_classifier_name = CLASSIFIER_CONFIGS[
                self.selected_classifier_id
            ]["display_name"]

        except Exception as error:
            self.show_error(str(error))

    # =====================================================
    # Matched reference audio
    # =====================================================

    def update_reference_audio(
        self,
        speaker,
        input_audio_path=None,
        query_embedding=None,
        model_id=None,
    ):
        """Display the five most similar reference embeddings."""

        self.stop_reference_audio()
        display_name = str(speaker).replace("_", " ") if speaker else "--"

        if speaker in (None, "--"):
            self._set_reference_audio_list(
                display_name="--",
                paths=[],
                message="No related audio selected",
            )
            return

        if speaker == "Unknown Speaker":
            if input_audio_path:
                unknown_path = Path(input_audio_path).resolve()
                if unknown_path.exists():
                    self.unknown_audio_history = [
                        path
                        for path in self.unknown_audio_history
                        if path != unknown_path
                    ]
                    self.unknown_audio_history.insert(0, unknown_path)
                    self.unknown_audio_history = self.unknown_audio_history[:5]

            related_unknown, related_labels = self._find_similar_reference_audio(
                query_embedding=query_embedding,
                model_id=model_id or self.selected_model_id,
                speaker=None,
                input_audio_path=input_audio_path,
                unknown_pool=True,
            )
            unknown_source_name = (
                str(related_labels[0]).replace("LibriSpeech_", "LibriSpeech ID ")
                if related_labels
                else None
            )
            if not related_unknown:
                related_unknown, unknown_source_name = (
                    self._find_unknown_reference_audio(input_audio_path)
                )
            unknown_paths = related_unknown or self.unknown_audio_history
            unknown_display_name = (
                f"Unknown Speaker ({unknown_source_name})"
                if unknown_source_name
                else "Unknown Speaker"
            )
            self._set_reference_audio_list(
                display_name=unknown_display_name,
                paths=unknown_paths,
                message=(
                    "No matching unknown reference speaker was found. "
                    "Recent unknown inputs will appear here."
                ),
                is_unknown=True,
            )
            return

        candidates, _ = self._find_similar_reference_audio(
            query_embedding=query_embedding,
            model_id=model_id or self.selected_model_id,
            speaker=str(speaker),
            input_audio_path=input_audio_path,
            unknown_pool=False,
        )

        if not candidates:
            self._set_reference_audio_list(
                display_name=display_name,
                paths=[],
                message="No WAV reference file was found for this speaker.",
            )
            return

        self._set_reference_audio_list(
            display_name=display_name,
            paths=candidates[:5],
            message="No related audio selected",
        )

    def _load_reference_embedding_index(self, model_id, unknown_pool=False):
        """Load and cache normalized embeddings, labels and source paths."""

        cache_key = (str(model_id), bool(unknown_pool))
        if cache_key in self.reference_embedding_cache:
            return self.reference_embedding_cache[cache_key]

        project_root = Path(__file__).resolve().parents[1]
        if unknown_pool:
            directory = (
                project_root
                / "research_results"
                / "embeddings"
                / "librispeech_dev_clean"
                / str(model_id)
                / "full"
            )
        else:
            directory = project_root / "embeddings" / str(model_id)

        required = {
            "X": directory / "X.npy",
            "y": directory / "y.npy",
            "paths": directory / "audio_paths.npy",
        }
        if not all(path.exists() for path in required.values()):
            return None

        embeddings = np.asarray(np.load(required["X"]), dtype=np.float32)
        labels = np.asarray(np.load(required["y"]), dtype=str)
        paths = np.asarray(np.load(required["paths"]), dtype=str)
        if not (len(embeddings) == len(labels) == len(paths)):
            raise ValueError(
                f"Reference index length mismatch for {model_id}."
            )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        index = (embeddings, labels, paths)
        self.reference_embedding_cache[cache_key] = index
        return index

    def _find_similar_reference_audio(
        self,
        query_embedding,
        model_id,
        speaker=None,
        input_audio_path=None,
        unknown_pool=False,
    ):
        """Rank reference files by cosine similarity to the query embedding."""

        if query_embedding is None or not model_id:
            return [], []
        index = self._load_reference_embedding_index(
            model_id,
            unknown_pool=unknown_pool,
        )
        if index is None:
            return [], []

        embeddings, labels, paths = index
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.size != embeddings.shape[1] or not np.isfinite(query).all():
            return [], []
        query /= max(float(np.linalg.norm(query)), 1e-12)

        candidate_indices = np.arange(len(labels))
        if speaker is not None:
            candidate_indices = candidate_indices[labels == str(speaker)]
        if candidate_indices.size == 0:
            return [], []

        input_resolved = None
        if input_audio_path:
            try:
                input_resolved = Path(input_audio_path).resolve()
            except OSError:
                input_resolved = None

        scores = embeddings[candidate_indices] @ query
        ordered = candidate_indices[np.argsort(scores)[::-1]]
        selected_paths = []
        selected_labels = []
        for index_value in ordered:
            path = Path(paths[index_value])
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[1] / path
            try:
                if input_resolved is not None and path.resolve() == input_resolved:
                    continue
            except OSError:
                continue
            if not path.is_file():
                continue
            selected_paths.append(path)
            selected_labels.append(str(labels[index_value]))
            if len(selected_paths) == 5:
                break
        return selected_paths, selected_labels

    def _find_unknown_reference_audio(self, input_audio_path):
        """Find five LibriSpeech clips from the same unknown source speaker."""

        if not input_audio_path:
            return [], None

        project_root = Path(__file__).resolve().parents[1]
        unknown_root = (
            project_root
            / "research_datasets"
            / "LibriSpeech"
            / "dev-clean"
        )
        if not unknown_root.exists():
            return [], None

        input_path = Path(input_audio_path).resolve()
        speaker_id = None

        try:
            relative_path = input_path.relative_to(unknown_root.resolve())
            if len(relative_path.parts) >= 2:
                speaker_id = relative_path.parts[0]
        except ValueError:
            # LibriSpeech filenames begin with speaker-id, so copied files can
            # still be mapped back to the local unknown reference collection.
            filename_parts = input_path.stem.split("-")
            if filename_parts and filename_parts[0].isdigit():
                candidate_id = filename_parts[0]
                if (unknown_root / candidate_id).is_dir():
                    speaker_id = candidate_id

        if not speaker_id:
            return [], None

        speaker_directory = unknown_root / speaker_id
        candidates = sorted(
            path
            for path in speaker_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in {".wav", ".flac"}
        )

        alternatives = [path for path in candidates if path.resolve() != input_path]
        if alternatives:
            candidates = alternatives
        if not candidates:
            return [], f"LibriSpeech ID {speaker_id}"

        selected_index = sum(str(input_path).encode("utf-8")) % len(candidates)
        ordered_candidates = (
            candidates[selected_index:] + candidates[:selected_index]
        )
        return ordered_candidates[:5], f"LibriSpeech ID {speaker_id}"

    def _get_related_audio_paths_for_report(
        self,
        speaker,
        input_audio_path,
        query_embedding=None,
        model_id=None,
    ):
        """Return cosine-ranked references used by the dashboard."""

        if speaker in (None, "--"):
            return []

        if speaker == "Unknown Speaker":
            related_unknown, _ = self._find_similar_reference_audio(
                query_embedding=query_embedding,
                model_id=model_id or self.selected_model_id,
                speaker=None,
                input_audio_path=input_audio_path,
                unknown_pool=True,
            )
            if related_unknown:
                return related_unknown[:5]
            related_unknown, _ = self._find_unknown_reference_audio(
                input_audio_path
            )
            return list(related_unknown or self.unknown_audio_history)[:5]

        candidates, _ = self._find_similar_reference_audio(
            query_embedding=query_embedding,
            model_id=model_id or self.selected_model_id,
            speaker=str(speaker),
            input_audio_path=input_audio_path,
            unknown_pool=False,
        )
        return candidates[:5]

    def _set_reference_audio_list(
        self,
        display_name,
        paths,
        message,
        is_unknown=False,
    ):
        self.reference_audio_paths = [Path(path) for path in paths[:5]]
        self.reference_speaker_label.configure(
            text=f"Speaker: {display_name}",
            text_color=WARNING if is_unknown else (SUCCESS if paths else TEXT),
        )

        if self.reference_audio_paths:
            self.reference_message_label.grid_remove()
        else:
            self.reference_message_label.configure(text=message)
            self.reference_message_label.grid()

        for index, row_widgets in enumerate(self.reference_audio_rows):
            audio_row, file_label, play_button, open_button = row_widgets
            if index >= len(self.reference_audio_paths):
                audio_row.grid_remove()
                continue

            path = self.reference_audio_paths[index]
            duration_text = ""
            try:
                import soundfile as sf

                duration_text = f"  |  {sf.info(str(path)).duration:.2f} seconds"
            except Exception:
                pass

            file_label.configure(
                text=f"{index + 1}. {path.name}{duration_text}"
            )
            play_button.configure(state="normal")
            open_button.configure(state="normal")
            audio_row.grid(
                row=index,
                column=0,
                sticky="ew",
                padx=4,
                pady=3,
            )

    def play_reference_audio(self, index=0):
        if index >= len(self.reference_audio_paths):
            return

        audio_path = self.reference_audio_paths[index]
        try:
            if winsound is not None and audio_path.suffix.lower() == ".wav":
                winsound.PlaySound(
                    str(audio_path),
                    winsound.SND_FILENAME | winsound.SND_ASYNC,
                )
            else:
                os.startfile(str(audio_path))
        except (OSError, RuntimeError) as error:
            self.show_error(f"Unable to play the reference audio.\n\n{error}")

    def open_reference_audio(self, index):
        if index >= len(self.reference_audio_paths):
            return
        try:
            os.startfile(str(self.reference_audio_paths[index]))
        except OSError as error:
            self.show_error(f"Unable to open the audio file.\n\n{error}")

    @staticmethod
    def stop_reference_audio():
        if winsound is not None:
            try:
                winsound.PlaySound(None, 0)
            except RuntimeError:
                pass

    def clear_reference_audio(self):
        self.stop_reference_audio()
        self.reference_audio_paths = []
        self.reference_speaker_label.configure(text="Speaker: --", text_color=TEXT)
        self.reference_message_label.configure(text="No related audio selected")
        self.reference_message_label.grid()
        for audio_row, _, play_button, open_button in self.reference_audio_rows:
            play_button.configure(state="disabled")
            open_button.configure(state="disabled")
            audio_row.grid_remove()

    def update_probability_distribution(
        self,
        probabilities,
        predicted_speaker=None,
        known_speaker_probability=None,
    ):
        """Render a hierarchical known/unknown probability distribution."""

        displayed = self.compose_probability_distribution(
            probabilities,
            known_speaker_probability=known_speaker_probability,
        )

        for index, widgets in enumerate(self.probability_rows):
            speaker_label, probability_bar, probability_label = widgets
            speaker, probability = displayed[index]
            is_unknown_row = index == len(self.probability_rows) - 1
            is_winner = speaker == predicted_speaker

            speaker_label.configure(
                text=f"{index + 1}.  {speaker.replace('_', ' ')}"
            )
            probability_bar.set(
                max(0.0, min(probability / 100.0, 1.0))
            )
            probability_label.configure(text=f"{probability:.2f}%")

            if is_unknown_row:
                row_color = WARNING if is_winner else "#64748B"
            else:
                row_color = "#22C55E" if is_winner else "#2563EB"

            probability_bar.configure(progress_color=row_color)
            probability_label.configure(
                text_color=row_color if is_winner else TEXT
            )

        return displayed

    @staticmethod
    def compose_probability_distribution(
        probabilities,
        known_speaker_probability=None,
    ):
        """Combine conditional identity scores with calibrated open-set score.

        The five classifier probabilities are conditional on the voice being
        enrolled.  Multiplying them by P(known) and assigning 1-P(known) to the
        fixed Unknown Speaker row produces one transparent six-row hierarchy
        that sums to 100 percent.
        """

        cleaned = [
            (str(speaker), float(probability))
            for speaker, probability in (probabilities or [])
            if str(speaker) != "Unknown Speaker"
        ]

        total = sum(probability for _, probability in cleaned)

        # Convert a 0-1 distribution to percentages.
        if 0 < total <= 1.01:
            cleaned = [
                (speaker, probability * 100.0)
                for speaker, probability in cleaned
            ]
            total *= 100.0

        # Keep the five known-speaker values as one valid distribution.
        if total > 0 and abs(total - 100.0) > 0.5:
            cleaned = [
                (speaker, (probability / total) * 100.0)
                for speaker, probability in cleaned
            ]

        cleaned.sort(key=lambda item: item[1], reverse=True)

        if known_speaker_probability is None:
            known_probability = 1.0
        else:
            known_probability = max(
                0.0,
                min(float(known_speaker_probability), 1.0),
            )

        displayed = [
            (speaker, probability * known_probability)
            for speaker, probability in cleaned[:5]
        ]
        while len(displayed) < 5:
            displayed.append(("--", 0.0))
        displayed.append(
            ("Unknown Speaker", (1.0 - known_probability) * 100.0)
        )
        return displayed

    # =====================================================
    # Reset Dashboard
    # =====================================================

    def reset_dashboard(self):

        self.audio_path = None

        self.upload_card.filename.configure(
            text="No audio selected"
        )

        self.audio_card.update_info(
            "--",
            0,
            "--",
            "--",
            "--",
            0
        )

        self.prediction_card.update_prediction(
            "--",
            0
        )

        self.meter.reset()

        waveform_page = self.pages.get("waveform")
        spectrogram_page = self.pages.get("spectrogram")
        if waveform_page is not None:
            waveform_page.clear()
        if spectrogram_page is not None:
            spectrogram_page.clear()

        self.last_speaker = None
        self.last_confidence = 0.0
        self.last_inference = 0.0
        self.last_probabilities = []
        self.last_prediction_diagnostics = {}
        self.last_classifier_id = self.selected_classifier_id
        self.last_classifier_name = CLASSIFIER_CONFIGS[
            self.selected_classifier_id
        ]["display_name"]
        self.clear_reference_audio()

        for index, widgets in enumerate(self.probability_rows):
            speaker_label, probability_bar, probability_label = widgets
            speaker_label.configure(
                text=(
                    f"{index + 1}.  Unknown Speaker"
                    if index == len(self.probability_rows) - 1
                    else f"{index + 1}.  --"
                )
            )
            probability_bar.set(0)
            probability_label.configure(text="0.00%", text_color=TEXT)


    # =====================================================
    # Error Popup
    # =====================================================

    def show_error(self, message):

        dialog = ctk.CTkToplevel(self)

        dialog.title("Error")

        dialog.geometry("420x180")

        dialog.grab_set()

        label = ctk.CTkLabel(
            dialog,
            text="An error occurred",
            font=("Segoe UI", 20, "bold")
        )

        label.pack(pady=(20, 10))

        msg = ctk.CTkLabel(
            dialog,
            text=message,
            wraplength=360,
            justify="center"
        )

        msg.pack(padx=20)

        btn = ctk.CTkButton(
            dialog,
            text="OK",
            command=dialog.destroy
        )

        btn.pack(pady=20)

    def export_complete_pdf(self):
            """Export one PDF containing results from every trained model."""

            if not self.audio_path:
                messagebox.showwarning(
                    "No Audio",
                    "Please select and analyse an audio file first.",
                )
                return

            trained_models = get_trained_models(
                classifier_id=self.selected_classifier_id
            )

            if not trained_models:
                messagebox.showwarning(
                    "No Trained Models",
                    "No trained speaker-identification models were found.",
                )
                return

            default_name = (
                f"multi_model_speaker_report_"
                f"{datetime.now():%Y%m%d_%H%M%S}.pdf"
            )

            output_path = filedialog.asksaveasfilename(
                title="Export Multi-Model PDF Report",
                initialfile=default_name,
                defaultextension=".pdf",
                filetypes=[
                    ("PDF document", "*.pdf"),
                ],
            )

            if not output_path:
                return

            # Used only when an older model_config.py does not yet contain
            # the embedding_dim field.
            known_embedding_dimensions = {
                "speechbrain_ecapa": 192,
                "speechbrain_xvector": 512,
                "wavlm_base_plus_sv": 512,
                "unispeech_sat_base_plus_sv": 512,
            }

            model_results = []

            try:
                self.configure(cursor="watch")
                self.update_idletasks()

                for model_number, model_info in enumerate(
                    trained_models,
                    start=1,
                ):
                    model_id = model_info["model_id"]
                    config = get_model_config(model_id)

                    result = {
                        "model_id": model_id,
                        "display_name": config.get(
                            "display_name",
                            model_id,
                        ),
                        "provider": config.get("provider", "-"),
                        "architecture": config.get(
                            "architecture",
                            "-",
                        ),
                        "pretrained_name": config.get(
                            "pretrained_name",
                            config.get("source", "-"),
                        ),
                        "embedding_dim": config.get(
                            "embedding_dim",
                            known_embedding_dimensions.get(
                                model_id,
                                "-",
                            ),
                        ),
                        "classifier_id": self.selected_classifier_id,
                        "classifier_name": CLASSIFIER_CONFIGS[
                            self.selected_classifier_id
                        ]["display_name"],
                    }

                    try:
                        (
                            speaker,
                            confidence,
                            inference_time,
                            probabilities,
                            diagnostics,
                        ) = predict_speaker(
                            self.audio_path,
                            model_id=model_id,
                            classifier_id=self.selected_classifier_id,
                            rejection_threshold=(
                                self.settings_page.get_settings().get(
                                    "prediction_threshold",
                                    60.0,
                                )
                                if hasattr(self, "settings_page")
                                else 60.0
                            ),
                            return_diagnostics=True,
                        )

                        displayed_probabilities = (
                            self.compose_probability_distribution(
                                probabilities,
                                known_speaker_probability=diagnostics.get(
                                    "known_speaker_probability"
                                ),
                            )
                        )
                        final_confidence = next(
                            (
                                probability
                                for name, probability in displayed_probabilities
                                if name == speaker
                            ),
                            confidence,
                        )

                        result.update({
                            "speaker": speaker,
                            "confidence": final_confidence,
                            "closed_set_confidence": confidence,
                            "inference_time": inference_time,
                            "probabilities": displayed_probabilities,
                            "known_speaker_probability": diagnostics.get(
                                "known_speaker_probability"
                            ),
                            "unknown_speaker_probability": diagnostics.get(
                                "unknown_speaker_probability"
                            ),
                            "open_set_threshold": diagnostics.get(
                                "open_set_threshold"
                            ),
                            "open_set_detector_model_name": diagnostics.get(
                                "open_set_detector_model_name"
                            ),
                            "related_audio_files": [
                                str(path)
                                for path in self._get_related_audio_paths_for_report(
                                    speaker,
                                    self.audio_path,
                                    query_embedding=diagnostics.get("embedding"),
                                    model_id=model_id,
                                )
                            ],
                        })

                    except Exception as model_error:
                        # One model failure should not stop the complete
                        # report. That model receives its own error page.
                        result["error"] = str(model_error)

                    model_results.append(result)

                    self.title(
                        "Speaker Identification System - "
                        f"Preparing model {model_number}/"
                        f"{len(trained_models)}"
                    )
                    self.update_idletasks()

                settings = (
                    self.settings_page.get_settings()
                    if hasattr(self, "settings_page")
                    else {}
                )
                colormap = settings.get(
                    "spectrogram_colormap",
                    "magma",
                )

                saved_path = export_complete_report(
                    output_path=output_path,
                    audio_path=self.audio_path,
                    model_results=model_results,
                    colormap=colormap,
                )

                successful_count = sum(
                    1
                    for result in model_results
                    if not result.get("error")
                )
                failed_count = (
                    len(model_results) - successful_count
                )

                messagebox.showinfo(
                    "PDF Exported",
                    (
                        "Multi-model report saved successfully.\n\n"
                        f"Successful models: {successful_count}\n"
                        f"Failed models: {failed_count}\n\n"
                        f"{saved_path}"
                    ),
                )

            except Exception as error:
                messagebox.showerror(
                    "PDF Export Error",
                    str(error),
                )

            finally:
                self.configure(cursor="")
                self.title("Speaker Identification System")
    # =====================================================
    # Run Application
    # =====================================================

    def start(self):

        self.mainloop()
