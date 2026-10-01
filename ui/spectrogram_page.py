# ==========================================================
# spectrogram_page.py
# High-Resolution Mel Spectrogram Viewer
# ==========================================================

from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import librosa
import librosa.display
import numpy as np
import soundfile as sf

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure

from ui.theme import BACKGROUND, CARD


class SpectrogramPage(ctk.CTkFrame):
    """Interactive Mel spectrogram page."""

    SUPPORTED_COLORMAPS = (
        "magma",
        "inferno",
        "plasma",
        "viridis",
        "cividis",
        "turbo",
    )

    def __init__(
        self,
        parent,
        default_sample_rate=16000,
        on_audio_loaded=None,
    ):
        super().__init__(parent, fg_color=BACKGROUND)

        self.default_sample_rate = int(default_sample_rate)
        self.on_audio_loaded = on_audio_loaded

        self.audio_path = None
        self.audio = None
        self.sample_rate = None
        self.mel_db = None

        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 128

        self.colormap_var = ctk.StringVar(value="magma")

        self._build_page()
        self._draw_empty_plot()

    # ======================================================
    # Page layout
    # ======================================================

    def _build_page(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._build_header()
        self._build_controls()
        self._build_plot_card()
        self._build_statistics()

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=20,
            pady=(20, 8),
        )
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="🌈 Mel Spectrogram",
            font=("Segoe UI", 28, "bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        self.status_label = ctk.CTkLabel(
            header,
            text="● Ready",
            font=("Segoe UI", 14, "bold"),
            text_color="#22C55E",
        )
        self.status_label.grid(row=0, column=1, sticky="e")

    def _build_controls(self):
        controls = ctk.CTkFrame(
            self,
            fg_color=CARD,
            corner_radius=14,
        )
        controls.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=15,
            pady=8,
        )

        for column in range(7):
            controls.grid_columnconfigure(column, weight=1)

        open_button = ctk.CTkButton(
            controls,
            text="📂 Open Audio",
            height=40,
            command=self.open_audio,
        )
        open_button.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(12, 5),
            pady=12,
        )

        save_button = ctk.CTkButton(
            controls,
            text="💾 Save PNG",
            height=40,
            command=self.save_png,
        )
        save_button.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=5,
            pady=12,
        )

        pdf_button = ctk.CTkButton(
            controls,
            text="🖨 Export PDF",
            height=40,
            command=self.export_pdf,
        )
        pdf_button.grid(
            row=0,
            column=2,
            sticky="ew",
            padx=5,
            pady=12,
        )

        clear_button = ctk.CTkButton(
            controls,
            text="🗑 Clear",
            height=40,
            fg_color="#475569",
            hover_color="#334155",
            command=self.clear,
        )
        clear_button.grid(
            row=0,
            column=3,
            sticky="ew",
            padx=5,
            pady=12,
        )

        colormap_label = ctk.CTkLabel(
            controls,
            text="Colormap:",
            font=("Segoe UI", 13),
        )
        colormap_label.grid(
            row=0,
            column=4,
            sticky="e",
            padx=(10, 5),
        )

        colormap_menu = ctk.CTkOptionMenu(
            controls,
            values=list(self.SUPPORTED_COLORMAPS),
            variable=self.colormap_var,
            command=self._on_colormap_changed,
        )
        colormap_menu.grid(
            row=0,
            column=5,
            sticky="ew",
            padx=5,
        )

        reset_view_button = ctk.CTkButton(
            controls,
            text="Reset View",
            width=100,
            height=40,
            command=self.reset_view,
        )
        reset_view_button.grid(
            row=0,
            column=6,
            sticky="ew",
            padx=(5, 12),
            pady=12,
        )

    def _build_plot_card(self):
        self.plot_card = ctk.CTkFrame(
            self,
            fg_color=CARD,
            corner_radius=15,
        )
        self.plot_card.grid(
            row=2,
            column=0,
            sticky="nsew",
            padx=15,
            pady=8,
        )
        self.plot_card.grid_columnconfigure(0, weight=1)
        self.plot_card.grid_rowconfigure(0, weight=1)

        self.figure = Figure(
            figsize=(12, 5.5),
            dpi=100,
            facecolor="#111827",
        )
        self.axes = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=self.plot_card,
        )
        self.canvas.get_tk_widget().grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=10,
            pady=(10, 0),
        )

        toolbar_container = ctk.CTkFrame(
            self.plot_card,
            fg_color="transparent",
        )
        toolbar_container.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=10,
            pady=(0, 10),
        )

        self.toolbar = NavigationToolbar2Tk(
            self.canvas,
            toolbar_container,
            pack_toolbar=False,
        )
        self.toolbar.update()
        self.toolbar.pack(side="left")

    def _build_statistics(self):
        statistics = ctk.CTkFrame(
            self,
            fg_color=CARD,
            corner_radius=12,
        )
        statistics.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=15,
            pady=(8, 20),
        )

        for column in range(6):
            statistics.grid_columnconfigure(column, weight=1)

        self.file_label = self._create_stat(
            statistics, 0, "File", "--"
        )
        self.duration_label = self._create_stat(
            statistics, 1, "Duration", "--"
        )
        self.sample_rate_label = self._create_stat(
            statistics, 2, "Sample Rate", "--"
        )
        self.fft_label = self._create_stat(
            statistics, 3, "FFT Size", str(self.n_fft)
        )
        self.hop_label = self._create_stat(
            statistics, 4, "Hop Length", str(self.hop_length)
        )
        self.mels_label = self._create_stat(
            statistics, 5, "Mel Bands", str(self.n_mels)
        )

    @staticmethod
    def _create_stat(parent, column, title, initial_value):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(
            row=0,
            column=column,
            sticky="ew",
            padx=10,
            pady=12,
        )

        ctk.CTkLabel(
            frame,
            text=title,
            font=("Segoe UI", 12),
            text_color="#94A3B8",
        ).pack()

        value_label = ctk.CTkLabel(
            frame,
            text=initial_value,
            font=("Segoe UI", 14, "bold"),
        )
        value_label.pack(pady=(3, 0))

        return value_label

    # ======================================================
    # Audio operations
    # ======================================================

    def open_audio(self):
        path = filedialog.askopenfilename(
            title="Select Audio",
            filetypes=[
                ("Wave files", "*.wav"),
                ("Audio files", "*.wav *.flac *.mp3 *.ogg *.m4a"),
                ("All files", "*.*"),
            ],
        )

        if path:
            self.load_audio(path)

    def load_audio(self, audio_path):
        try:
            path = Path(audio_path)

            if not path.is_file():
                raise FileNotFoundError(
                    f"Audio file was not found:\n{path}"
                )

            self._set_status("● Loading...", "#F59E0B")
            self.update_idletasks()

            self.audio_path = str(path)

            self.audio, self.sample_rate = librosa.load(
                self.audio_path,
                sr=self.default_sample_rate,
                mono=True,
            )

            if self.audio.size == 0:
                raise ValueError("The selected audio file is empty.")

            if not np.isfinite(self.audio).all():
                raise ValueError(
                    "The audio contains invalid numerical samples."
                )

            self._calculate_spectrogram()
            self.draw_spectrogram()
            self._update_statistics()

            self._set_status("● Completed", "#22C55E")

            if callable(self.on_audio_loaded):
                self.on_audio_loaded(self.audio_path)

        except Exception as error:
            self._set_status("● Error", "#EF4444")
            messagebox.showerror(
                "Spectrogram Error",
                str(error),
            )

    def _calculate_spectrogram(self):
        maximum_frequency = min(
            8000,
            self.sample_rate // 2,
        )

        mel_power = librosa.feature.melspectrogram(
            y=self.audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmax=maximum_frequency,
            power=2.0,
        )

        self.mel_db = librosa.power_to_db(
            mel_power,
            ref=np.max,
        )

    # ======================================================
    # Drawing
    # ======================================================

    def draw_spectrogram(self):
        if self.mel_db is None:
            return

        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        self.figure.patch.set_facecolor("#111827")
        self.axes.set_facecolor("#111827")

        image = librosa.display.specshow(
            self.mel_db,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis="time",
            y_axis="mel",
            cmap=self.colormap_var.get(),
            ax=self.axes,
        )

        self.axes.set_title(
            "High-Resolution Mel Spectrogram",
            color="white",
            fontsize=15,
            pad=12,
        )
        self.axes.set_xlabel(
            "Time (seconds)",
            color="white",
        )
        self.axes.set_ylabel(
            "Mel Frequency (Hz)",
            color="white",
        )
        self.axes.tick_params(colors="white")

        for spine in self.axes.spines.values():
            spine.set_color("#475569")

        colorbar = self.figure.colorbar(
            image,
            ax=self.axes,
            pad=0.015,
            fraction=0.028,
        )
        colorbar.set_label(
            "Power (dB)",
            color="white",
        )
        colorbar.ax.tick_params(colors="white")

        self.figure.subplots_adjust(
            left=0.075,
            right=0.93,
            top=0.90,
            bottom=0.15,
        )

        self.canvas.draw_idle()

    def _draw_empty_plot(self):
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

        self.figure.patch.set_facecolor("#111827")
        self.axes.set_facecolor("#111827")

        self.axes.text(
            0.5,
            0.5,
            "Open an audio file to display its Mel spectrogram",
            transform=self.axes.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            color="#94A3B8",
            fontsize=14,
        )

        self.axes.set_xticks([])
        self.axes.set_yticks([])

        for spine in self.axes.spines.values():
            spine.set_color("#334155")

        self.canvas.draw_idle()

    def _on_colormap_changed(self, _selected_value=None):
        if self.mel_db is not None:
            self.draw_spectrogram()

    def reset_view(self):
        if self.mel_db is not None:
            self.draw_spectrogram()

    # ======================================================
    # Statistics
    # ======================================================

    def _update_statistics(self):
        duration = librosa.get_duration(
            y=self.audio,
            sr=self.sample_rate,
        )

        self.file_label.configure(
            text=Path(self.audio_path).name
        )
        self.duration_label.configure(
            text=f"{duration:.2f} sec"
        )
        self.sample_rate_label.configure(
            text=f"{self.sample_rate} Hz"
        )
        self.fft_label.configure(
            text=str(self.n_fft)
        )
        self.hop_label.configure(
            text=str(self.hop_length)
        )
        self.mels_label.configure(
            text=str(self.n_mels)
        )

    # ======================================================
    # Export
    # ======================================================

    def save_png(self):
        if self.mel_db is None:
            messagebox.showwarning(
                "No Audio",
                "Please load an audio file first.",
            )
            return

        default_name = (
            f"{Path(self.audio_path).stem}_spectrogram.png"
        )

        path = filedialog.asksaveasfilename(
            title="Save Spectrogram",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )

        if not path:
            return

        try:
            self.figure.savefig(
                path,
                dpi=300,
                bbox_inches="tight",
                facecolor=self.figure.get_facecolor(),
            )
            messagebox.showinfo(
                "Saved",
                f"Spectrogram saved successfully:\n\n{path}",
            )
        except Exception as error:
            messagebox.showerror(
                "Save Error",
                str(error),
            )

    def export_pdf(self):
        if self.mel_db is None:
            messagebox.showwarning(
                "No Audio",
                "Please load an audio file first.",
            )
            return

        default_name = (
            f"{Path(self.audio_path).stem}_spectrogram.pdf"
        )

        path = filedialog.asksaveasfilename(
            title="Export Spectrogram as PDF",
            initialfile=default_name,
            defaultextension=".pdf",
            filetypes=[("PDF document", "*.pdf")],
        )

        if not path:
            return

        try:
            self.figure.savefig(
                path,
                dpi=300,
                bbox_inches="tight",
                facecolor=self.figure.get_facecolor(),
            )
            messagebox.showinfo(
                "Exported",
                f"PDF exported successfully:\n\n{path}",
            )
        except Exception as error:
            messagebox.showerror(
                "Export Error",
                str(error),
            )

    # ======================================================
    # Page utilities
    # ======================================================

    def set_default_sample_rate(self, sample_rate):
        self.default_sample_rate = int(sample_rate)

        if self.audio_path:
            self.load_audio(self.audio_path)

    def set_colormap(self, colormap):
        if colormap in self.SUPPORTED_COLORMAPS:
            self.colormap_var.set(colormap)
            self._on_colormap_changed(colormap)

    def clear(self):
        self.audio_path = None
        self.audio = None
        self.sample_rate = None
        self.mel_db = None

        self.file_label.configure(text="--")
        self.duration_label.configure(text="--")
        self.sample_rate_label.configure(text="--")
        self.fft_label.configure(text=str(self.n_fft))
        self.hop_label.configure(text=str(self.hop_length))
        self.mels_label.configure(text=str(self.n_mels))

        self._set_status("● Ready", "#22C55E")
        self._draw_empty_plot()

    def _set_status(self, text, color):
        self.status_label.configure(
            text=text,
            text_color=color,
        )