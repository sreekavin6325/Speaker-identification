# ============================================
# plots.py
# Waveform & Mel Spectrogram Widgets
# ============================================

import customtkinter as ctk
import librosa
import librosa.display
import matplotlib

# Use dark theme
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np


# -------------------------------------------------
# Common Plot Style
# -------------------------------------------------

plt.rcParams["figure.facecolor"] = "#111827"
plt.rcParams["axes.facecolor"] = "#111827"
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["text.color"] = "white"


# =================================================
# Waveform Widget
# =================================================

class WaveformPlot(ctk.CTkFrame):

    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        self.canvas = None

    def clear(self):
        for widget in self.winfo_children():
            widget.destroy()

    def draw(self, audio_path):

        self.clear()

        y, sr = librosa.load(audio_path, sr=16000)

        fig = plt.Figure(
            figsize=(10, 1.7),
            dpi=100,
            facecolor="#111827"
        )

        ax = fig.add_subplot(111)

        librosa.display.waveshow(
            y,
            sr=sr,
            color="#38BDF8",
            ax=ax
        )

        # Remove this because the card already says Waveform
        # ax.set_title("Audio Waveform")

        ax.grid(
            color="#334155",
            linestyle="--",
            alpha=0.4
        )

        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

        fig.subplots_adjust(
            left=0.075,
            right=0.985,
            top=0.95,
            bottom=0.30
        )

        self.canvas = FigureCanvasTkAgg(
            fig,
            master=self
        )

        self.canvas.draw()

        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=2,
            pady=2
        )


# =================================================
# Spectrogram Widget
# =================================================

class SpectrogramPlot:

    def __init__(self, parent):
        self.parent = parent
        self.canvas = None

    def clear(self):
        for widget in self.parent.winfo_children():
            widget.destroy()

    def draw(self, audio_path):

        self.clear()

        y, sr = librosa.load(audio_path, sr=16000)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128,
            fmax=8000
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        fig = plt.Figure(
            figsize=(10, 1.9),
            dpi=100,
            facecolor="#111827"
        )

        ax = fig.add_subplot(111)

        img = librosa.display.specshow(
            mel_db,
            sr=sr,
            x_axis="time",
            y_axis="mel",
            cmap="magma",
            ax=ax
        )

        # Remove duplicate title
        # ax.set_title("Mel Spectrogram")

        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Mel Frequency (Hz)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

        cbar = fig.colorbar(
            img,
            ax=ax,
            pad=0.01,
            fraction=0.025
        )

        cbar.ax.tick_params(
            colors="white",
            labelsize=8
        )

        fig.subplots_adjust(
            left=0.075,
            right=0.94,
            top=0.96,
            bottom=0.28
        )

        self.canvas = FigureCanvasTkAgg(
            fig,
            master=self.parent
        )

        self.canvas.draw()

        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=2,
            pady=2
        )