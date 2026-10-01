# ==========================================================
# waveform_page.py
# Large Waveform Viewer
# Part 1
# ==========================================================

import os

import librosa
import numpy as np

import customtkinter as ctk
from tkinter import filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)

from ui.theme import *


class WaveformPage(ctk.CTkFrame):

    def __init__(self, parent):

        super().__init__(
            parent,
            fg_color=BACKGROUND
        )

        self.audio = None
        self.sr = None
        self.audio_path = None

        self.build_page()

    # --------------------------------------------------
    # Build UI
    # --------------------------------------------------

    def build_page(self):

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        ###################################################
        # Title
        ###################################################

        title = ctk.CTkLabel(
            self,
            text="📈 Waveform Viewer",
            font=("Segoe UI", 28, "bold")
        )

        title.grid(
            row=0,
            column=0,
            sticky="w",
            padx=20,
            pady=(20,10)
        )
        self.build_buttons()
        ###################################################
        # Main Card
        ###################################################

        self.card = ctk.CTkFrame(
            self,
            fg_color=CARD,
            corner_radius=15
        )

        self.card.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=15,
            pady=10
        )

        self.card.grid_rowconfigure(0, weight=1)
        self.card.grid_columnconfigure(0, weight=1)

        ###################################################
        # Figure
        ###################################################

        self.figure = Figure(
            figsize=(12,5),
            dpi=100
        )

        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=self.card
        )

        self.canvas.get_tk_widget().grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=10,
            pady=10
        )

        ###################################################
        # Toolbar
        ###################################################

        toolbar_frame = ctk.CTkFrame(
            self.card,
            fg_color="transparent"
        )

        toolbar_frame.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=10,
            pady=(0,10)
        )

        self.toolbar = NavigationToolbar2Tk(
            self.canvas,
            toolbar_frame
        )

        self.toolbar.update()

        ###################################################
        # Statistics
        ###################################################

        self.stats = ctk.CTkFrame(
            self,
            fg_color=CARD,
            corner_radius=12
        )

        self.stats.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=15,
            pady=(0,20)
        )

        self.duration_label = ctk.CTkLabel(
            self.stats,
            text="Duration : --"
        )

        self.duration_label.pack(
            side="left",
            padx=20,
            pady=15
        )

        self.samplerate_label = ctk.CTkLabel(
            self.stats,
            text="Sample Rate : --"
        )

        self.samplerate_label.pack(
            side="left",
            padx=20
        )

        self.peak_label = ctk.CTkLabel(
            self.stats,
            text="Peak : --"
        )

        self.peak_label.pack(
            side="left",
            padx=20
        )

        self.rms_label = ctk.CTkLabel(
            self.stats,
            text="RMS : --"
        )

        self.rms_label.pack(
            side="left",
            padx=20
        )
# ==========================================================
# Load Audio
# ==========================================================
    def load_audio(self, audio_path):
            try:
                self.audio_path = audio_path

                self.audio, self.sr = librosa.load(
                    audio_path,
                    sr=16000,
                    mono=True
                )

                self.draw_waveform()
                self.update_statistics()

            except Exception as error:
                messagebox.showerror(
                    "Waveform Error",
                    str(error)
                )


# ==========================================================
# Draw Waveform
# ==========================================================

    def draw_waveform(self):

        self.ax.clear()

        self.ax.set_facecolor("#111827")

        self.figure.patch.set_facecolor("#111827")

        duration = len(self.audio) / self.sr

        time = np.linspace(
            0,
            duration,
            len(self.audio)
        )

        self.ax.plot(
            time,
            self.audio,
            linewidth=0.8,
            color="#38BDF8"
        )

        self.ax.set_title(
            "Audio Waveform",
            color="white",
            fontsize=14
        )

        self.ax.set_xlabel(
            "Time (seconds)",
            color="white"
        )

        self.ax.set_ylabel(
            "Amplitude",
            color="white"
        )

        self.ax.grid(
            alpha=0.3
        )

        self.ax.tick_params(
            colors="white"
        )

        for spine in self.ax.spines.values():

            spine.set_color("white")

        self.canvas.draw()


# ==========================================================
# Update Statistics
# ==========================================================

    def update_statistics(self):

        duration = librosa.get_duration(
            y=self.audio,
            sr=self.sr
        )

        peak = np.max(
            np.abs(self.audio)
        )

        rms = np.sqrt(
            np.mean(self.audio ** 2)
        )

        self.duration_label.configure(
            text=f"Duration : {duration:.2f} sec"
        )

        self.samplerate_label.configure(
            text=f"Sample Rate : {self.sr} Hz"
        )

        self.peak_label.configure(
            text=f"Peak : {peak:.3f}"
        )

        self.rms_label.configure(
            text=f"RMS : {rms:.3f}"
        )

# ==========================================================
# Bottom Buttons
# ==========================================================

    def build_buttons(self):

        self.button_frame = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        self.button_frame.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=15,
            pady=(0,20)
        )

        self.button_frame.grid_columnconfigure(
            (0,1,2,3),
            weight=1
        )

        ctk.CTkButton(
            self.button_frame,
            text="📂 Open Audio",
            command=self.open_audio
        ).grid(row=0,column=0,padx=5,sticky="ew")

        ctk.CTkButton(
            self.button_frame,
            text="💾 Save PNG",
            command=self.save_png
        ).grid(row=0,column=1,padx=5,sticky="ew")

        ctk.CTkButton(
            self.button_frame,
            text="🖨 Print",
            command=self.print_waveform
        ).grid(row=0,column=2,padx=5,sticky="ew")

        ctk.CTkButton(
            self.button_frame,
            text="🗑 Clear",
            fg_color="#475569",
            hover_color="#334155",
            command=self.clear
        ).grid(row=0,column=3,padx=5,sticky="ew")


# ==========================================================
# Browse Audio
# ==========================================================

    def open_audio(self):

        path = filedialog.askopenfilename(

            title="Select Audio",

            filetypes=[
                ("Wave Files","*.wav"),
                ("All Files","*.*")
            ]
        )

        if path:

            self.load_audio(path)


# ==========================================================
# Save PNG
# ==========================================================

    def save_png(self):

        if self.audio is None:

            messagebox.showwarning(
                "No Audio",
                "Please load an audio file first."
            )

            return

        filename = filedialog.asksaveasfilename(

            defaultextension=".png",

            filetypes=[
                ("PNG Image","*.png")
            ]
        )

        if filename:

            self.figure.savefig(
                filename,
                dpi=300,
                bbox_inches="tight"
            )

            messagebox.showinfo(
                "Saved",
                "Waveform image saved successfully."
            )


# ==========================================================
# Print
# ==========================================================

    def print_waveform(self):

        if self.audio is None:

            messagebox.showwarning(
                "No Audio",
                "Please load an audio file first."
            )

            return

        filename = filedialog.asksaveasfilename(

            defaultextension=".pdf",

            filetypes=[
                ("PDF","*.pdf")
            ]
        )

        if filename:

            self.figure.savefig(
                filename,
                dpi=300
            )

            messagebox.showinfo(
                "Exported",
                "Waveform exported successfully."
            )


# ==========================================================
# Improved Clear
# ==========================================================

    def clear(self):

        self.audio = None
        self.sr = None
        self.audio_path = None

        self.ax.clear()

        self.ax.set_title("No Audio Loaded")

        self.canvas.draw()

        self.duration_label.configure(
            text="Duration : --"
        )

        self.samplerate_label.configure(
            text="Sample Rate : --"
        )

        self.peak_label.configure(
            text="Peak : --"
        )

        self.rms_label.configure(
            text="RMS : --"
        )
# ==========================================================
# Clear Page
# ==========================================================

    def clear(self):

        self.ax.clear()

        self.canvas.draw()

        self.duration_label.configure(
            text="Duration : --"
        )

        self.samplerate_label.configure(
            text="Sample Rate : --"
        )

        self.peak_label.configure(
            text="Peak : --"
        )

        self.rms_label.configure(
            text="RMS : --"
        )