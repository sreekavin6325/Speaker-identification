# ============================================
# cards.py
# Modern Dashboard Cards
# ============================================

import customtkinter as ctk
from ui.theme import *


# ======================================================
# Base Card
# ======================================================

class DashboardCard(ctk.CTkFrame):

    def __init__(self, parent, title="", icon="", **kwargs):

        super().__init__(
            parent,
            fg_color=CARD,
            corner_radius=CARD_RADIUS,
            border_width=1,
            border_color=BORDER,
            **kwargs
        )

        self.grid_propagate(False)

        self.header = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )
        self.header.pack(fill="x", padx=15, pady=(15, 8))

        self.icon = ctk.CTkLabel(
            self.header,
            text=icon,
            font=("Segoe UI Emoji", 22)
        )
        self.icon.pack(side="left")

        self.title = ctk.CTkLabel(
            self.header,
            text=title,
            font=CARD_TITLE
        )
        self.title.pack(side="left", padx=10)

        self.body = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )
        self.body.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=(0,15)
        )


# ======================================================
# Upload Card
# ======================================================

class UploadCard(DashboardCard):

    def __init__(self, parent, command):

        super().__init__(
            parent,
            title="Upload Audio",
            icon="📂"
        )

        self.filename = ctk.CTkLabel(
            self.body,
            text="No audio selected",
            font=BODY_FONT,
            text_color=TEXT_SECONDARY
        )
        self.filename.pack(pady=(15,20))

        self.button = ctk.CTkButton(
            self.body,
            text="Browse WAV",
            command=command,
            height=45,
            font=BUTTON_FONT,
            fg_color=BUTTON_COLOR,
            hover_color=BUTTON_HOVER,
            corner_radius=BUTTON_RADIUS
        )

        self.button.pack(fill="x")

        self.note = ctk.CTkLabel(
            self.body,
            text="Supported Format : WAV (16kHz)",
            font=SMALL_FONT,
            text_color=TEXT_SECONDARY
        )

        self.note.pack(pady=15)


# ======================================================
# Audio Information Card
# ======================================================

class AudioInfoCard(DashboardCard):

    def __init__(self, parent):

        super().__init__(
            parent,
            title="Audio Information",
            icon="🎵"
        )

        self.labels = {}

        fields = [
            "File",
            "Duration",
            "Sample Rate",
            "Channels",
            "Bit Depth",
            "File Size"
        ]

        for field in fields:

            row = ctk.CTkFrame(
                self.body,
                fg_color="transparent"
            )
            row.pack(fill="x", pady=4)

            left = ctk.CTkLabel(
                row,
                text=f"{field}:",
                width=120,
                anchor="w",
                font=BODY_FONT
            )

            left.pack(side="left")

            value = ctk.CTkLabel(
                row,
                text="--",
                anchor="w",
                font=BODY_FONT,
                text_color=SUCCESS
            )

            value.pack(side="left")

            self.labels[field] = value

    def update_info(
        self,
        filename,
        duration,
        sample_rate,
        channels,
        bitdepth,
        filesize
    ):

        self.labels["File"].configure(text=filename)
        self.labels["Duration"].configure(text=f"{duration:.2f} sec")
        self.labels["Sample Rate"].configure(text=f"{sample_rate} Hz")
        self.labels["Channels"].configure(text=str(channels))
        self.labels["Bit Depth"].configure(text=str(bitdepth))
        self.labels["File Size"].configure(text=f"{filesize:.2f} KB")


# ======================================================
# Prediction Card
# ======================================================

class PredictionCard(DashboardCard):

    def __init__(self, parent):

        super().__init__(
            parent,
            title="Prediction",
            icon="🎯"
        )

        self.speaker = ctk.CTkLabel(
            self.body,
            text="--",
            font=("Segoe UI",24,"bold"),
            text_color=SUCCESS
        )

        self.speaker.pack(pady=(5,15))

        self.time = ctk.CTkLabel(
            self.body,
            text="Inference : -- ms",
            font=BODY_FONT
        )

        self.time.pack(pady=10)

        self.status = ctk.CTkLabel(
            self.body,
            text="Ready",
            font=BODY_FONT,
            text_color=SUCCESS
        )

        self.status.pack()
        self.model_label = ctk.CTkLabel(
            self.body,
            text="Model: --",
            font=("Segoe UI", 13),
            text_color=TEXT_SECONDARY
        )

        self.model_label.pack(
            pady=(5, 8)
        )

    def update_prediction(
        self,
        speaker,
        inference
    ):

        self.speaker.configure(
            text=speaker
        )

        self.time.configure(
            text=f"Inference : {inference:.2f} ms"
        )

        self.status.configure(
            text="Completed"
        )


# ======================================================
# Plot Card
# ======================================================

class PlotCard(DashboardCard):

    def __init__(self, parent, title, icon):

        super().__init__(
            parent,
            title=title,
            icon=icon
        )

        self.canvas = ctk.CTkFrame(
            self.body,
            fg_color="#111827",
            corner_radius=12
        )

        self.canvas.pack(
            fill="both",
            expand=True
        )
