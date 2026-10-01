# ============================================
# meter.py
# Custom Animated Circular Confidence Meter
# ============================================

import customtkinter as ctk
import tkinter as tk


class ConfidenceMeter(ctk.CTkFrame):

    def __init__(
        self,
        master,
        size=220,
        thickness=18,
        max_value=100,
        fg_color="#1E293B",
        progress_color="#3B82F6",
        text_color="white",
        **kwargs
    ):

        super().__init__(master, fg_color=fg_color, **kwargs)

        self.size = size
        self.max_value = max_value
        self.value = 0
        self.target = 0
        self.thickness = thickness
        self.progress_color = progress_color
        self.text_color = text_color

        self.canvas = tk.Canvas(
            self,
            width=size,
            height=size,
            bg=fg_color,
            highlightthickness=0
        )

        self.canvas.pack(padx=10, pady=10)

        self.draw()

    # --------------------------------------

    def draw(self):

        self.canvas.delete("all")

        pad = self.thickness

        # Background Circle

        self.canvas.create_oval(
            pad,
            pad,
            self.size - pad,
            self.size - pad,
            outline="#334155",
            width=self.thickness
        )

        angle = (self.value / self.max_value) * 360

        # Progress Arc

        self.canvas.create_arc(
            pad,
            pad,
            self.size - pad,
            self.size - pad,
            start=90,
            extent=-angle,
            style="arc",
            outline=self.progress_color,
            width=self.thickness
        )

        # Percentage

        self.canvas.create_text(
            self.size / 2,
            self.size / 2 - 10,
            text=f"{int(self.value)}%",
            fill=self.text_color,
            font=("Segoe UI", 26, "bold")
        )

        # Caption

        self.canvas.create_text(
            self.size / 2,
            self.size / 2 + 28,
            text="Confidence",
            fill="#94A3B8",
            font=("Segoe UI", 13)
        )

    # --------------------------------------

    def animate(self):

        if self.value < self.target:

            self.value += 1

            self.draw()

            self.after(12, self.animate)

        else:

            self.value = self.target

            self.draw()

    # --------------------------------------

    def set_value(self, value):

        if value < 0:
            value = 0

        if value > self.max_value:
            value = self.max_value

        self.target = value

        self.value = 0

        self.animate()

    # --------------------------------------

    def reset(self):

        self.value = 0
        self.target = 0
        self.draw()