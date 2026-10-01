# ============================================
# sidebar.py
# Sidebar Navigation
# ============================================

import customtkinter as ctk

from ui.theme import (
    SIDEBAR,
    CARD,
    TEXT,
    TEXT_SECONDARY,
    BORDER,
    ACCENT,
)


class Sidebar(ctk.CTkFrame):
    """
    Sidebar navigation for the Speaker Identification System.

    Parameters
    ----------
    parent:
        Parent CustomTkinter widget.

    page_callback:
        Function called when a navigation button is clicked.
        Example: page_callback("prediction")
    """

    def __init__(self, parent, page_callback):

        super().__init__(
            parent,
            width=250,
            corner_radius=0,
            fg_color=SIDEBAR
        )

        self.page_callback = page_callback
        self.buttons = {}
        self.active_page = None

        # Prevent sidebar width from shrinking.
        self.grid_propagate(False)

        self.build_header()
        self.build_navigation()
        self.build_footer()

        # Default selected page.
        self.set_active_button("dashboard")

    # =================================================
    # Header
    # =================================================

    def build_header(self):

        title = ctk.CTkLabel(
            self,
            text="Speaker\nIdentification",
            justify="left",
            anchor="w",
            font=("Segoe UI", 28, "bold"),
            text_color=TEXT
        )

        title.pack(
            fill="x",
            padx=25,
            pady=(35, 8)
        )

        subtitle = ctk.CTkLabel(
            self,
            text="Deep Learning Dashboard",
            anchor="w",
            font=("Segoe UI", 13),
            text_color=TEXT_SECONDARY
        )

        subtitle.pack(
            fill="x",
            padx=25
        )

        separator = ctk.CTkFrame(
            self,
            height=2,
            corner_radius=0,
            fg_color=BORDER
        )

        separator.pack(
            fill="x",
            padx=20,
            pady=(25, 20)
        )

    # =================================================
    # Navigation
    # =================================================

    def build_navigation(self):

        navigation_label = ctk.CTkLabel(
            self,
            text="NAVIGATION",
            anchor="w",
            font=("Segoe UI", 11, "bold"),
            text_color=TEXT_SECONDARY
        )

        navigation_label.pack(
            fill="x",
            padx=25,
            pady=(0, 8)
        )

        menu_items = [
            ("dashboard", "🏠  Dashboard"),
            ("research", "📊  Research Results"),
            ("waveform", "📈  Waveform"),
            ("spectrogram", "🌈  Spectrogram"),
            ("settings", "⚙  Settings"),
        ]

        for page_name, button_text in menu_items:

            button = ctk.CTkButton(
                self,
                text=button_text,
                anchor="w",
                height=44,
                corner_radius=8,
                border_width=0,
                fg_color="transparent",
                hover_color=CARD,
                text_color=TEXT,
                font=("Segoe UI", 14),
                command=lambda name=page_name: self.open_page(name)
            )

            button.pack(
                fill="x",
                padx=15,
                pady=4
            )

            self.buttons[page_name] = button

        action_separator = ctk.CTkFrame(
            self,
            height=1,
            corner_radius=0,
            fg_color=BORDER
        )

        action_separator.pack(
            fill="x",
            padx=20,
            pady=(16, 10)
        )

        self.export_button = ctk.CTkButton(
            self,
            text="📄  Export Complete PDF",
            anchor="w",
            height=44,
            corner_radius=8,
            border_width=1,
            border_color=ACCENT,
            fg_color="transparent",
            hover_color=CARD,
            text_color=TEXT,
            font=("Segoe UI", 14, "bold"),
            command=lambda: self.open_page("export_pdf")
        )

        self.export_button.pack(
            fill="x",
            padx=15,
            pady=4
        )

    # =================================================
    # Footer
    # =================================================

    def build_footer(self):

        footer = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        footer.pack(
            side="bottom",
            fill="x",
            padx=20,
            pady=20
        )

        separator = ctk.CTkFrame(
            footer,
            height=2,
            corner_radius=0,
            fg_color=BORDER
        )

        separator.pack(
            fill="x",
            pady=(0, 15)
        )

        version_label = ctk.CTkLabel(
            footer,
            text="Speaker ID System v1.0",
            font=("Segoe UI", 11),
            text_color=TEXT_SECONDARY
        )

        version_label.pack(anchor="w")

        model_label = ctk.CTkLabel(
            footer,
            text="4 Embedding Models + Classifiers",
            font=("Segoe UI", 11),
            text_color=TEXT_SECONDARY
        )

        model_label.pack(
            anchor="w",
            pady=(3, 0)
        )

    # =================================================
    # Navigation Actions
    # =================================================

    def open_page(self, page_name):

        # Export is an action, not another page. Keep the
        # currently selected navigation button active.
        if page_name == "export_pdf":
            if callable(self.page_callback):
                self.page_callback(page_name)
            return

        self.set_active_button(page_name)

        if callable(self.page_callback):
            self.page_callback(page_name)

    def set_active_button(self, page_name):

        self.active_page = page_name

        for name, button in self.buttons.items():

            if name == page_name:

                button.configure(
                    fg_color=ACCENT,
                    hover_color=ACCENT,
                    text_color="white"
                )

            else:

                button.configure(
                    fg_color="transparent",
                    hover_color=CARD,
                    text_color=TEXT
                )
