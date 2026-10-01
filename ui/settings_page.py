# ==========================================================
# settings_page.py
# Application Settings Page
# ==========================================================

from tkinter import filedialog, messagebox

import customtkinter as ctk

from ui.theme import BACKGROUND, CARD


class SettingsPage(ctk.CTkFrame):
    """
    Application settings page.

    Parameters
    ----------
    parent:
        Parent CustomTkinter widget.

    on_settings_change:
        Optional callback called when settings are applied or reset.

        Example:
            def handle_settings(settings):
                print(settings)

    initial_settings:
        Optional dictionary containing initial settings.
    """

    DEFAULTS = {
        "appearance_mode": "Dark",
        "accent": "Blue",
        "sample_rate": 16000,
        "prediction_threshold": 60.0,
        "normalize_audio": False,
        "convert_to_mono": True,
        "auto_plot": True,
        "spectrogram_colormap": "magma",
        "export_folder": "",
    }

    ACCENT_COLORS = {
        "Blue": "#2563EB",
        "Green": "#16A34A",
        "Purple": "#7C3AED",
        "Orange": "#EA580C",
        "Cyan": "#0891B2",
    }

    def __init__(
        self,
        parent,
        on_settings_change=None,
        initial_settings=None,
    ):
        super().__init__(parent, fg_color=BACKGROUND)

        self.on_settings_change = on_settings_change

        self.settings = self.DEFAULTS.copy()
        if initial_settings:
            self.settings.update(initial_settings)

        self.appearance_var = ctk.StringVar(
            value=str(self.settings["appearance_mode"])
        )
        self.accent_var = ctk.StringVar(
            value=str(self.settings["accent"])
        )
        self.sample_rate_var = ctk.StringVar(
            value=str(self.settings["sample_rate"])
        )
        self.threshold_var = ctk.DoubleVar(
            value=float(self.settings["prediction_threshold"])
        )
        self.normalize_var = ctk.BooleanVar(
            value=bool(self.settings["normalize_audio"])
        )
        self.mono_var = ctk.BooleanVar(
            value=bool(self.settings["convert_to_mono"])
        )
        self.auto_plot_var = ctk.BooleanVar(
            value=bool(self.settings["auto_plot"])
        )
        self.colormap_var = ctk.StringVar(
            value=str(self.settings["spectrogram_colormap"])
        )
        self.export_folder_var = ctk.StringVar(
            value=str(self.settings["export_folder"])
        )

        self._build_page()
        self._update_threshold_label()

    # ======================================================
    # Page layout
    # ======================================================

    def _build_page(self):
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_appearance_card()
        self._build_audio_card()
        self._build_prediction_card()
        self._build_export_card()
        self._build_about_card()
        self._build_action_buttons()

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=20,
            pady=(20, 10),
        )
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="⚙ Application Settings",
            font=("Segoe UI", 28, "bold"),
        ).grid(row=0, column=0, sticky="w")

        self.status_label = ctk.CTkLabel(
            header,
            text="Settings ready",
            font=("Segoe UI", 13),
            text_color="#94A3B8",
        )
        self.status_label.grid(row=0, column=1, sticky="e")

    def _new_card(self, title, row, column):
        card = ctk.CTkFrame(
            self,
            fg_color=CARD,
            corner_radius=15,
        )
        card.grid(
            row=row,
            column=column,
            sticky="nsew",
            padx=10,
            pady=10,
        )
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card,
            text=title,
            font=("Segoe UI", 20, "bold"),
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=20,
            pady=(18, 12),
        )

        return card

    # ======================================================
    # Appearance settings
    # ======================================================

    def _build_appearance_card(self):
        card = self._new_card(
            "🎨 Appearance",
            row=1,
            column=0,
        )

        ctk.CTkLabel(
            card,
            text="Appearance mode",
            anchor="w",
            font=("Segoe UI", 14),
        ).grid(
            row=1,
            column=0,
            sticky="ew",
            padx=20,
            pady=(5, 6),
        )

        appearance_selector = ctk.CTkSegmentedButton(
            card,
            values=["Dark", "Light", "System"],
            variable=self.appearance_var,
            command=self._preview_appearance,
        )
        appearance_selector.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 16),
        )

        ctk.CTkLabel(
            card,
            text="Accent color",
            anchor="w",
            font=("Segoe UI", 14),
        ).grid(
            row=3,
            column=0,
            sticky="ew",
            padx=20,
            pady=(5, 6),
        )

        accent_menu = ctk.CTkOptionMenu(
            card,
            values=list(self.ACCENT_COLORS),
            variable=self.accent_var,
            command=self._preview_accent,
        )
        accent_menu.grid(
            row=4,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 12),
        )

        self.accent_preview = ctk.CTkFrame(
            card,
            height=8,
            fg_color=self._current_accent_color(),
            corner_radius=4,
        )
        self.accent_preview.grid(
            row=5,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 18),
        )

    # ======================================================
    # Audio settings
    # ======================================================

    def _build_audio_card(self):
        card = self._new_card(
            "🎵 Audio Processing",
            row=1,
            column=1,
        )

        ctk.CTkLabel(
            card,
            text="Visualization sample rate (plots only)",
            anchor="w",
            font=("Segoe UI", 14),
        ).grid(
            row=1,
            column=0,
            sticky="ew",
            padx=20,
            pady=(5, 6),
        )

        sample_rate_menu = ctk.CTkOptionMenu(
            card,
            values=["8000", "16000", "22050", "44100", "48000"],
            variable=self.sample_rate_var,
        )
        sample_rate_menu.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 12),
        )

        normalize_switch = ctk.CTkSwitch(
            card,
            text="Peak normalization: not applied during live inference",
            variable=self.normalize_var,
            state="disabled",
        )
        normalize_switch.grid(
            row=3,
            column=0,
            sticky="w",
            padx=20,
            pady=8,
        )

        mono_switch = ctk.CTkSwitch(
            card,
            text="Mono conversion: required by embedding models",
            variable=self.mono_var,
            state="disabled",
        )
        mono_switch.grid(
            row=4,
            column=0,
            sticky="w",
            padx=20,
            pady=8,
        )

        auto_plot_switch = ctk.CTkSwitch(
            card,
            text="Automatically update waveform and spectrogram",
            variable=self.auto_plot_var,
        )
        auto_plot_switch.grid(
            row=5,
            column=0,
            sticky="w",
            padx=20,
            pady=(8, 18),
        )

    # ======================================================
    # Prediction settings
    # ======================================================

    def _build_prediction_card(self):
        card = self._new_card(
            "🎯 Prediction",
            row=2,
            column=0,
        )

        threshold_header = ctk.CTkFrame(
            card,
            fg_color="transparent",
        )
        threshold_header.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=20,
            pady=(5, 0),
        )
        threshold_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            threshold_header,
            text="Fallback identity-confidence threshold",
            font=("Segoe UI", 14),
        ).grid(row=0, column=0, sticky="w")

        self.threshold_label = ctk.CTkLabel(
            threshold_header,
            text="60%",
            font=("Segoe UI", 14, "bold"),
            text_color="#22C55E",
        )
        self.threshold_label.grid(row=0, column=1, sticky="e")

        threshold_slider = ctk.CTkSlider(
            card,
            from_=0,
            to=100,
            number_of_steps=100,
            variable=self.threshold_var,
            command=self._on_threshold_changed,
        )
        threshold_slider.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=20,
            pady=(10, 16),
        )

        note = (
            "Used only if the selected model has no calibrated open-set "
            "detector. Normal unknown-speaker decisions use the saved "
            "model-specific open-set threshold."
        )

        ctk.CTkLabel(
            card,
            text=note,
            wraplength=450,
            justify="left",
            anchor="w",
            font=("Segoe UI", 12),
            text_color="#94A3B8",
        ).grid(
            row=3,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 12),
        )

        ctk.CTkLabel(
            card,
            text="Spectrogram colormap",
            anchor="w",
            font=("Segoe UI", 14),
        ).grid(
            row=4,
            column=0,
            sticky="ew",
            padx=20,
            pady=(5, 6),
        )

        colormap_menu = ctk.CTkOptionMenu(
            card,
            values=[
                "magma",
                "inferno",
                "plasma",
                "viridis",
                "cividis",
                "turbo",
            ],
            variable=self.colormap_var,
        )
        colormap_menu.grid(
            row=5,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 18),
        )

    # ======================================================
    # Export settings
    # ======================================================

    def _build_export_card(self):
        card = self._new_card(
            "📁 Export",
            row=2,
            column=1,
        )

        ctk.CTkLabel(
            card,
            text="Default export folder",
            anchor="w",
            font=("Segoe UI", 14),
        ).grid(
            row=1,
            column=0,
            sticky="ew",
            padx=20,
            pady=(5, 6),
        )

        folder_row = ctk.CTkFrame(
            card,
            fg_color="transparent",
        )
        folder_row.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 15),
        )
        folder_row.grid_columnconfigure(0, weight=1)

        self.folder_entry = ctk.CTkEntry(
            folder_row,
            textvariable=self.export_folder_var,
            placeholder_text="No default folder selected",
        )
        self.folder_entry.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 8),
        )

        browse_folder_button = ctk.CTkButton(
            folder_row,
            text="Browse",
            width=90,
            command=self._choose_export_folder,
        )
        browse_folder_button.grid(
            row=0,
            column=1,
        )

        ctk.CTkLabel(
            card,
            text=(
                "Waveform images, spectrograms and reports can "
                "use this folder as their default destination."
            ),
            wraplength=450,
            justify="left",
            anchor="w",
            font=("Segoe UI", 12),
            text_color="#94A3B8",
        ).grid(
            row=3,
            column=0,
            sticky="ew",
            padx=20,
            pady=(0, 18),
        )

    # ======================================================
    # About
    # ======================================================

    def _build_about_card(self):
        about = ctk.CTkFrame(
            self,
            fg_color=CARD,
            corner_radius=15,
        )
        about.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=10,
            pady=10,
        )
        about.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            about,
            text="ℹ About",
            font=("Segoe UI", 20, "bold"),
        ).grid(
            row=0,
            column=0,
            rowspan=3,
            sticky="nw",
            padx=20,
            pady=18,
        )

        ctk.CTkLabel(
            about,
            text="Speaker Identification System",
            anchor="w",
            font=("Segoe UI", 15, "bold"),
        ).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=15,
            pady=(18, 3),
        )

        ctk.CTkLabel(
            about,
            text="Application version 1.0",
            anchor="w",
            font=("Segoe UI", 12),
            text_color="#94A3B8",
        ).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=15,
            pady=(3, 18),
        )

    # ======================================================
    # Action buttons
    # ======================================================

    def _build_action_buttons(self):
        actions = ctk.CTkFrame(
            self,
            fg_color="transparent",
        )
        actions.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=10,
            pady=(5, 20),
        )

        actions.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkButton(
            actions,
            text="Apply Settings",
            height=44,
            command=self.apply_settings,
        ).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=5,
        )

        ctk.CTkButton(
            actions,
            text="Reset to Defaults",
            height=44,
            fg_color="#475569",
            hover_color="#334155",
            command=self.reset_to_defaults,
        ).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=5,
        )

        ctk.CTkButton(
            actions,
            text="Print Current Settings",
            height=44,
            fg_color="#334155",
            hover_color="#475569",
            command=self.print_settings,
        ).grid(
            row=0,
            column=2,
            sticky="ew",
            padx=5,
        )

    # ======================================================
    # Settings behavior
    # ======================================================

    def get_settings(self):
        """Return the current UI settings as a dictionary."""

        try:
            sample_rate = int(self.sample_rate_var.get())
        except (TypeError, ValueError):
            sample_rate = self.DEFAULTS["sample_rate"]

        return {
            "appearance_mode": self.appearance_var.get(),
            "accent": self.accent_var.get(),
            "accent_color": self._current_accent_color(),
            "sample_rate": sample_rate,
            "prediction_threshold": float(
                self.threshold_var.get()
            ),
            "normalize_audio": bool(self.normalize_var.get()),
            "convert_to_mono": bool(self.mono_var.get()),
            "auto_plot": bool(self.auto_plot_var.get()),
            "spectrogram_colormap": self.colormap_var.get(),
            "export_folder": self.export_folder_var.get().strip(),
        }

    def apply_settings(self, show_confirmation=True):
        try:
            settings = self.get_settings()
            self.settings.update(settings)

            ctk.set_appearance_mode(
                settings["appearance_mode"]
            )

            self.status_label.configure(
                text="Settings applied",
                text_color="#22C55E",
            )

            if callable(self.on_settings_change):
                self.on_settings_change(settings)

            if show_confirmation:
                messagebox.showinfo(
                    "Settings Applied",
                    "Application settings were updated successfully.",
                )

            return settings

        except Exception as error:
            self.status_label.configure(
                text="Unable to apply settings",
                text_color="#EF4444",
            )
            messagebox.showerror(
                "Settings Error",
                str(error),
            )
            return None

    def reset_to_defaults(self):
        confirmed = messagebox.askyesno(
            "Reset Settings",
            "Reset all application settings to their defaults?",
        )

        if not confirmed:
            return

        defaults = self.DEFAULTS.copy()

        self.appearance_var.set(defaults["appearance_mode"])
        self.accent_var.set(defaults["accent"])
        self.sample_rate_var.set(str(defaults["sample_rate"]))
        self.threshold_var.set(
            defaults["prediction_threshold"]
        )
        self.normalize_var.set(defaults["normalize_audio"])
        self.mono_var.set(defaults["convert_to_mono"])
        self.auto_plot_var.set(defaults["auto_plot"])
        self.colormap_var.set(
            defaults["spectrogram_colormap"]
        )
        self.export_folder_var.set(
            defaults["export_folder"]
        )

        self._update_threshold_label()
        self._preview_appearance(
            defaults["appearance_mode"]
        )
        self._preview_accent(defaults["accent"])

        settings = self.apply_settings(
            show_confirmation=False
        )

        if settings is not None:
            messagebox.showinfo(
                "Settings Reset",
                "Default settings were restored.",
            )

    def set_settings(self, settings):
        """
        Update the controls from an external settings dictionary.
        """

        if not settings:
            return

        merged = self.DEFAULTS.copy()
        merged.update(settings)

        self.appearance_var.set(merged["appearance_mode"])
        self.accent_var.set(merged["accent"])
        self.sample_rate_var.set(str(merged["sample_rate"]))
        self.threshold_var.set(
            float(merged["prediction_threshold"])
        )
        self.normalize_var.set(
            bool(merged["normalize_audio"])
        )
        self.mono_var.set(
            bool(merged["convert_to_mono"])
        )
        self.auto_plot_var.set(
            bool(merged["auto_plot"])
        )
        self.colormap_var.set(
            merged["spectrogram_colormap"]
        )
        self.export_folder_var.set(
            merged["export_folder"]
        )

        self._update_threshold_label()
        self._preview_accent(self.accent_var.get())

    # ======================================================
    # Control callbacks
    # ======================================================

    def _preview_appearance(self, selected_mode=None):
        mode = selected_mode or self.appearance_var.get()
        ctk.set_appearance_mode(mode)

    def _preview_accent(self, selected_accent=None):
        accent = selected_accent or self.accent_var.get()

        if accent not in self.ACCENT_COLORS:
            accent = self.DEFAULTS["accent"]
            self.accent_var.set(accent)

        self.accent_preview.configure(
            fg_color=self.ACCENT_COLORS[accent]
        )

        self.status_label.configure(
            text=(
                "Accent selected — apply settings to notify "
                "the dashboard"
            ),
            text_color="#94A3B8",
        )

    def _current_accent_color(self):
        return self.ACCENT_COLORS.get(
            self.accent_var.get(),
            self.ACCENT_COLORS["Blue"],
        )

    def _on_threshold_changed(self, _value=None):
        self._update_threshold_label()

    def _update_threshold_label(self):
        value = float(self.threshold_var.get())

        self.threshold_label.configure(
            text=f"{value:.0f}%"
        )

        if value >= 80:
            color = "#22C55E"
        elif value >= 50:
            color = "#F59E0B"
        else:
            color = "#EF4444"

        self.threshold_label.configure(
            text_color=color
        )

    def _choose_export_folder(self):
        folder = filedialog.askdirectory(
            title="Select Default Export Folder"
        )

        if folder:
            self.export_folder_var.set(folder)

    def print_settings(self):
        settings = self.get_settings()

        print("\nApplication Settings")
        print("=" * 40)

        for key, value in settings.items():
            print(f"{key}: {value}")

        self.status_label.configure(
            text="Settings printed to the terminal",
            text_color="#22C55E",
        )
