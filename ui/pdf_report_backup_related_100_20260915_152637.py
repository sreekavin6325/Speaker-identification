from datetime import datetime
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BLUE = colors.HexColor("#2563EB")
DARK_BLUE = colors.HexColor("#0F172A")
SLATE = colors.HexColor("#334155")
MUTED = colors.HexColor("#64748B")
LIGHT_SLATE = colors.HexColor("#E2E8F0")
PALE_BLUE = colors.HexColor("#EFF6FF")
GREEN = colors.HexColor("#16A34A")
RED = colors.HexColor("#DC2626")
WHITE = colors.white


def _clean_text(value):
    """Return safe text that renders correctly with PDF base fonts."""

    text = str(value)
    text = text.replace("â€”", "-")
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    return escape(text)


def _page_header_footer(canvas, document):
    canvas.saveState()
    width, height = A4

    canvas.setFillColor(DARK_BLUE)
    canvas.rect(
        0,
        height - 1.3 * cm,
        width,
        1.3 * cm,
        fill=1,
        stroke=0,
    )

    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(
        1.4 * cm,
        height - 0.82 * cm,
        "Speaker Identification System",
    )

    canvas.setFillColor(SLATE)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(
        1.4 * cm,
        0.72 * cm,
        "Multi-Model Speaker Identification Report",
    )
    canvas.drawRightString(
        width - 1.4 * cm,
        0.72 * cm,
        f"Page {document.page}",
    )
    canvas.restoreState()


def _create_waveform_image(audio, sample_rate, output_path):
    duration = len(audio) / float(sample_rate)
    time_axis = np.linspace(
        0.0,
        duration,
        len(audio),
        endpoint=False,
    )

    figure, axes = plt.subplots(
        figsize=(11, 3.2),
        dpi=160,
    )
    figure.patch.set_facecolor("#111827")
    axes.set_facecolor("#111827")

    axes.plot(
        time_axis,
        audio,
        color="#38BDF8",
        linewidth=0.7,
    )
    axes.fill_between(
        time_axis,
        audio,
        0,
        color="#38BDF8",
        alpha=0.14,
    )
    axes.set_title(
        "Audio Waveform",
        color="white",
        fontsize=14,
        pad=10,
    )
    axes.set_xlabel("Time (seconds)", color="white")
    axes.set_ylabel("Amplitude", color="white")
    axes.tick_params(colors="white", labelsize=8)
    axes.grid(
        color="#475569",
        linestyle="--",
        alpha=0.35,
    )

    for spine in axes.spines.values():
        spine.set_color("#64748B")

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def _create_spectrogram_image(
    audio,
    sample_rate,
    output_path,
    colormap="magma",
):
    n_fft = 1024
    hop_length = 256
    n_mels = 128

    if audio.size < n_fft:
        audio = np.pad(
            audio,
            (0, n_fft - audio.size),
            mode="constant",
        )

    window = np.hanning(n_fft).astype(np.float32)
    frame_count = 1 + (audio.size - n_fft) // hop_length
    power_spectrum = np.empty(
        (n_fft // 2 + 1, frame_count),
        dtype=np.float32,
    )

    # Calculate frames one at a time. This avoids allocating one very
    # large strided matrix for long recordings.
    for frame_index in range(frame_count):
        start = frame_index * hop_length
        frame = audio[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame, n=n_fft)
        power_spectrum[:, frame_index] = (
            np.abs(spectrum).astype(np.float32) ** 2
        )

    minimum_frequency = 0.0
    maximum_frequency = float(min(8000, sample_rate // 2))

    def hz_to_mel(frequency):
        return 2595.0 * np.log10(1.0 + frequency / 700.0)

    def mel_to_hz(mel_value):
        return 700.0 * (10.0 ** (mel_value / 2595.0) - 1.0)

    mel_points = np.linspace(
        hz_to_mel(minimum_frequency),
        hz_to_mel(maximum_frequency),
        n_mels + 2,
    )
    frequency_points = mel_to_hz(mel_points)
    fft_frequencies = np.fft.rfftfreq(
        n_fft,
        d=1.0 / sample_rate,
    )
    filter_bank = np.zeros(
        (n_mels, fft_frequencies.size),
        dtype=np.float32,
    )

    for mel_index in range(n_mels):
        lower = frequency_points[mel_index]
        centre = frequency_points[mel_index + 1]
        upper = frequency_points[mel_index + 2]

        rising = (fft_frequencies - lower) / max(
            centre - lower,
            np.finfo(np.float32).eps,
        )
        falling = (upper - fft_frequencies) / max(
            upper - centre,
            np.finfo(np.float32).eps,
        )
        filter_bank[mel_index] = np.maximum(
            0.0,
            np.minimum(rising, falling),
        )

    mel_power = filter_bank @ power_spectrum
    mel_db = 10.0 * np.log10(
        np.maximum(mel_power, np.finfo(np.float32).eps)
    )
    mel_db -= float(np.max(mel_db))

    figure, axes = plt.subplots(
        figsize=(11, 3.7),
        dpi=160,
    )
    figure.patch.set_facecolor("#111827")
    axes.set_facecolor("#111827")

    frame_times = (
        np.arange(frame_count, dtype=np.float32)
        * hop_length
        / sample_rate
    )
    mel_centres_hz = frequency_points[1:-1]

    spectrogram_image = axes.pcolormesh(
        frame_times,
        mel_centres_hz,
        mel_db,
        shading="auto",
        cmap=colormap,
    )
    axes.set_title(
        "Mel Spectrogram",
        color="white",
        fontsize=14,
        pad=10,
    )
    axes.set_xlabel("Time (seconds)", color="white")
    axes.set_ylabel("Mel Frequency (Hz)", color="white")
    axes.tick_params(colors="white", labelsize=8)

    for spine in axes.spines.values():
        spine.set_color("#64748B")

    colorbar = figure.colorbar(
        spectrogram_image,
        ax=axes,
        pad=0.015,
        fraction=0.03,
    )
    colorbar.set_label("Power (dB)", color="white")
    colorbar.ax.tick_params(colors="white", labelsize=8)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def _normalise_probabilities(probabilities):
    cleaned = []

    for speaker, probability in probabilities or []:
        try:
            numeric_probability = float(probability)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(numeric_probability):
            continue

        cleaned.append(
            (_clean_text(speaker), max(0.0, numeric_probability))
        )

    if not cleaned:
        return []

    total = sum(item[1] for item in cleaned)

    # Accept both 0-1 and 0-100 input formats, then make the
    # displayed distribution add up to exactly 100 percent.
    if total <= 1.01:
        cleaned = [
            (speaker, probability * 100.0)
            for speaker, probability in cleaned
        ]
        total *= 100.0

    if total > 0 and abs(total - 100.0) > 0.05:
        cleaned = [
            (speaker, probability / total * 100.0)
            for speaker, probability in cleaned
        ]

    cleaned.sort(
        key=lambda item: item[1],
        reverse=True,
    )
    return cleaned


def _make_styles():
    styles = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=DARK_BLUE,
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=25,
            spaceAfter=7,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            textColor=MUTED,
            fontSize=9.5,
            leading=13,
            spaceAfter=14,
        ),
        "section": ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            textColor=DARK_BLUE,
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=14,
            spaceBefore=5,
            spaceAfter=4,
        ),
        "model_title": ParagraphStyle(
            "ModelTitle",
            parent=styles["Heading1"],
            alignment=TA_CENTER,
            textColor=DARK_BLUE,
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=20,
            spaceAfter=7,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=styles["BodyText"],
            textColor=SLATE,
            fontSize=9,
            leading=12,
        ),
        "small": ParagraphStyle(
            "SmallText",
            parent=styles["BodyText"],
            textColor=MUTED,
            fontSize=8,
            leading=10,
        ),
        "table": ParagraphStyle(
            "TableText",
            parent=styles["BodyText"],
            textColor=SLATE,
            fontSize=8.0,
            leading=9.0,
        ),
        "table_header": ParagraphStyle(
            "TableHeaderText",
            parent=styles["BodyText"],
            textColor=WHITE,
            fontName="Helvetica-Bold",
            fontSize=8.0,
            leading=9.0,
        ),
    }


def _cell(value, style):
    return Paragraph(_clean_text(value), style)


def _styled_key_value_table(rows, styles):
    table_rows = [
        [
            _cell("Field", styles["table_header"]),
            _cell("Value", styles["table_header"]),
        ]
    ]

    for field, value in rows:
        table_rows.append([
            _cell(field, styles["table"]),
            _cell(value, styles["table"]),
        ])

    table = Table(
        table_rows,
        colWidths=[5.0 * cm, 11.8 * cm],
        repeatRows=1,
        hAlign="CENTER",
    )
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("BACKGROUND", (0, 1), (0, -1), LIGHT_SLATE),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.45, SLATE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ])
    )
    return table


def _probability_table(probabilities, styles):
    table_rows = [[
        _cell("Rank", styles["table_header"]),
        _cell("Speaker", styles["table_header"]),
        _cell("Probability", styles["table_header"]),
    ]]

    for rank, (speaker, probability) in enumerate(
        _normalise_probabilities(probabilities),
        start=1,
    ):
        table_rows.append([
            _cell(rank, styles["table"]),
            Paragraph(speaker, styles["table"]),
            _cell(f"{probability:.2f}%", styles["table"]),
        ])

    if len(table_rows) == 1:
        table_rows.append([
            _cell("-", styles["table"]),
            _cell("No probability data", styles["table"]),
            _cell("-", styles["table"]),
        ])

    table = Table(
        table_rows,
        colWidths=[
            2.0 * cm,
            10.5 * cm,
            4.3 * cm,
        ],
        repeatRows=1,
        hAlign="CENTER",
    )
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("GRID", (0, 0), (-1, -1), 0.45, SLATE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 1), (0, -1), "CENTER"),
            ("ALIGN", (2, 1), (2, -1), "RIGHT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE_BLUE]),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    return table


def _related_audio_table(related_audio_files, styles):
    """Build a compact numbered table of related audio filenames."""

    table_rows = [[
        _cell("No.", styles["table_header"]),
        _cell("Audio filename", styles["table_header"]),
    ]]

    cleaned_names = []
    for audio_file in related_audio_files or []:
        filename = Path(str(audio_file)).name.strip()
        if filename and filename not in cleaned_names:
            cleaned_names.append(filename)

    for index, filename in enumerate(cleaned_names[:5], start=1):
        table_rows.append([
            _cell(index, styles["table"]),
            _cell(filename, styles["table"]),
        ])

    if len(table_rows) == 1:
        table_rows.append([
            _cell("-", styles["table"]),
            _cell("No related audio files available", styles["table"]),
        ])

    table = Table(
        table_rows,
        colWidths=[1.5 * cm, 15.3 * cm],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("GRID", (0, 0), (-1, -1), 0.45, LIGHT_SLATE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 1), (0, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE_BLUE]),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ])
    )
    return table


def _comparison_table(model_results, styles):
    rows = [[
        _cell("Model", styles["table_header"]),
        _cell("Predicted Speaker", styles["table_header"]),
        _cell("Confidence", styles["table_header"]),
        _cell("Inference", styles["table_header"]),
    ]]

    for result in model_results:
        if result.get("error"):
            speaker = "Prediction failed"
            confidence = "-"
            inference = "-"
        else:
            speaker = result.get("speaker", "-")
            confidence = f"{float(result.get('confidence', 0.0)):.2f}%"
            inference = f"{float(result.get('inference_time', 0.0)):.2f} ms"

        rows.append([
            _cell(result.get("display_name", result.get("model_id", "-")), styles["table"]),
            _cell(speaker, styles["table"]),
            _cell(confidence, styles["table"]),
            _cell(inference, styles["table"]),
        ])

    table = Table(
        rows,
        colWidths=[
            6.0 * cm,
            5.0 * cm,
            2.8 * cm,
            3.0 * cm,
        ],
        repeatRows=1,
        hAlign="CENTER",
    )
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("GRID", (0, 0), (-1, -1), 0.45, SLATE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE_BLUE]),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ])
    )
    return table


def export_complete_report(
    output_path,
    audio_path,
    model_results,
    colormap="magma",
):
    """
    Export one aligned PDF containing:

    - report summary and audio details;
    - one separate page for every trained model;
    - waveform and spectrogram together on the final page.
    """

    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file was not found: {audio_path}"
        )

    if not model_results:
        raise ValueError(
            "No model prediction results were supplied."
        )

    audio_info = sf.info(str(audio_path))
    audio, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True,
    )
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.mean(audio, axis=1).reshape(-1)

    if audio.size == 0:
        raise ValueError("The selected audio file is empty.")

    duration = float(audio_info.duration)
    file_size_kb = audio_path.stat().st_size / 1024.0
    peak = float(np.max(np.abs(audio)))
    rms = float(
        np.sqrt(
            np.mean(
                np.square(audio, dtype=np.float64)
            )
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory)
        waveform_path = temporary_directory / "waveform.png"
        spectrogram_path = temporary_directory / "spectrogram.png"

        _create_waveform_image(
            audio,
            sample_rate,
            waveform_path,
        )
        _create_spectrogram_image(
            audio,
            sample_rate,
            spectrogram_path,
            colormap,
        )

        document = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=1.4 * cm,
            leftMargin=1.4 * cm,
            topMargin=1.75 * cm,
            bottomMargin=1.35 * cm,
            title="Multi-Model Speaker Identification Report",
            author="Speaker Identification System",
        )
        styles = _make_styles()
        story = []

        # --------------------------------------------------
        # Page 1: report overview and shared audio details
        # --------------------------------------------------
        story.append(
            Paragraph(
                "Multi-Model Speaker Identification Report",
                styles["title"],
            )
        )
        story.append(
            Paragraph(
                (
                    f"Generated {datetime.now():%d-%m-%Y %I:%M:%S %p} "
                    f"- {len(model_results)} trained models evaluated"
                ),
                styles["subtitle"],
            )
        )
        story.append(
            Paragraph("Audio Details", styles["section"])
        )
        story.append(
            _styled_key_value_table(
                [
                    ("Filename", audio_path.name),
                    ("Duration", f"{duration:.3f} seconds"),
                    ("Sample Rate", f"{audio_info.samplerate} Hz"),
                    ("Channels", audio_info.channels),
                    ("Bit Depth / Subtype", audio_info.subtype),
                    ("Format", audio_info.format),
                    ("File Size", f"{file_size_kb:.2f} KB"),
                    ("Peak Amplitude", f"{peak:.6f}"),
                    ("RMS Energy", f"{rms:.6f}"),
                ],
                styles,
            )
        )
        story.append(Spacer(1, 0.25 * cm))
        story.append(
            Paragraph(
                "Model Comparison Summary",
                styles["section"],
            )
        )
        story.append(
            _comparison_table(model_results, styles)
        )

        # --------------------------------------------------
        # One separate page for every model
        # --------------------------------------------------
        for model_index, result in enumerate(model_results, start=1):
            story.append(PageBreak())

            display_name = result.get(
                "display_name",
                result.get("model_id", f"Model {model_index}"),
            )
            story.append(
                Paragraph(
                    f"Model {model_index}: {_clean_text(display_name)}",
                    styles["model_title"],
                )
            )

            metadata_rows = [
                ("Model ID", result.get("model_id", "-")),
                ("Provider", result.get("provider", "-")),
                ("Architecture", result.get("architecture", "-")),
                ("Pretrained Source", result.get("pretrained_name", "-")),
                (
                    "Embedding Dimension",
                    result.get(
                        "embedding_dim",
                        result.get("embedding_dimension", "-"),
                    ),
                ),
                (
                    "Classifier",
                    result.get("classifier_name", "Not specified"),
                ),
                (
                    "Open-set Detector",
                    result.get(
                        "open_set_detector_model_name",
                        result.get("display_name", "Not specified"),
                    ),
                ),
            ]
            story.append(
                Paragraph(
                    "Model Details",
                    styles["section"],
                )
            )
            story.append(
                _styled_key_value_table(
                    metadata_rows,
                    styles,
                )
            )

            if result.get("error"):
                story.append(Spacer(1, 0.35 * cm))
                story.append(
                    Paragraph(
                        "Prediction Error",
                        styles["section"],
                    )
                )
                error_table = _styled_key_value_table(
                    [
                        ("Status", "Prediction failed"),
                        ("Reason", result["error"]),
                    ],
                    styles,
                )
                error_table.setStyle(
                    TableStyle([
                        ("TEXTCOLOR", (1, 1), (1, -1), RED),
                    ])
                )
                story.append(error_table)
                continue

            story.append(Spacer(1, 0.3 * cm))
            story.append(
                Paragraph(
                    "Prediction Details",
                    styles["section"],
                )
            )
            story.append(
                _styled_key_value_table(
                    [
                        ("Predicted Speaker", result.get("speaker", "-")),
                        (
                            "Displayed Decision Probability",
                            f"{float(result.get('confidence', 0.0)):.2f}%",
                        ),
                        (
                            "Closed-set Identity Confidence",
                            f"{float(result.get('closed_set_confidence', 0.0)):.2f}%",
                        ),
                        (
                            "Known-speaker Probability",
                            (
                                "Not available"
                                if result.get("known_speaker_probability") is None
                                else (
                                    f"{float(result['known_speaker_probability']) * 100.0:.2f}%"
                                )
                            ),
                        ),
                        (
                            "Unknown-speaker Probability",
                            (
                                "Not available"
                                if result.get("unknown_speaker_probability") is None
                                else (
                                    f"{float(result['unknown_speaker_probability']) * 100.0:.2f}%"
                                )
                            ),
                        ),
                        (
                            "Open-set Decision Threshold",
                            (
                                "Not available"
                                if result.get("open_set_threshold") is None
                                else (
                                    f"{float(result['open_set_threshold']) * 100.0:.2f}% known score"
                                )
                            ),
                        ),
                        (
                            "Inference Time",
                            f"{float(result.get('inference_time', 0.0)):.2f} ms",
                        ),
                    ],
                    styles,
                )
            )
            story.append(Spacer(1, 0.25 * cm))
            story.append(
                Paragraph(
                    "Speaker Probability Distribution",
                    styles["section"],
                )
            )
            story.append(
                _probability_table(
                    result.get("probabilities", []),
                    styles,
                )
            )
            story.append(Spacer(1, 0.25 * cm))
            story.append(
                Paragraph(
                    "Related Audio Files",
                    styles["section"],
                )
            )
            story.append(
                _related_audio_table(
                    result.get("related_audio_files", []),
                    styles,
                )
            )

        # --------------------------------------------------
        # Final page: both visualisations on one page
        # --------------------------------------------------
        story.append(PageBreak())
        story.append(
            Paragraph(
                "Audio Signal Visualisations",
                styles["model_title"],
            )
        )

        waveform_block = [
            Paragraph("Waveform", styles["section"]),
            Image(
                str(waveform_path),
                width=17.0 * cm,
                height=4.95 * cm,
            ),
            Spacer(1, 0.08 * cm),
            Paragraph(
                (
                    "Signal amplitude over time. "
                    f"Peak: {peak:.6f}; RMS: {rms:.6f}."
                ),
                styles["small"],
            ),
        ]
        story.append(KeepTogether(waveform_block))
        story.append(Spacer(1, 0.18 * cm))

        spectrogram_block = [
            Paragraph("Mel Spectrogram", styles["section"]),
            Image(
                str(spectrogram_path),
                width=17.0 * cm,
                height=5.75 * cm,
            ),
            Spacer(1, 0.08 * cm),
            Paragraph(
                (
                    "Acoustic energy across time and "
                    "Mel-scaled frequency. "
                    f"Colormap: {_clean_text(colormap)}."
                ),
                styles["small"],
            ),
        ]
        story.append(KeepTogether(spectrogram_block))

        document.build(
            story,
            onFirstPage=_page_header_footer,
            onLaterPages=_page_header_footer,
        )

    return str(output_path)
