"""Extract condition-specific embeddings for robustness experiments.

The strict chapter-held-out protocol determines the ordered clip set.  Each
audio file is transformed in memory, written to a short-lived WAV file, and
passed through the project's existing pre-trained embedding extractor.  No
derived audio is retained.  Outputs are checkpointed and remain separate from
both application artifacts and clean research embeddings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from model_config import MODEL_CONFIGS, get_model_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "manifests" / "librispeech_dev_clean_manifest.csv"
)
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "manifests"
    / "librispeech_dev_clean_chapter_heldout_2fold_protocol.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "research_results" / "robustness_embeddings"

CONDITIONS: dict[str, dict[str, Any]] = {
    "short_0p5s": {
        "family": "short_duration",
        "duration_seconds": 0.5,
        "crop": "center",
    },
    "short_1s": {
        "family": "short_duration",
        "duration_seconds": 1.0,
        "crop": "center",
    },
    "short_2s": {
        "family": "short_duration",
        "duration_seconds": 2.0,
        "crop": "center",
    },
    "short_3s": {
        "family": "short_duration",
        "duration_seconds": 3.0,
        "crop": "center",
    },
    "short_5s": {
        "family": "short_duration",
        "duration_seconds": 5.0,
        "crop": "center",
    },
    "short_10s": {
        "family": "short_duration",
        "duration_seconds": 10.0,
        "crop": "center",
    },
    "noise_white_20db": {
        "family": "additive_white_noise",
        "snr_db": 20.0,
    },
    "noise_white_10db": {
        "family": "additive_white_noise",
        "snr_db": 10.0,
    },
    "noise_white_0db": {
        "family": "additive_white_noise",
        "snr_db": 0.0,
    },
    # Controlled environmental-noise proxies.  These are generated
    # deterministically and mixed at the same 10 dB SNR, making model
    # comparisons repeatable without licensing an external noise corpus.
    "noise_traffic_10db": {
        "family": "additive_environmental_noise",
        "noise_type": "traffic",
        "snr_db": 10.0,
    },
    "noise_office_10db": {
        "family": "additive_environmental_noise",
        "noise_type": "office",
        "snr_db": 10.0,
    },
    "noise_cafe_10db": {
        "family": "additive_environmental_noise",
        "noise_type": "cafe",
        "snr_db": 10.0,
    },
    "noise_rain_10db": {
        "family": "additive_environmental_noise",
        "noise_type": "rain",
        "snr_db": 10.0,
    },
    "noise_fan_10db": {
        "family": "additive_environmental_noise",
        "noise_type": "fan",
        "snr_db": 10.0,
    },
    "noise_street_10db": {
        "family": "additive_environmental_noise",
        "noise_type": "street",
        "snr_db": 10.0,
    },
    # Sample-rate ablation.  The transformed file is stored at the stated
    # rate; each pre-trained extractor then performs its normal 16 kHz input
    # conversion.  The 16 kHz condition is the explicit resampling baseline.
    "sample_rate_8k": {
        "family": "sample_rate_ablation",
        "target_sample_rate": 8000,
    },
    "sample_rate_16k": {
        "family": "sample_rate_ablation",
        "target_sample_rate": 16000,
    },
    "sample_rate_22k05": {
        "family": "sample_rate_ablation",
        "target_sample_rate": 22050,
    },
}

PROTOCOL_COLUMNS = {"clip_id", "speaker_label", "source_group", "fold"}
MANIFEST_COLUMNS = {
    "clip_id",
    "speaker_id",
    "speaker_label",
    "chapter_id",
    "source_group",
    "absolute_path",
    "fold",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract aligned speaker embeddings for deterministic short-"
            "duration and additive-noise robustness conditions."
        )
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", choices=list(MODEL_CONFIGS))
    model_group.add_argument("--all-models", action="store_true")

    condition_group = parser.add_mutually_exclusive_group(required=True)
    condition_group.add_argument("--condition", choices=list(CONDITIONS))
    condition_group.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITIONS),
        help="Extract only the listed conditions.",
    )
    condition_group.add_argument("--all-conditions", action="store_true")

    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-file", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def safe_tag(value: str) -> str:
    value = value.strip()
    if not value or any(character in value for character in r'\/:*?"<>|'):
        raise ValueError("--tag is empty or contains an invalid character.")
    return value


def safe_protocol_name(path: Path, protocol_hash: str) -> str:
    stem = path.stem.lower()
    if stem.endswith("_protocol"):
        stem = stem[: -len("_protocol")]
    stem = re.sub(r"[^a-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-") or "external_protocol"
    return f"{stem}__{protocol_hash[:12]}"


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file was not found:\n{path}")
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"CSV is missing columns ({path.name}): " + ", ".join(missing)
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def load_protocol_and_manifest(
    protocol_path: Path,
    manifest_path: Path,
    limit: int | None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    protocol_path = protocol_path.resolve()
    manifest_path = manifest_path.resolve()
    protocol_rows = read_csv(protocol_path, PROTOCOL_COLUMNS)
    manifest_rows = read_csv(manifest_path, MANIFEST_COLUMNS)
    manifest_by_clip = {row["clip_id"].strip(): row for row in manifest_rows}
    if len(manifest_by_clip) != len(manifest_rows):
        raise ValueError("Manifest contains duplicate clip IDs.")

    seen: set[str] = set()
    group_to_fold: dict[str, int] = {}
    rows: list[dict[str, str]] = []
    assignment_digest = hashlib.sha256()

    for item in protocol_rows:
        clip_id = item["clip_id"].strip()
        if clip_id in seen:
            raise ValueError(f"Protocol contains duplicate clip ID: {clip_id}")
        seen.add(clip_id)
        if clip_id not in manifest_by_clip:
            raise ValueError(f"Protocol clip is absent from manifest: {clip_id}")

        source = manifest_by_clip[clip_id]
        speaker_label = item["speaker_label"].strip()
        source_group = item["source_group"].strip()
        fold = int(item["fold"])
        if speaker_label != source["speaker_label"].strip():
            raise ValueError(f"Speaker mismatch for {clip_id}.")
        if source_group != source["source_group"].strip():
            raise ValueError(f"Source-group mismatch for {clip_id}.")
        if fold < 1:
            raise ValueError(f"Invalid fold for {clip_id}: {fold}")
        previous_fold = group_to_fold.setdefault(source_group, fold)
        if previous_fold != fold:
            raise ValueError(f"Source group spans folds: {source_group}")

        audio_path = Path(source["absolute_path"])
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file was not found:\n{audio_path}")

        row = dict(source)
        row["fold"] = str(fold)
        rows.append(row)
        assignment_digest.update(
            (
                f"{clip_id}\t{speaker_label}\t{source_group}\t{fold}\n"
            ).encode("utf-8")
        )

    fold_values = sorted({int(row["fold"]) for row in rows})
    speakers = sorted({row["speaker_label"] for row in rows})
    for fold in fold_values:
        train = [row for row in rows if int(row["fold"]) != fold]
        test = [row for row in rows if int(row["fold"]) == fold]
        train_speakers = {row["speaker_label"] for row in train}
        test_speakers = {row["speaker_label"] for row in test}
        train_groups = {row["source_group"] for row in train}
        test_groups = {row["source_group"] for row in test}
        if test_speakers - train_speakers:
            raise ValueError(f"Fold {fold} contains test-only speakers.")
        if train_groups & test_groups:
            raise ValueError(f"Fold {fold} contains source-group leakage.")

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least one.")
        rows = rows[:limit]

    protocol_hash = file_sha256(protocol_path)
    metadata = {
        "path": str(protocol_path),
        "sha256": protocol_hash,
        "protocol_id": safe_protocol_name(protocol_path, protocol_hash),
        "ordered_assignment_sha256": assignment_digest.hexdigest(),
        "full_row_count": len(protocol_rows),
        "selected_row_count": len(rows),
        "fold_values": fold_values,
        "speaker_count": len(speakers),
        "source_group_count": len(group_to_fold),
        "source_group_disjoint": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }
    return rows, metadata


def deterministic_rng(seed: int, clip_id: str, condition: str) -> np.random.Generator:
    digest = hashlib.sha256(
        f"{seed}|{condition}|{clip_id}".encode("utf-8")
    ).digest()
    derived_seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(derived_seed)


def center_crop_or_pad(
    waveform: np.ndarray,
    target_samples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if target_samples < 1:
        raise ValueError("Target duration produced no samples.")
    original_samples = int(waveform.size)
    if original_samples >= target_samples:
        start = (original_samples - target_samples) // 2
        transformed = waveform[start : start + target_samples]
        padded_samples = 0
    else:
        left = (target_samples - original_samples) // 2
        right = target_samples - original_samples - left
        transformed = np.pad(waveform, (left, right), mode="constant")
        start = 0
        padded_samples = left + right
    return transformed.astype(np.float32), {
        "original_samples": original_samples,
        "output_samples": int(transformed.size),
        "crop_start_sample": int(start),
        "padded_samples": int(padded_samples),
    }


def add_white_noise(
    waveform: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    signal_power = float(np.mean(np.square(waveform, dtype=np.float64)))
    if not math.isfinite(signal_power) or signal_power <= 1e-12:
        raise ValueError("Audio power is too small for an SNR transformation.")
    noise = rng.standard_normal(waveform.size).astype(np.float32)
    noise_power = float(np.mean(np.square(noise, dtype=np.float64)))
    target_noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise *= np.float32(math.sqrt(target_noise_power / noise_power))
    mixture = waveform.astype(np.float32, copy=False) + noise
    peak_before_scaling = float(np.max(np.abs(mixture)))
    scale = 1.0
    if peak_before_scaling > 0.99:
        scale = 0.99 / peak_before_scaling
        mixture = mixture * np.float32(scale)

    residual = mixture.astype(np.float64) - waveform.astype(np.float64) * scale
    scaled_signal = waveform.astype(np.float64) * scale
    measured_snr = 10.0 * math.log10(
        float(np.mean(scaled_signal**2)) / float(np.mean(residual**2))
    )
    return mixture.astype(np.float32), {
        "signal_power": signal_power,
        "target_snr_db": float(snr_db),
        "measured_snr_db": float(measured_snr),
        "mixture_scale": float(scale),
        "peak_before_scaling": peak_before_scaling,
    }


def _fft_colored_noise(
    size: int,
    sample_rate: int,
    rng: np.random.Generator,
    spectral_exponent: float,
) -> np.ndarray:
    """Return deterministic unit-RMS coloured noise using FFT shaping."""

    frequencies = np.fft.rfftfreq(size, d=1.0 / sample_rate)
    spectrum = rng.standard_normal(frequencies.size) + 1j * rng.standard_normal(
        frequencies.size
    )
    safe_frequency = np.maximum(frequencies, max(1.0, sample_rate / size))
    spectrum *= safe_frequency ** (spectral_exponent / 2.0)
    spectrum[0] = 0.0
    noise = np.fft.irfft(spectrum, n=size).astype(np.float32)
    rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
    if rms <= 1e-12:
        raise ValueError("Generated environmental noise has zero power.")
    return noise / np.float32(rms)


def generate_environmental_noise(
    noise_type: str,
    size: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a controlled proxy for a named real-world noise family."""

    time_axis = np.arange(size, dtype=np.float64) / float(sample_rate)

    def consume_coloured_noise_draws() -> None:
        # _fft_colored_noise draws two vectors of rfft size.  Consuming those
        # values without performing an unused FFT preserves bit-for-bit RNG
        # compatibility with the original implementation and saved checkpoints.
        frequency_bins = size // 2 + 1
        rng.standard_normal(frequency_bins)
        rng.standard_normal(frequency_bins)

    if noise_type == "traffic":
        rng.standard_normal(size)  # unused white-noise draw
        pink = _fft_colored_noise(size, sample_rate, rng, -1.0)
        brown = _fft_colored_noise(size, sample_rate, rng, -2.0)
        consume_coloured_noise_draws()  # unused blue-noise draw
        noise = 0.72 * brown + 0.20 * pink
        noise += 0.20 * np.sin(2 * np.pi * 55.0 * time_axis)
        noise += 0.10 * np.sin(2 * np.pi * 110.0 * time_axis)
    elif noise_type == "office":
        white = rng.standard_normal(size).astype(np.float32)
        pink = _fft_colored_noise(size, sample_rate, rng, -1.0)
        consume_coloured_noise_draws()  # unused brown-noise draw
        consume_coloured_noise_draws()  # unused blue-noise draw
        noise = 0.38 * pink + 0.22 * white
        noise += 0.30 * np.sin(2 * np.pi * 50.0 * time_axis)
        impulse_count = max(1, int(size / sample_rate * 3.0))
        locations = rng.integers(0, size, size=impulse_count)
        impulses = np.zeros(size, dtype=np.float32)
        impulses[locations] = rng.uniform(2.0, 4.0, size=impulse_count)
        kernel = np.exp(-np.arange(max(8, sample_rate // 250)) / 7.0)
        noise += 0.30 * np.convolve(impulses, kernel, mode="same")
    elif noise_type == "cafe":
        # Speech-like, amplitude-modulated coloured bands approximate babble.
        rng.standard_normal(size)  # unused white-noise draw
        consume_coloured_noise_draws()  # unused pink-noise draw
        consume_coloured_noise_draws()  # unused brown-noise draw
        consume_coloured_noise_draws()  # unused blue-noise draw
        noise = np.zeros(size, dtype=np.float64)
        for voice_index in range(6):
            carrier = _fft_colored_noise(size, sample_rate, rng, -0.6)
            rate = rng.uniform(1.5, 4.5)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            envelope = 0.35 + 0.65 * np.square(
                np.sin(2.0 * np.pi * rate * time_axis + phase)
            )
            noise += carrier * envelope / (voice_index + 2) ** 0.25
        noise = noise.astype(np.float32)
    elif noise_type == "rain":
        white = rng.standard_normal(size).astype(np.float32)
        consume_coloured_noise_draws()  # unused pink-noise draw
        consume_coloured_noise_draws()  # unused brown-noise draw
        blue = _fft_colored_noise(size, sample_rate, rng, 1.0)
        noise = 0.62 * blue + 0.20 * white
        drop_count = max(1, int(size / sample_rate * 25.0))
        locations = rng.integers(0, size, size=drop_count)
        drops = np.zeros(size, dtype=np.float32)
        drops[locations] = rng.uniform(0.5, 2.5, size=drop_count)
        noise += 0.24 * drops
    elif noise_type == "fan":
        rng.standard_normal(size)  # unused white-noise draw
        pink = _fft_colored_noise(size, sample_rate, rng, -1.0)
        consume_coloured_noise_draws()  # unused brown-noise draw
        consume_coloured_noise_draws()  # unused blue-noise draw
        fundamental = rng.uniform(70.0, 115.0)
        modulation = 0.70 + 0.30 * np.sin(2 * np.pi * 0.35 * time_axis)
        noise = 0.25 * pink
        for harmonic in range(1, 7):
            noise += (
                modulation
                * np.sin(2 * np.pi * fundamental * harmonic * time_axis)
                / harmonic
            )
    elif noise_type == "street":
        white = rng.standard_normal(size).astype(np.float32)
        pink = _fft_colored_noise(size, sample_rate, rng, -1.0)
        brown = _fft_colored_noise(size, sample_rate, rng, -2.0)
        consume_coloured_noise_draws()  # unused blue-noise draw
        noise = 0.45 * brown + 0.25 * pink + 0.12 * white
        noise += 0.15 * np.sin(2 * np.pi * 65.0 * time_axis)
        burst_count = max(1, int(size / sample_rate * 1.2))
        for _ in range(burst_count):
            start = int(rng.integers(0, max(1, size - sample_rate // 8)))
            length = min(sample_rate // 8, size - start)
            if length > 0:
                window = np.hanning(length)
                noise[start : start + length] += (
                    rng.uniform(0.8, 1.8) * window
                )
    else:
        raise ValueError(f"Unsupported environmental noise type: {noise_type}")

    noise = np.asarray(noise, dtype=np.float32)
    noise -= np.mean(noise, dtype=np.float64)
    rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
    if not math.isfinite(rms) or rms <= 1e-12:
        raise ValueError(f"Generated {noise_type} noise has invalid power.")
    return noise / np.float32(rms)


def add_noise_at_snr(
    waveform: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    signal_power = float(np.mean(waveform.astype(np.float64) ** 2))
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if signal_power <= 1e-12 or noise_power <= 1e-12:
        raise ValueError("Signal or noise power is too small for SNR mixing.")
    target_noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    scaled_noise = noise * np.float32(math.sqrt(target_noise_power / noise_power))
    mixture = waveform.astype(np.float32, copy=False) + scaled_noise
    peak = float(np.max(np.abs(mixture)))
    scale = 1.0 if peak <= 0.99 else 0.99 / peak
    mixture *= np.float32(scale)
    measured = 10.0 * math.log10(
        float(np.mean((waveform.astype(np.float64) * scale) ** 2))
        / float(np.mean((scaled_noise.astype(np.float64) * scale) ** 2))
    )
    return mixture.astype(np.float32), {
        "target_snr_db": float(snr_db),
        "measured_snr_db": float(measured),
        "mixture_scale": float(scale),
        "peak_before_scaling": peak,
    }


def resample_waveform(
    waveform: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from math import gcd
    from scipy.signal import resample_poly

    common = gcd(int(source_rate), int(target_rate))
    transformed = resample_poly(
        waveform,
        int(target_rate) // common,
        int(source_rate) // common,
    ).astype(np.float32)
    return transformed, {
        "source_sample_rate": int(source_rate),
        "target_sample_rate": int(target_rate),
        "resampling_method": "scipy.signal.resample_poly",
        "output_sample_rate": int(target_rate),
    }


def transform_audio(
    waveform: np.ndarray,
    sample_rate: int,
    condition_name: str,
    clip_id: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    condition = CONDITIONS[condition_name]
    family = condition["family"]
    if family == "short_duration":
        duration = float(condition["duration_seconds"])
        transformed, details = center_crop_or_pad(
            waveform, int(round(duration * sample_rate))
        )
    elif family == "additive_white_noise":
        transformed, details = add_white_noise(
            waveform,
            float(condition["snr_db"]),
            deterministic_rng(seed, clip_id, condition_name),
        )
    elif family == "additive_environmental_noise":
        rng = deterministic_rng(seed, clip_id, condition_name)
        noise_type = str(condition["noise_type"])
        noise = generate_environmental_noise(
            noise_type,
            waveform.size,
            sample_rate,
            rng,
        )
        transformed, details = add_noise_at_snr(
            waveform,
            noise,
            float(condition["snr_db"]),
        )
        details["noise_type"] = noise_type
        details["noise_source"] = "deterministic controlled synthetic proxy"
    elif family == "sample_rate_ablation":
        transformed, details = resample_waveform(
            waveform,
            sample_rate,
            int(condition["target_sample_rate"]),
        )
    else:
        raise ValueError(f"Unsupported condition family: {family}")

    if transformed.ndim != 1 or transformed.size == 0:
        raise ValueError("Transformed waveform is empty or not mono.")
    if not np.isfinite(transformed).all():
        raise ValueError("Transformed waveform contains invalid values.")
    details.update(
        {
            "condition": condition_name,
            "family": family,
            "sample_rate": int(details.get("output_sample_rate", sample_rate)),
            "output_duration_seconds": float(
                transformed.size / int(details.get("output_sample_rate", sample_rate))
            ),
        }
    )
    return transformed, details


def stack_embeddings(
    embeddings: list[np.ndarray], dimension: int | None
) -> np.ndarray:
    if embeddings:
        return np.stack(embeddings).astype(np.float32)
    return np.empty((0, int(dimension or 0)), dtype=np.float32)


def prepare_output(path: Path, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be combined.")
    if resume:
        if not (path / "checkpoint_state.json").is_file():
            raise FileNotFoundError(f"Checkpoint was not found:\n{path}")
        return
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output is not empty:\n{path}\nUse --resume or --overwrite."
        )
    path.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    path: Path,
    state: dict[str, Any],
    embeddings: list[np.ndarray],
    labels: list[str],
    clip_ids: list[str],
    folds: list[int],
    records: list[dict[str, Any]],
) -> None:
    atomic_npy(
        path / "checkpoint_X.npy",
        stack_embeddings(embeddings, state.get("embedding_dimension")),
    )
    atomic_npy(path / "checkpoint_y.npy", np.asarray(labels, dtype=str))
    atomic_npy(path / "checkpoint_clip_ids.npy", np.asarray(clip_ids, dtype=str))
    atomic_npy(path / "checkpoint_folds.npy", np.asarray(folds, dtype=np.int16))
    atomic_json(path / "checkpoint_records.json", records)
    state["processed_rows"] = len(records)
    state["successful_rows"] = len(embeddings)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    atomic_json(path / "checkpoint_state.json", state)


def load_checkpoint(
    path: Path,
    model_id: str,
    condition_name: str,
    protocol_hash: str,
    seed: int,
) -> tuple[
    dict[str, Any],
    list[np.ndarray],
    list[str],
    list[str],
    list[int],
    list[dict[str, Any]],
]:
    with (path / "checkpoint_state.json").open(
        "r", encoding="utf-8"
    ) as state_file:
        state = json.load(state_file)
    expected = {
        "model_id": model_id,
        "condition": condition_name,
        "protocol_sha256": protocol_hash,
        "seed": seed,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"Checkpoint {key} does not match this run.")
    X = np.load(path / "checkpoint_X.npy")
    labels = np.load(path / "checkpoint_y.npy").astype(str).tolist()
    clip_ids = np.load(path / "checkpoint_clip_ids.npy").astype(str).tolist()
    folds = np.load(path / "checkpoint_folds.npy").astype(int).tolist()
    with (path / "checkpoint_records.json").open(
        "r", encoding="utf-8"
    ) as records_file:
        records = json.load(records_file)
    if len({len(X), len(labels), len(clip_ids), len(folds)}) != 1:
        raise RuntimeError("Checkpoint arrays have inconsistent lengths.")
    return (
        state,
        [row.astype(np.float32) for row in X],
        labels,
        clip_ids,
        folds,
        records,
    )


def progress_rows(rows: list[dict[str, str]], description: str):
    indexed = list(enumerate(rows))
    try:
        from tqdm import tqdm
    except ImportError:
        return indexed
    return tqdm(indexed, desc=description, unit="file")


def timing_summary(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {key: 0.0 for key in ("minimum", "median", "mean", "p95", "maximum", "total")}
    ordered = sorted(finite)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "minimum": min(ordered),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[p95_index],
        "maximum": max(ordered),
        "total": sum(ordered),
    }


def write_records(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "manifest_index",
        "clip_id",
        "speaker_label",
        "source_group",
        "fold",
        "status",
        "embedding_index",
        "extraction_time_ms",
        "output_duration_seconds",
        "measured_snr_db",
        "padded_samples",
        "embedding_l2_norm",
        "error_type",
        "error_message",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def finalize(
    path: Path,
    model_id: str,
    condition_name: str,
    tag: str,
    seed: int,
    protocol: dict[str, Any],
    embeddings: list[np.ndarray],
    labels: list[str],
    clip_ids: list[str],
    folds: list[int],
    records: list[dict[str, Any]],
    wall_seconds: float,
) -> dict[str, Any]:
    if not embeddings:
        raise RuntimeError("All rows failed embedding extraction.")
    X = stack_embeddings(embeddings, embeddings[0].size)
    y = np.asarray(labels, dtype=str)
    ids = np.asarray(clip_ids, dtype=str)
    fold_array = np.asarray(folds, dtype=np.int16)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise RuntimeError("Final embedding matrix is invalid.")
    if len({len(X), len(y), len(ids), len(fold_array)}) != 1:
        raise RuntimeError("Final arrays have inconsistent lengths.")
    if len(set(ids.tolist())) != len(ids):
        raise RuntimeError("Final clip IDs are not unique.")

    artifacts = {
        "X.npy": X,
        "y.npy": y,
        "clip_ids.npy": ids,
        "folds.npy": fold_array,
    }
    for filename, value in artifacts.items():
        atomic_npy(path / filename, value)
    write_records(records, path / "records.csv")
    failures = [record for record in records if record["status"] == "failed"]
    success_times = [
        float(record["extraction_time_ms"])
        for record in records
        if record["status"] == "success"
    ]
    config = get_model_config(model_id)
    metadata = {
        "schema_version": 1,
        "experiment": "robustness_embedding_extraction",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": model_id,
        "model": {
            "display_name": config["display_name"],
            "provider": config["provider"],
            "architecture": config["architecture"],
            "pretrained_name": config.get("pretrained_name") or config.get("source"),
        },
        "condition": condition_name,
        "condition_definition": CONDITIONS[condition_name],
        "seed": seed,
        "tag": tag,
        "protocol": protocol,
        "requested_rows": len(records),
        "successful_rows": len(embeddings),
        "failed_rows": len(failures),
        "coverage_percent": 100.0 * len(embeddings) / len(records),
        "embedding_dimension": int(X.shape[1]),
        "number_of_speakers": len(set(labels)),
        "number_of_folds": len(set(folds)),
        "timing_ms": timing_summary(success_times),
        "wall_seconds": float(wall_seconds),
        "failure_types": dict(
            sorted(Counter(record["error_type"] for record in failures).items())
        ),
        "artifact_sha256": {
            filename: file_sha256(path / filename) for filename in artifacts
        },
        "derived_audio_retained": False,
        "output_directory": str(path),
    }
    atomic_json(path / "metadata.json", metadata)
    return metadata


def extract_condition(
    model_id: str,
    condition_name: str,
    rows: list[dict[str, str]],
    protocol: dict[str, Any],
    output_root: Path,
    tag: str,
    seed: int,
    checkpoint_every: int,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least one.")
    output_directory = (
        output_root.resolve()
        / str(protocol["protocol_id"])
        / model_id
        / condition_name
        / tag
    )
    prepare_output(output_directory, overwrite, resume)

    if resume:
        state, embeddings, labels, clip_ids, folds, records = load_checkpoint(
            output_directory,
            model_id,
            condition_name,
            str(protocol["sha256"]),
            seed,
        )
        start_index = int(state["next_row_index"])
        dimension = (
            int(state["embedding_dimension"])
            if state.get("embedding_dimension") is not None
            else None
        )
    else:
        embeddings: list[np.ndarray] = []
        labels: list[str] = []
        clip_ids: list[str] = []
        folds: list[int] = []
        records: list[dict[str, Any]] = []
        start_index = 0
        dimension = None
        state = {
            "schema_version": 1,
            "model_id": model_id,
            "condition": condition_name,
            "protocol_sha256": protocol["sha256"],
            "seed": seed,
            "next_row_index": 0,
            "embedding_dimension": None,
        }

    config = get_model_config(model_id)
    print()
    print("=" * 72)
    print("ROBUSTNESS EMBEDDING EXTRACTION")
    print("=" * 72)
    print(f"Model       : {config['display_name']}")
    print(f"Condition   : {condition_name}")
    print(f"Rows        : {len(rows)}")
    print(f"Resume from : {start_index}")
    print(f"Output      : {output_directory}")
    print("=" * 72)

    try:
        from embedding_extractors import (
            clear_model_cache,
            extract_embedding,
            load_audio,
        )
    except Exception as error:
        raise RuntimeError(
            "Unable to import the project embedding extractors."
        ) from error

    sample_rate = int(config.get("sample_rate", 16000))
    wall_start = time.perf_counter()
    remaining = rows[start_index:]
    try:
        with tempfile.TemporaryDirectory(prefix="speaker_robustness_") as temp_name:
            temporary_audio = Path(temp_name) / "condition.wav"
            for offset, row in progress_rows(
                remaining, f"{model_id}/{condition_name}"
            ):
                manifest_index = start_index + offset
                started = time.perf_counter()
                embedding_index: int | str = ""
                embedding_norm: float | str = ""
                output_duration: float | str = ""
                measured_snr: float | str = ""
                padded_samples: int | str = ""
                error_type = ""
                error_message = ""
                try:
                    waveform = load_audio(row["absolute_path"], sample_rate)
                    transformed, transform_details = transform_audio(
                        np.asarray(waveform, dtype=np.float32).reshape(-1),
                        sample_rate,
                        condition_name,
                        row["clip_id"],
                        seed,
                    )
                    output_duration = transform_details["output_duration_seconds"]
                    measured_snr = transform_details.get("measured_snr_db", "")
                    padded_samples = transform_details.get("padded_samples", "")
                    sf.write(
                        temporary_audio,
                        transformed,
                        int(transform_details.get("sample_rate", sample_rate)),
                        subtype="FLOAT",
                    )
                    embedding = np.asarray(
                        extract_embedding(temporary_audio, model_id),
                        dtype=np.float32,
                    ).reshape(-1)
                    if embedding.size == 0:
                        raise ValueError("Embedding is empty.")
                    if not np.isfinite(embedding).all():
                        raise ValueError("Embedding contains invalid values.")
                    if dimension is None:
                        dimension = int(embedding.size)
                    elif embedding.size != dimension:
                        raise ValueError(
                            f"Embedding dimension mismatch: expected {dimension}, "
                            f"got {embedding.size}."
                        )
                    embedding_index = len(embeddings)
                    embedding_norm = float(np.linalg.norm(embedding))
                    embeddings.append(embedding)
                    labels.append(row["speaker_label"])
                    clip_ids.append(row["clip_id"])
                    folds.append(int(row["fold"]))
                    status = "success"
                except Exception as error:
                    status = "failed"
                    error_type = type(error).__name__
                    error_message = str(error).replace("\r", " ").replace("\n", " ")

                elapsed_ms = (time.perf_counter() - started) * 1000.0
                records.append(
                    {
                        "manifest_index": manifest_index,
                        "clip_id": row["clip_id"],
                        "speaker_label": row["speaker_label"],
                        "source_group": row["source_group"],
                        "fold": int(row["fold"]),
                        "status": status,
                        "embedding_index": embedding_index,
                        "extraction_time_ms": round(elapsed_ms, 6),
                        "output_duration_seconds": output_duration,
                        "measured_snr_db": measured_snr,
                        "padded_samples": padded_samples,
                        "embedding_l2_norm": embedding_norm,
                        "error_type": error_type,
                        "error_message": error_message,
                    }
                )
                next_index = manifest_index + 1
                state["next_row_index"] = next_index
                state["embedding_dimension"] = dimension
                if next_index % checkpoint_every == 0 or next_index == len(rows):
                    save_checkpoint(
                        output_directory,
                        state,
                        embeddings,
                        labels,
                        clip_ids,
                        folds,
                        records,
                    )

        metadata = finalize(
            output_directory,
            model_id,
            condition_name,
            tag,
            seed,
            protocol,
            embeddings,
            labels,
            clip_ids,
            folds,
            records,
            time.perf_counter() - wall_start,
        )
    finally:
        clear_model_cache()

    print()
    print("=" * 72)
    print("ROBUSTNESS EXTRACTION COMPLETED")
    print("=" * 72)
    print(f"Successful : {metadata['successful_rows']}")
    print(f"Failed     : {metadata['failed_rows']}")
    print(f"Coverage   : {metadata['coverage_percent']:.2f}%")
    print(f"Dimension  : {metadata['embedding_dimension']}")
    print(f"Artifacts  : {output_directory}")
    print("=" * 72)
    return metadata


def dry_run(
    model_ids: list[str],
    conditions: list[str],
    rows: list[dict[str, str]],
    protocol: dict[str, Any],
    output_root: Path,
    tag: str,
) -> None:
    print("=" * 72)
    print("ROBUSTNESS EXTRACTION DRY RUN")
    print("=" * 72)
    print(f"Protocol   : {protocol['protocol_id']}")
    print(f"Rows       : {len(rows)}")
    print(f"Models     : {len(model_ids)}")
    print(f"Conditions : {len(conditions)}")
    for model_id in model_ids:
        for condition in conditions:
            destination = (
                output_root.resolve()
                / str(protocol["protocol_id"])
                / model_id
                / condition
                / tag
            )
            print(f"- {model_id} / {condition} -> {destination}")
    print("=" * 72)


def main() -> None:
    arguments = parse_arguments()
    tag = safe_tag(arguments.tag)
    rows, protocol = load_protocol_and_manifest(
        arguments.protocol_file,
        arguments.manifest,
        arguments.limit,
    )
    model_ids = list(MODEL_CONFIGS) if arguments.all_models else [arguments.model]
    if arguments.all_conditions:
        conditions = list(CONDITIONS)
    elif arguments.conditions:
        conditions = list(dict.fromkeys(arguments.conditions))
    else:
        conditions = [arguments.condition]
    if arguments.dry_run:
        dry_run(model_ids, conditions, rows, protocol, arguments.output_root, tag)
        return
    for model_id in model_ids:
        for condition in conditions:
            condition_directory = (
                arguments.output_root.resolve()
                / str(protocol["protocol_id"])
                / model_id
                / condition
                / tag
            )
            condition_resume = bool(
                arguments.resume
                and (condition_directory / "checkpoint_state.json").is_file()
            )
            extract_condition(
                model_id=model_id,
                condition_name=condition,
                rows=rows,
                protocol=protocol,
                output_root=arguments.output_root,
                tag=tag,
                seed=arguments.seed,
                checkpoint_every=arguments.checkpoint_every,
                resume=condition_resume,
                overwrite=arguments.overwrite,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print("ROBUSTNESS EXTRACTION FAILED")
        print("=" * 72)
        print(error)
        raise
