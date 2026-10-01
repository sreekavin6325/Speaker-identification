"""Extract embeddings for a deterministic 2^3 preprocessing experiment.

The three factors are silence trimming, stationary spectral-gate noise
reduction, and peak normalization.  The raw condition reuses the already
validated clean embeddings, so this extractor materializes the seven non-raw
factor combinations for the exact strict-protocol clip order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from model_config import MODEL_CONFIGS, get_model_config
from research.extract_robustness_embeddings import (
    atomic_json,
    atomic_npy,
    file_sha256,
    load_protocol_and_manifest,
    progress_rows,
    safe_tag,
    stack_embeddings,
    timing_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "manifests" / "librispeech_dev_clean_manifest.csv"
)
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "manifests"
    / "librispeech_dev_clean_chapter_heldout_2fold_protocol.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "research_results" / "preprocessing_embeddings"

PREPROCESSING_PARAMETERS: dict[str, Any] = {
    "trim": {
        "method": "numpy frame-RMS threshold relative to maximum frame RMS",
        "top_db": 30.0,
        "frame_length": 2048,
        "hop_length": 512,
        "minimum_output_seconds": 0.5,
    },
    "denoise": {
        "algorithm": "noisereduce stationary spectral gate",
        "stationary": True,
        "prop_decrease": 0.8,
        "n_std_thresh_stationary": 1.5,
        "n_fft": 1024,
        "use_torch": False,
        "n_jobs": 1,
    },
    "normalize": {
        "method": "peak",
        "target_peak": 0.95,
    },
    "order": ["trim", "denoise", "normalize"],
}

PREPROCESSING_VARIANTS: dict[str, dict[str, bool]] = {
    "normalize": {"trim": False, "denoise": False, "normalize": True},
    "trim": {"trim": True, "denoise": False, "normalize": False},
    "denoise": {"trim": False, "denoise": True, "normalize": False},
    "trim_normalize": {"trim": True, "denoise": False, "normalize": True},
    "denoise_normalize": {"trim": False, "denoise": True, "normalize": True},
    "trim_denoise": {"trim": True, "denoise": True, "normalize": False},
    "trim_denoise_normalize": {
        "trim": True,
        "denoise": True,
        "normalize": True,
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract embeddings for the seven non-raw combinations in a "
            "trim x denoise x normalize factorial experiment."
        )
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", choices=list(MODEL_CONFIGS))
    model_group.add_argument("--all-models", action="store_true")
    variant_group = parser.add_mutually_exclusive_group(required=True)
    variant_group.add_argument("--variant", choices=list(PREPROCESSING_VARIANTS))
    variant_group.add_argument("--all-variants", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-file", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="full")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def trim_silence(
    waveform: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    parameters = PREPROCESSING_PARAMETERS["trim"]
    frame_length = int(parameters["frame_length"])
    hop_length = int(parameters["hop_length"])
    top_db = float(parameters["top_db"])

    # Avoid librosa's optional Numba dependency here.  Some managed Windows
    # machines block Numba's native DLLs through Application Control.  A
    # deterministic frame-RMS gate implements the same relative-dB trimming
    # principle without a compiled extension.
    if waveform.size <= frame_length:
        frame_starts = np.array([0], dtype=np.int64)
    else:
        frame_starts = np.arange(
            0,
            waveform.size - frame_length + 1,
            hop_length,
            dtype=np.int64,
        )
        final_start = waveform.size - frame_length
        if frame_starts[-1] != final_start:
            frame_starts = np.append(frame_starts, final_start)

    frame_rms = np.empty(frame_starts.size, dtype=np.float64)
    for index, start in enumerate(frame_starts):
        frame = waveform[start : start + frame_length]
        frame_rms[index] = math.sqrt(
            float(np.mean(np.square(frame, dtype=np.float64)))
        )

    reference_rms = float(np.max(frame_rms))
    if not math.isfinite(reference_rms) or reference_rms <= 1e-8:
        raise ValueError("Waveform energy is too small for silence trimming.")
    threshold = reference_rms * (10.0 ** (-top_db / 20.0))
    non_silent = np.flatnonzero(frame_rms > threshold)
    if non_silent.size == 0:
        raise ValueError("Silence trimming found no non-silent frames.")

    interval = (
        int(frame_starts[int(non_silent[0])]),
        int(
            min(
                waveform.size,
                frame_starts[int(non_silent[-1])] + frame_length,
            )
        ),
    )
    trimmed = waveform[interval[0] : interval[1]]
    minimum_samples = int(
        round(float(parameters["minimum_output_seconds"]) * sample_rate)
    )
    padded_samples = 0
    if trimmed.size < minimum_samples:
        left = (minimum_samples - trimmed.size) // 2
        right = minimum_samples - trimmed.size - left
        trimmed = np.pad(trimmed, (left, right), mode="constant")
        padded_samples = left + right
    if trimmed.size == 0:
        raise ValueError("Silence trimming produced an empty waveform.")
    return np.asarray(trimmed, dtype=np.float32), {
        "trim_start_sample": int(interval[0]),
        "trim_end_sample": int(interval[1]),
        "trimmed_samples": int(waveform.size - (interval[1] - interval[0])),
        "trim_padding_samples": int(padded_samples),
    }


def reduce_stationary_noise(
    waveform: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    import noisereduce as nr

    parameters = PREPROCESSING_PARAMETERS["denoise"]
    output = nr.reduce_noise(
        y=waveform,
        sr=sample_rate,
        stationary=bool(parameters["stationary"]),
        prop_decrease=float(parameters["prop_decrease"]),
        n_std_thresh_stationary=float(parameters["n_std_thresh_stationary"]),
        n_fft=int(parameters["n_fft"]),
        use_torch=bool(parameters["use_torch"]),
        n_jobs=int(parameters["n_jobs"]),
        use_tqdm=False,
    )
    output = np.asarray(output, dtype=np.float32).reshape(-1)
    if output.size != waveform.size:
        raise ValueError(
            "Noise reduction changed waveform length: "
            f"{waveform.size} -> {output.size}."
        )
    return output


def peak_normalize(waveform: np.ndarray) -> tuple[np.ndarray, float]:
    target = float(PREPROCESSING_PARAMETERS["normalize"]["target_peak"])
    peak = float(np.max(np.abs(waveform)))
    if not math.isfinite(peak) or peak <= 1e-8:
        raise ValueError("Waveform peak is too small for normalization.")
    scale = target / peak
    return (waveform * np.float32(scale)).astype(np.float32), float(scale)


def signal_statistics(waveform: np.ndarray) -> tuple[float, float]:
    peak = float(np.max(np.abs(waveform)))
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    return peak, rms


def apply_preprocessing(
    waveform: np.ndarray,
    sample_rate: int,
    variant_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    factors = PREPROCESSING_VARIANTS[variant_name]
    output = np.asarray(waveform, dtype=np.float32).reshape(-1)
    input_peak, input_rms = signal_statistics(output)
    details: dict[str, Any] = {
        "input_samples": int(output.size),
        "input_duration_seconds": float(output.size / sample_rate),
        "input_peak": input_peak,
        "input_rms": input_rms,
        "trim_start_sample": 0,
        "trim_end_sample": int(output.size),
        "trimmed_samples": 0,
        "trim_padding_samples": 0,
        "normalization_scale": 1.0,
    }

    if factors["trim"]:
        output, trim_details = trim_silence(output, sample_rate)
        details.update(trim_details)
    if factors["denoise"]:
        output = reduce_stationary_noise(output, sample_rate)
    if factors["normalize"]:
        output, scale = peak_normalize(output)
        details["normalization_scale"] = scale

    if output.ndim != 1 or output.size == 0:
        raise ValueError("Preprocessed waveform is empty or not mono.")
    if not np.isfinite(output).all():
        raise ValueError("Preprocessed waveform contains invalid values.")
    output_peak, output_rms = signal_statistics(output)
    if output_rms < 1e-6:
        raise ValueError("Preprocessed waveform has insufficient energy.")
    details.update(
        {
            "output_samples": int(output.size),
            "output_duration_seconds": float(output.size / sample_rate),
            "output_peak": output_peak,
            "output_rms": output_rms,
        }
    )
    return output.astype(np.float32), details


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
    variant: str,
    protocol_hash: str,
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
    ) as input_file:
        state = json.load(input_file)
    expected = {
        "model_id": model_id,
        "variant": variant,
        "protocol_sha256": protocol_hash,
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
    ) as input_file:
        records = json.load(input_file)
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
        "input_duration_seconds",
        "output_duration_seconds",
        "trimmed_samples",
        "trim_padding_samples",
        "input_peak",
        "output_peak",
        "input_rms",
        "output_rms",
        "normalization_scale",
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
    variant: str,
    tag: str,
    protocol: dict[str, Any],
    embeddings: list[np.ndarray],
    labels: list[str],
    clip_ids: list[str],
    folds: list[int],
    records: list[dict[str, Any]],
    wall_seconds: float,
) -> dict[str, Any]:
    if not embeddings:
        raise RuntimeError("All rows failed preprocessing embedding extraction.")
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
    times = [
        float(record["extraction_time_ms"])
        for record in records
        if record["status"] == "success"
    ]
    config = get_model_config(model_id)
    metadata = {
        "schema_version": 1,
        "experiment": "preprocessing_factorial_embedding_extraction",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": model_id,
        "model": {
            "display_name": config["display_name"],
            "provider": config["provider"],
            "architecture": config["architecture"],
            "pretrained_name": config.get("pretrained_name") or config.get("source"),
        },
        "variant": variant,
        "factors": PREPROCESSING_VARIANTS[variant],
        "preprocessing_parameters": PREPROCESSING_PARAMETERS,
        "tag": tag,
        "protocol": protocol,
        "requested_rows": len(records),
        "successful_rows": len(embeddings),
        "failed_rows": len(failures),
        "coverage_percent": 100.0 * len(embeddings) / len(records),
        "embedding_dimension": int(X.shape[1]),
        "number_of_speakers": len(set(labels)),
        "number_of_folds": len(set(folds)),
        "timing_ms": timing_summary(times),
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


def extract_variant(
    model_id: str,
    variant: str,
    rows: list[dict[str, str]],
    protocol: dict[str, Any],
    output_root: Path,
    tag: str,
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
        / variant
        / tag
    )
    prepare_output(output_directory, overwrite, resume)
    if resume:
        state, embeddings, labels, clip_ids, folds, records = load_checkpoint(
            output_directory,
            model_id,
            variant,
            str(protocol["sha256"]),
        )
        start_index = int(state["next_row_index"])
        dimension = (
            int(state["embedding_dimension"])
            if state.get("embedding_dimension") is not None
            else None
        )
    else:
        state = {
            "schema_version": 1,
            "model_id": model_id,
            "variant": variant,
            "protocol_sha256": protocol["sha256"],
            "next_row_index": 0,
            "embedding_dimension": None,
        }
        start_index = 0
        dimension = None
        embeddings: list[np.ndarray] = []
        labels: list[str] = []
        clip_ids: list[str] = []
        folds: list[int] = []
        records: list[dict[str, Any]] = []

    config = get_model_config(model_id)
    print()
    print("=" * 72)
    print("PREPROCESSING FACTORIAL EMBEDDING EXTRACTION")
    print("=" * 72)
    print(f"Model       : {config['display_name']}")
    print(f"Variant     : {variant}")
    print(f"Factors     : {PREPROCESSING_VARIANTS[variant]}")
    print(f"Rows        : {len(rows)}")
    print(f"Resume from : {start_index}")
    print(f"Output      : {output_directory}")
    print("=" * 72)

    from embedding_extractors import clear_model_cache, extract_embedding, load_audio

    sample_rate = int(config.get("sample_rate", 16000))
    wall_started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="speaker_preprocessing_") as temporary:
            temporary_audio = Path(temporary) / "preprocessed.wav"
            for offset, row in progress_rows(
                rows[start_index:], f"{model_id}/{variant}"
            ):
                manifest_index = start_index + offset
                started = time.perf_counter()
                details: dict[str, Any] = {}
                embedding_index: int | str = ""
                embedding_norm: float | str = ""
                error_type = ""
                error_message = ""
                try:
                    waveform = load_audio(row["absolute_path"], sample_rate)
                    output, details = apply_preprocessing(
                        np.asarray(waveform, dtype=np.float32).reshape(-1),
                        sample_rate,
                        variant,
                    )
                    sf.write(
                        temporary_audio,
                        output,
                        sample_rate,
                        subtype="FLOAT",
                    )
                    embedding = np.asarray(
                        extract_embedding(temporary_audio, model_id),
                        dtype=np.float32,
                    ).reshape(-1)
                    if embedding.size == 0 or not np.isfinite(embedding).all():
                        raise ValueError("Embedding is empty or contains invalid values.")
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
                        "input_duration_seconds": details.get(
                            "input_duration_seconds", ""
                        ),
                        "output_duration_seconds": details.get(
                            "output_duration_seconds", ""
                        ),
                        "trimmed_samples": details.get("trimmed_samples", ""),
                        "trim_padding_samples": details.get(
                            "trim_padding_samples", ""
                        ),
                        "input_peak": details.get("input_peak", ""),
                        "output_peak": details.get("output_peak", ""),
                        "input_rms": details.get("input_rms", ""),
                        "output_rms": details.get("output_rms", ""),
                        "normalization_scale": details.get(
                            "normalization_scale", ""
                        ),
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
            variant,
            tag,
            protocol,
            embeddings,
            labels,
            clip_ids,
            folds,
            records,
            time.perf_counter() - wall_started,
        )
    finally:
        clear_model_cache()

    print()
    print("=" * 72)
    print("PREPROCESSING EXTRACTION COMPLETED")
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
    variants: list[str],
    rows: list[dict[str, str]],
    protocol: dict[str, Any],
    output_root: Path,
    tag: str,
) -> None:
    print("=" * 72)
    print("PREPROCESSING FACTORIAL DRY RUN")
    print("=" * 72)
    print(f"Protocol : {protocol['protocol_id']}")
    print(f"Rows     : {len(rows)}")
    print(f"Models   : {len(model_ids)}")
    print(f"Variants : {len(variants)} (+ raw reference)")
    for model_id in model_ids:
        for variant in variants:
            destination = (
                output_root.resolve()
                / str(protocol["protocol_id"])
                / model_id
                / variant
                / tag
            )
            print(f"- {model_id} / {variant} -> {destination}")
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
    variants = (
        list(PREPROCESSING_VARIANTS)
        if arguments.all_variants
        else [arguments.variant]
    )
    if arguments.dry_run:
        dry_run(model_ids, variants, rows, protocol, arguments.output_root, tag)
        return
    for model_id in model_ids:
        for variant in variants:
            extract_variant(
                model_id=model_id,
                variant=variant,
                rows=rows,
                protocol=protocol,
                output_root=arguments.output_root,
                tag=tag,
                checkpoint_every=arguments.checkpoint_every,
                resume=arguments.resume,
                overwrite=arguments.overwrite,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("=" * 72)
        print("PREPROCESSING EXTRACTION FAILED")
        print("=" * 72)
        print(error)
        raise
