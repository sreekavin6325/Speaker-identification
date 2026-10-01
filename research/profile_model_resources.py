"""Measure CPU inference resource usage for the four embedding models.

Each model is profiled in a fresh subprocess to avoid cache and memory carryover.
The parent mode launches workers sequentially, while worker mode samples its own
resident memory during a cold extraction followed by warm repetitions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "research_results" / "ieee_complete" / "task_10_computation"
MODEL_IDS = (
    "speechbrain_ecapa",
    "speechbrain_xvector",
    "wavlm_base_plus_sv",
    "unispeech_sat_base_plus_sv",
)
MODEL_NAMES = {
    "speechbrain_ecapa": "ECAPA-TDNN",
    "speechbrain_xvector": "X-Vector",
    "wavlm_base_plus_sv": "WavLM",
    "unispeech_sat_base_plus_sv": "UniSpeech-SAT",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-model", choices=MODEL_IDS)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warm-repetitions", type=int, default=5)
    return parser.parse_args()


def default_audio() -> Path:
    manifest = PROJECT_ROOT / "manifests" / "librispeech_dev_clean_manifest.csv"
    import csv

    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        first = next(csv.DictReader(handle))
    return Path(first["absolute_path"]).resolve()


def primary_weight_path(model_id: str) -> Path:
    if model_id == "speechbrain_ecapa":
        return PROJECT_ROOT / "pretrained_models" / model_id / "embedding_model.ckpt"
    if model_id == "speechbrain_xvector":
        return PROJECT_ROOT / "pretrained_models" / model_id / "embedding_model.ckpt"
    repo = {
        "wavlm_base_plus_sv": "models--microsoft--wavlm-base-plus-sv",
        "unispeech_sat_base_plus_sv": "models--microsoft--unispeech-sat-base-plus-sv",
    }[model_id]
    root = Path.home() / ".cache" / "huggingface" / "hub" / repo
    candidates = list(root.rglob("model.safetensors")) or list(root.rglob("pytorch_model.bin"))
    if not candidates:
        raise FileNotFoundError(f"Could not find cached primary weights for {model_id}.")
    return max(candidates, key=lambda path: path.stat().st_size)


def worker(model_id: str, audio_path: Path, output_path: Path, repetitions: int) -> int:
    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss
    stop_event = threading.Event()

    def sample_memory() -> None:
        nonlocal peak_rss
        while not stop_event.wait(0.02):
            try:
                peak_rss = max(peak_rss, process.memory_info().rss)
            except psutil.Error:
                return

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    started_wall = time.perf_counter()
    cpu_start = process.cpu_times()

    from embedding_extractors import clear_model_cache, extract_embedding

    clear_model_cache()
    cold_start = time.perf_counter()
    cold_embedding = extract_embedding(audio_path, model_id)
    cold_ms = (time.perf_counter() - cold_start) * 1000.0
    warm_times: list[float] = []
    for _ in range(repetitions):
        warm_start = time.perf_counter()
        embedding = extract_embedding(audio_path, model_id)
        warm_times.append((time.perf_counter() - warm_start) * 1000.0)
        if np.asarray(embedding).shape != np.asarray(cold_embedding).shape:
            raise ValueError("Embedding shape changed during resource profiling.")

    wall_seconds = time.perf_counter() - started_wall
    cpu_end = process.cpu_times()
    stop_event.set()
    sampler.join(timeout=1.0)
    peak_rss = max(peak_rss, process.memory_info().rss)
    cpu_seconds = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
    logical_cpus = psutil.cpu_count(logical=True) or 1
    weight_path = primary_weight_path(model_id)
    info = sf.info(str(audio_path))

    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A — CPU-only host"
        peak_gpu_mb = (
            float(torch.cuda.max_memory_allocated(0) / (1024**2))
            if gpu_available
            else 0.0
        )
    except Exception:
        gpu_available = False
        gpu_name = "N/A — PyTorch GPU query failed"
        peak_gpu_mb = 0.0

    payload: dict[str, Any] = {
        "model_id": model_id,
        "model": MODEL_NAMES[model_id],
        "device": "cuda" if gpu_available else "cpu",
        "audio_path": str(audio_path),
        "audio_duration_seconds": float(info.duration),
        "audio_sample_rate": int(info.samplerate),
        "embedding_dimension": int(np.asarray(cold_embedding).size),
        "cold_wall_ms": cold_ms,
        "warm_repetitions": repetitions,
        "warm_wall_ms_mean": float(np.mean(warm_times)),
        "warm_wall_ms_median": float(np.median(warm_times)),
        "warm_wall_ms_sd": float(np.std(warm_times, ddof=1)) if repetitions > 1 else 0.0,
        "process_cpu_seconds": cpu_seconds,
        "profile_wall_seconds": wall_seconds,
        "cpu_utilization_percent_one_core_equivalent": 100.0 * cpu_seconds / wall_seconds,
        "cpu_utilization_percent_of_logical_capacity": (
            100.0 * cpu_seconds / wall_seconds / logical_cpus
        ),
        "logical_cpu_count": logical_cpus,
        "baseline_rss_mb": baseline_rss / (1024**2),
        "peak_rss_mb": peak_rss / (1024**2),
        "incremental_peak_rss_mb": (peak_rss - baseline_rss) / (1024**2),
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "peak_gpu_memory_mb": peak_gpu_mb,
        "primary_weight_path": str(weight_path),
        "primary_weight_size_mb": weight_path.stat().st_size / (1024**2),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    clear_model_cache()
    return 0


def parent(audio_path: Path, repetitions: int) -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for model_id in MODEL_IDS:
        output = OUTPUT_ROOT / f"{model_id}_resource_profile.json"
        command = [
            sys.executable,
            "-m",
            "research.profile_model_resources",
            "--worker-model",
            model_id,
            "--audio",
            str(audio_path),
            "--output",
            str(output),
            "--warm-repetitions",
            str(repetitions),
        ]
        print(f"Profiling {MODEL_NAMES[model_id]}...")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    rows = [
        json.loads((OUTPUT_ROOT / f"{model_id}_resource_profile.json").read_text(encoding="utf-8"))
        for model_id in MODEL_IDS
    ]
    import csv

    csv_path = OUTPUT_ROOT / "computational_performance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol = {
        "fresh_subprocess_per_model": True,
        "execution": "sequential CPU profiling",
        "memory_sampling_interval_ms": 20,
        "audio_path": str(audio_path),
        "warm_repetitions": repetitions,
        "model_size_definition": "primary pretrained embedding-weight file",
        "gpu_note": "GPU metrics are N/A when torch.cuda.is_available() is false.",
    }
    (OUTPUT_ROOT / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(f"Computational profile completed: {csv_path}")
    return 0


def main() -> int:
    arguments = parse_arguments()
    audio_path = (arguments.audio or default_audio()).resolve()
    if arguments.worker_model:
        if arguments.output is None:
            raise ValueError("Worker mode requires --output.")
        return worker(
            arguments.worker_model,
            audio_path,
            arguments.output.resolve(),
            arguments.warm_repetitions,
        )
    return parent(audio_path, arguments.warm_repetitions)


if __name__ == "__main__":
    raise SystemExit(main())
