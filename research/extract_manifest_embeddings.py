"""Manifest-driven embedding extraction for reproducible experiments.

Every encoder receives the same persisted manifest rows. Research artifacts
are written below ``research_results`` and never overwrite the desktop
application's operational embeddings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from model_config import MODEL_CONFIGS, get_model_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "manifests"
    / "librispeech_dev_clean_manifest.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "research_results"
    / "embeddings"
    / "librispeech_dev_clean"
)
REQUIRED_COLUMNS = {
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
            "Extract speaker embeddings from a fixed research manifest."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS),
        help="One configured embedding model.",
    )
    selection.add_argument(
        "--all-models",
        action="store_true",
        help="Run all configured models sequentially.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--tag",
        default="full",
        help="Artifact tag, for example full or smoke.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows for a smoke test.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without loading a model.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_manifest(
    manifest_path: Path,
    limit: int | None,
) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest was not found:\n{manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as manifest_file:
        reader = csv.DictReader(manifest_file)

        if reader.fieldnames is None:
            raise ValueError("Manifest has no header.")

        missing = sorted(
            REQUIRED_COLUMNS - set(reader.fieldnames)
        )

        if missing:
            raise ValueError(
                "Manifest is missing columns: "
                + ", ".join(missing)
            )

        rows = list(reader)

    if not rows:
        raise ValueError("Manifest is empty.")

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least one.")
        rows = rows[:limit]

    seen: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        clip_id = row["clip_id"].strip()

        if not clip_id:
            raise ValueError(
                f"Manifest row {row_number} has no clip_id."
            )

        if clip_id in seen:
            raise ValueError(
                f"Duplicate clip_id: {clip_id}"
            )

        seen.add(clip_id)
        audio_path = Path(row["absolute_path"])

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Manifest audio was not found:\n{audio_path}"
            )

        fold = int(row["fold"])

        if fold < 1:
            raise ValueError(
                f"Invalid fold for {clip_id}: {fold}"
            )

    return rows


def safe_tag(value: str) -> str:
    value = value.strip()

    if not value:
        raise ValueError("--tag cannot be empty.")

    if any(character in value for character in r'\/:*?"<>|'):
        raise ValueError(
            "--tag contains an invalid Windows filename character."
        )

    return value


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(
            value,
            output_file,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, value)
    os.replace(temporary, path)


def stack_embeddings(
    embeddings: list[np.ndarray],
    dimension: int | None,
) -> np.ndarray:
    if embeddings:
        return np.stack(
            embeddings,
            axis=0,
        ).astype(np.float32)

    return np.empty(
        (0, int(dimension or 0)),
        dtype=np.float32,
    )


def save_checkpoint(
    output_directory: Path,
    state: dict[str, object],
    embeddings: list[np.ndarray],
    labels: list[str],
    clip_ids: list[str],
    paths: list[str],
    folds: list[int],
    times_ms: list[float],
    records: list[dict[str, object]],
) -> None:
    atomic_npy(
        output_directory / "checkpoint_X.npy",
        stack_embeddings(
            embeddings,
            state.get("embedding_dimension"),
        ),
    )
    atomic_npy(
        output_directory / "checkpoint_y.npy",
        np.asarray(labels, dtype=str),
    )
    atomic_npy(
        output_directory / "checkpoint_clip_ids.npy",
        np.asarray(clip_ids, dtype=str),
    )
    atomic_npy(
        output_directory / "checkpoint_paths.npy",
        np.asarray(paths, dtype=str),
    )
    atomic_npy(
        output_directory / "checkpoint_folds.npy",
        np.asarray(folds, dtype=np.int16),
    )
    atomic_npy(
        output_directory / "checkpoint_times_ms.npy",
        np.asarray(times_ms, dtype=np.float64),
    )
    atomic_json(
        output_directory / "checkpoint_records.json",
        records,
    )
    state["successful_rows"] = len(embeddings)
    state["processed_rows"] = len(records)
    state["updated_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")
    atomic_json(
        output_directory / "checkpoint_state.json",
        state,
    )


def load_checkpoint(
    output_directory: Path,
    model_id: str,
    manifest_hash: str,
) -> tuple[
    dict[str, object],
    list[np.ndarray],
    list[str],
    list[str],
    list[str],
    list[int],
    list[float],
    list[dict[str, object]],
]:
    state_path = output_directory / "checkpoint_state.json"

    if not state_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint was not found:\n{state_path}"
        )

    with state_path.open("r", encoding="utf-8") as state_file:
        state = json.load(state_file)

    if state.get("model_id") != model_id:
        raise ValueError("Checkpoint model does not match.")

    if state.get("manifest_sha256") != manifest_hash:
        raise ValueError("Checkpoint manifest does not match.")

    X = np.load(output_directory / "checkpoint_X.npy")
    labels = np.load(
        output_directory / "checkpoint_y.npy"
    ).astype(str).tolist()
    clip_ids = np.load(
        output_directory / "checkpoint_clip_ids.npy"
    ).astype(str).tolist()
    paths = np.load(
        output_directory / "checkpoint_paths.npy"
    ).astype(str).tolist()
    folds = np.load(
        output_directory / "checkpoint_folds.npy"
    ).astype(int).tolist()
    times_ms = np.load(
        output_directory / "checkpoint_times_ms.npy"
    ).astype(float).tolist()

    with (
        output_directory / "checkpoint_records.json"
    ).open("r", encoding="utf-8") as records_file:
        records = json.load(records_file)

    lengths = {
        len(X),
        len(labels),
        len(clip_ids),
        len(paths),
        len(folds),
        len(times_ms),
    }

    if len(lengths) != 1:
        raise RuntimeError(
            "Checkpoint success arrays have inconsistent lengths."
        )

    return (
        state,
        [row.astype(np.float32) for row in X],
        labels,
        clip_ids,
        paths,
        folds,
        times_ms,
        records,
    )


def prepare_output(
    output_directory: Path,
    overwrite: bool,
    resume: bool,
) -> None:
    if overwrite and resume:
        raise ValueError(
            "--overwrite and --resume cannot be combined."
        )

    if resume:
        if not (
            output_directory / "checkpoint_state.json"
        ).is_file():
            raise FileNotFoundError(
                f"No checkpoint exists in:\n{output_directory}"
            )
        return

    if (
        output_directory.exists()
        and any(output_directory.iterdir())
        and not overwrite
    ):
        raise FileExistsError(
            f"Output is not empty:\n{output_directory}\n"
            "Use --resume, --overwrite, or another --tag."
        )

    output_directory.mkdir(parents=True, exist_ok=True)


def progress_rows(
    rows: list[dict[str, str]],
    description: str,
):
    indexed_rows = list(enumerate(rows))

    try:
        from tqdm import tqdm
    except ImportError:
        return indexed_rows

    return tqdm(
        indexed_rows,
        desc=description,
        unit="file",
    )


def timing_summary(values: list[float]) -> dict[str, float]:
    values = [
        float(value)
        for value in values
        if math.isfinite(float(value))
    ]

    if not values:
        return {
            "minimum": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p95": 0.0,
            "maximum": 0.0,
            "total": 0.0,
        }

    ordered = sorted(values)
    p95_index = max(
        0,
        math.ceil(0.95 * len(ordered)) - 1,
    )

    return {
        "minimum": min(ordered),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[p95_index],
        "maximum": max(ordered),
        "total": sum(ordered),
    }


def write_records(
    records: list[dict[str, object]],
    path: Path,
) -> None:
    fields = [
        "manifest_index",
        "clip_id",
        "speaker_id",
        "speaker_label",
        "chapter_id",
        "source_group",
        "fold",
        "absolute_path",
        "status",
        "embedding_index",
        "extraction_time_ms",
        "embedding_l2_norm",
        "error_type",
        "error_message",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(records)


def save_final_artifacts(
    output_directory: Path,
    model_id: str,
    tag: str,
    manifest_path: Path,
    manifest_hash: str,
    requested_rows: int,
    embeddings: list[np.ndarray],
    labels: list[str],
    clip_ids: list[str],
    paths: list[str],
    folds: list[int],
    times_ms: list[float],
    records: list[dict[str, object]],
    wall_seconds: float,
) -> dict[str, object]:
    if not embeddings:
        raise RuntimeError(
            "All manifest rows failed embedding extraction."
        )

    X = stack_embeddings(
        embeddings,
        embeddings[0].size,
    )

    if X.ndim != 2 or not np.isfinite(X).all():
        raise RuntimeError("Final embedding matrix is invalid.")

    arrays = {
        "X.npy": X,
        "y.npy": np.asarray(labels, dtype=str),
        "clip_ids.npy": np.asarray(clip_ids, dtype=str),
        "audio_paths.npy": np.asarray(paths, dtype=str),
        "folds.npy": np.asarray(folds, dtype=np.int16),
        "extraction_times_ms.npy": np.asarray(
            times_ms,
            dtype=np.float64,
        ),
    }
    array_lengths = {len(value) for value in arrays.values()}

    if len(array_lengths) != 1:
        raise RuntimeError(
            "Final research arrays have inconsistent lengths."
        )

    for filename, value in arrays.items():
        atomic_npy(output_directory / filename, value)

    write_records(
        records,
        output_directory / "records.csv",
    )
    failures = [
        record
        for record in records
        if record["status"] == "failed"
    ]
    config = get_model_config(model_id)
    failure_types = Counter(
        str(record["error_type"])
        for record in failures
    )

    try:
        from embedding_extractors import get_device

        device = str(get_device())
    except Exception:
        device = "unknown"

    metadata = {
        "schema_version": 1,
        "experiment": "manifest_embedding_extraction",
        "tag": tag,
        "model_id": model_id,
        "display_name": config["display_name"],
        "provider": config["provider"],
        "architecture": config["architecture"],
        "pretrained_name": (
            config.get("pretrained_name")
            or config.get("source")
        ),
        "device": device,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "requested_rows": requested_rows,
        "processed_rows": len(records),
        "successful_rows": len(embeddings),
        "failed_rows": len(failures),
        "coverage_percent": (
            100.0 * len(embeddings) / len(records)
        ),
        "embedding_dimension": int(X.shape[1]),
        "number_of_speakers": len(set(labels)),
        "number_of_folds": len(set(folds)),
        "cold_first_success_ms": float(times_ms[0]),
        "all_success_timing_ms": timing_summary(times_ms),
        "warm_success_timing_ms": timing_summary(times_ms[1:]),
        "wall_seconds": float(wall_seconds),
        "failure_types": dict(sorted(failure_types.items())),
        "completed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),
    }
    atomic_json(
        output_directory / "metadata.json",
        metadata,
    )
    return metadata


def extract_model(
    model_id: str,
    rows: list[dict[str, str]],
    manifest_path: Path,
    output_root: Path,
    tag: str,
    checkpoint_every: int,
    resume: bool,
    overwrite: bool,
) -> dict[str, object]:
    if checkpoint_every < 1:
        raise ValueError(
            "--checkpoint-every must be at least one."
        )

    output_directory = (
        output_root.resolve()
        / model_id
        / tag
    )
    prepare_output(
        output_directory,
        overwrite=overwrite,
        resume=resume,
    )
    manifest_hash = file_sha256(manifest_path)

    if resume:
        (
            state,
            embeddings,
            labels,
            clip_ids,
            paths,
            folds,
            times_ms,
            records,
        ) = load_checkpoint(
            output_directory,
            model_id,
            manifest_hash,
        )
        start_index = int(state["next_row_index"])
        dimension = (
            int(state["embedding_dimension"])
            if state.get("embedding_dimension") is not None
            else None
        )
    else:
        start_index = 0
        dimension = None
        embeddings = []
        labels = []
        clip_ids = []
        paths = []
        folds = []
        times_ms = []
        records = []
        state = {
            "schema_version": 1,
            "model_id": model_id,
            "manifest_sha256": manifest_hash,
            "next_row_index": 0,
            "embedding_dimension": None,
        }

    if start_index > len(rows):
        raise RuntimeError(
            "Checkpoint index exceeds manifest length."
        )

    config = get_model_config(model_id)

    print()
    print("=" * 72)
    print("RESEARCH EMBEDDING EXTRACTION")
    print("=" * 72)
    print(f"Model       : {config['display_name']}")
    print(f"Rows        : {len(rows)}")
    print(f"Resume from : {start_index}")
    print(f"Output      : {output_directory}")
    print("=" * 72)

    try:
        from embedding_extractors import (
            clear_model_cache,
            extract_embedding,
        )
    except Exception as error:
        raise RuntimeError(
            "Unable to import project embedding extractors. "
            "Run with the project .venv Python."
        ) from error

    wall_start = time.perf_counter()
    remaining_rows = rows[start_index:]

    try:
        for offset, row in progress_rows(
            remaining_rows,
            config["display_name"],
        ):
            manifest_index = start_index + offset
            started = time.perf_counter()
            embedding_index: int | str = ""
            embedding_norm: float | str = ""
            error_type = ""
            error_message = ""

            try:
                embedding = np.asarray(
                    extract_embedding(
                        row["absolute_path"],
                        model_id,
                    ),
                    dtype=np.float32,
                ).reshape(-1)

                if embedding.size == 0:
                    raise ValueError("Embedding is empty.")

                if not np.isfinite(embedding).all():
                    raise ValueError(
                        "Embedding contains invalid values."
                    )

                if dimension is None:
                    dimension = int(embedding.size)
                elif embedding.size != dimension:
                    raise ValueError(
                        "Embedding dimension mismatch: "
                        f"expected {dimension}, got {embedding.size}."
                    )

                elapsed_ms = (
                    time.perf_counter() - started
                ) * 1000.0
                embedding_index = len(embeddings)
                embedding_norm = float(
                    np.linalg.norm(embedding)
                )
                embeddings.append(embedding)
                labels.append(row["speaker_label"])
                clip_ids.append(row["clip_id"])
                paths.append(row["absolute_path"])
                folds.append(int(row["fold"]))
                times_ms.append(elapsed_ms)
                status = "success"

            except Exception as error:
                elapsed_ms = (
                    time.perf_counter() - started
                ) * 1000.0
                status = "failed"
                error_type = type(error).__name__
                error_message = str(error).replace(
                    "\r",
                    " ",
                ).replace("\n", " ")

            records.append(
                {
                    "manifest_index": manifest_index,
                    "clip_id": row["clip_id"],
                    "speaker_id": row["speaker_id"],
                    "speaker_label": row["speaker_label"],
                    "chapter_id": row["chapter_id"],
                    "source_group": row["source_group"],
                    "fold": int(row["fold"]),
                    "absolute_path": row["absolute_path"],
                    "status": status,
                    "embedding_index": embedding_index,
                    "extraction_time_ms": round(
                        elapsed_ms,
                        6,
                    ),
                    "embedding_l2_norm": embedding_norm,
                    "error_type": error_type,
                    "error_message": error_message,
                }
            )
            next_index = manifest_index + 1
            state["next_row_index"] = next_index
            state["embedding_dimension"] = dimension

            if (
                next_index % checkpoint_every == 0
                or next_index == len(rows)
            ):
                save_checkpoint(
                    output_directory,
                    state,
                    embeddings,
                    labels,
                    clip_ids,
                    paths,
                    folds,
                    times_ms,
                    records,
                )

        metadata = save_final_artifacts(
            output_directory=output_directory,
            model_id=model_id,
            tag=tag,
            manifest_path=manifest_path,
            manifest_hash=manifest_hash,
            requested_rows=len(rows),
            embeddings=embeddings,
            labels=labels,
            clip_ids=clip_ids,
            paths=paths,
            folds=folds,
            times_ms=times_ms,
            records=records,
            wall_seconds=time.perf_counter() - wall_start,
        )

    finally:
        clear_model_cache()

    print()
    print("=" * 72)
    print("EXTRACTION COMPLETED")
    print("=" * 72)
    print(f"Successful : {metadata['successful_rows']}")
    print(f"Failed     : {metadata['failed_rows']}")
    print(f"Coverage   : {metadata['coverage_percent']:.2f}%")
    print(f"Dimension  : {metadata['embedding_dimension']}")
    print(
        "Warm median: "
        f"{metadata['warm_success_timing_ms']['median']:.2f} ms"
    )
    print(f"Artifacts  : {output_directory}")
    print("=" * 72)
    return metadata


def dry_run(
    model_ids: list[str],
    rows: list[dict[str, str]],
    manifest_path: Path,
    output_root: Path,
    tag: str,
) -> None:
    print("=" * 72)
    print("RESEARCH EXTRACTION DRY RUN")
    print("=" * 72)
    print(f"Manifest : {manifest_path}")
    print(f"Rows     : {len(rows)}")
    print(
        "Speakers : "
        f"{len({row['speaker_id'] for row in rows})}"
    )
    print(
        "Folds    : "
        + ", ".join(
            sorted(
                {row["fold"] for row in rows},
                key=int,
            )
        )
    )

    for model_id in model_ids:
        config = get_model_config(model_id)
        destination = (
            output_root.resolve()
            / model_id
            / tag
        )
        print(
            f"- {model_id}: {config['display_name']} "
            f"-> {destination}"
        )

    print("=" * 72)


def main() -> None:
    arguments = parse_arguments()
    manifest_path = arguments.manifest.resolve()
    tag = safe_tag(arguments.tag)

    if arguments.limit is not None and tag == "full":
        raise ValueError(
            "Use --tag smoke (or another non-full tag) "
            "together with --limit."
        )

    rows = load_manifest(
        manifest_path,
        arguments.limit,
    )
    model_ids = (
        list(MODEL_CONFIGS)
        if arguments.all_models
        else [arguments.model]
    )

    if arguments.dry_run:
        dry_run(
            model_ids,
            rows,
            manifest_path,
            arguments.output_root,
            tag,
        )
        return

    for model_id in model_ids:
        extract_model(
            model_id=model_id,
            rows=rows,
            manifest_path=manifest_path,
            output_root=arguments.output_root,
            tag=tag,
            checkpoint_every=arguments.checkpoint_every,
            resume=arguments.resume,
            overwrite=arguments.overwrite,
        )


if __name__ == "__main__":
    main()
