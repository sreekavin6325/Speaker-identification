"""Build a reproducible five-fold manifest for LibriSpeech dev-clean.

The LibriSpeech file layout is:

    <speaker_id>/<chapter_id>/<speaker>-<chapter>-<utterance>.flac

The generated manifest preserves the speaker and chapter identifiers and
assigns every utterance to one deterministic stratified fold.  The requested
five-fold experiment is utterance-level because the downloaded dev-clean
subset contains only one to four chapters per speaker; that limitation is
recorded explicitly in the summary JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:  # Standard-library FLAC fallback is used below.
    sf = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "research_datasets"
    / "LibriSpeech"
    / "dev-clean"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "manifests"
    / "librispeech_dev_clean_manifest.csv"
)

FIELDNAMES = [
    "dataset",
    "clip_id",
    "speaker_id",
    "speaker_label",
    "chapter_id",
    "source_group",
    "utterance_id",
    "absolute_path",
    "relative_path",
    "duration_seconds",
    "sample_rate",
    "channels",
    "fold",
]


def read_flac_metadata(audio_path: Path) -> tuple[int, int, int]:
    """Return total samples, sample rate, and channels for a FLAC file."""

    if sf is not None:
        info = sf.info(str(audio_path))
        return (
            int(info.frames),
            int(info.samplerate),
            int(info.channels),
        )

    with audio_path.open("rb") as audio_file:
        if audio_file.read(4) != b"fLaC":
            raise ValueError(f"Not a FLAC file: {audio_path}")

        while True:
            block_header = audio_file.read(4)

            if len(block_header) != 4:
                raise ValueError(
                    f"FLAC metadata is incomplete: {audio_path}"
                )

            is_last = bool(block_header[0] & 0x80)
            block_type = block_header[0] & 0x7F
            block_length = int.from_bytes(
                block_header[1:4],
                byteorder="big",
            )
            block_data = audio_file.read(block_length)

            if len(block_data) != block_length:
                raise ValueError(
                    f"FLAC metadata block is incomplete: {audio_path}"
                )

            if block_type == 0:
                if block_length < 18:
                    raise ValueError(
                        f"Invalid FLAC STREAMINFO block: {audio_path}"
                    )

                packed = int.from_bytes(
                    block_data[10:18],
                    byteorder="big",
                )
                sample_rate = (packed >> 44) & 0xFFFFF
                channels = ((packed >> 41) & 0x7) + 1
                total_samples = packed & 0xFFFFFFFFF

                if sample_rate <= 0 or total_samples <= 0:
                    raise ValueError(
                        f"Invalid FLAC stream values: {audio_path}"
                    )

                return total_samples, sample_rate, channels

            if is_last:
                break

    raise ValueError(f"FLAC STREAMINFO was not found: {audio_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic five-fold LibriSpeech dev-clean "
            "research manifest."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the LibriSpeech dev-clean directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV manifest.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified folds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for fold assignment.",
    )
    return parser.parse_args()


def parse_librispeech_path(
    audio_path: Path,
    dataset_root: Path,
) -> dict[str, object]:
    relative = audio_path.relative_to(dataset_root)

    if len(relative.parts) != 3:
        raise ValueError(
            "Unexpected LibriSpeech path layout: "
            f"{relative.as_posix()}"
        )

    speaker_directory, chapter_directory, filename = relative.parts
    stem_parts = Path(filename).stem.split("-")

    if len(stem_parts) != 3:
        raise ValueError(
            f"Unexpected LibriSpeech filename: {filename}"
        )

    file_speaker, file_chapter, utterance_id = stem_parts

    if file_speaker != speaker_directory:
        raise ValueError(
            f"Speaker mismatch in path: {relative.as_posix()}"
        )

    if file_chapter != chapter_directory:
        raise ValueError(
            f"Chapter mismatch in path: {relative.as_posix()}"
        )

    total_samples, sample_rate, channels = read_flac_metadata(
        audio_path
    )
    duration = float(total_samples) / float(sample_rate)
    clip_id = f"{file_speaker}-{file_chapter}-{utterance_id}"

    return {
        "dataset": "librispeech_dev_clean",
        "clip_id": clip_id,
        "speaker_id": file_speaker,
        "speaker_label": f"LibriSpeech_{file_speaker}",
        "chapter_id": file_chapter,
        "source_group": f"{file_speaker}/{file_chapter}",
        "utterance_id": utterance_id,
        "absolute_path": str(audio_path.resolve()),
        "relative_path": relative.as_posix(),
        "duration_seconds": round(duration, 6),
        "sample_rate": sample_rate,
        "channels": channels,
        "fold": -1,
    }


def collect_rows(dataset_root: Path) -> list[dict[str, object]]:
    dataset_root = dataset_root.resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            "LibriSpeech dev-clean directory was not found:\n"
            f"{dataset_root}"
        )

    audio_files = sorted(
        dataset_root.rglob("*.flac"),
        key=lambda path: path.as_posix().lower(),
    )

    if not audio_files:
        raise FileNotFoundError(
            f"No FLAC files were found below:\n{dataset_root}"
        )

    rows = [
        parse_librispeech_path(audio_path, dataset_root)
        for audio_path in audio_files
    ]

    clip_ids = [str(row["clip_id"]) for row in rows]

    if len(clip_ids) != len(set(clip_ids)):
        raise RuntimeError("Duplicate clip identifiers were detected.")

    return rows


def assign_stratified_folds(
    rows: list[dict[str, object]],
    n_splits: int,
    seed: int,
) -> None:
    if n_splits < 2:
        raise ValueError("At least two folds are required.")

    labels = np.asarray(
        [str(row["speaker_id"]) for row in rows],
        dtype=str,
    )
    speaker_counts = Counter(labels.tolist())
    smallest_class = min(speaker_counts.values())

    if smallest_class < n_splits:
        raise ValueError(
            "Every speaker needs at least one utterance per fold. "
            f"Smallest speaker class: {smallest_class}; "
            f"requested folds: {n_splits}."
        )

    indices_by_speaker: dict[str, list[int]] = defaultdict(list)

    for row_index, label in enumerate(labels.tolist()):
        indices_by_speaker[label].append(row_index)

    random_generator = np.random.default_rng(seed)

    for speaker_offset, speaker in enumerate(
        sorted(indices_by_speaker)
    ):
        speaker_indices = np.asarray(
            indices_by_speaker[speaker],
            dtype=np.int64,
        )
        random_generator.shuffle(speaker_indices)
        starting_fold = speaker_offset % n_splits

        for offset, row_index in enumerate(speaker_indices):
            fold_index = (
                (starting_fold + offset) % n_splits
            ) + 1
            rows[int(row_index)]["fold"] = fold_index

    if any(int(row["fold"]) < 1 for row in rows):
        raise RuntimeError("At least one utterance has no fold assignment.")


def validate_fold_coverage(
    rows: list[dict[str, object]],
    n_splits: int,
) -> None:
    all_speakers = {
        str(row["speaker_id"])
        for row in rows
    }

    for fold in range(1, n_splits + 1):
        fold_speakers = {
            str(row["speaker_id"])
            for row in rows
            if int(row["fold"]) == fold
        }

        missing = sorted(all_speakers - fold_speakers)

        if missing:
            raise RuntimeError(
                f"Fold {fold} is missing speakers: {missing}"
            )


def write_manifest(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)


def manifest_sha256(output_path: Path) -> str:
    digest = hashlib.sha256()

    with output_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_summary(
    rows: list[dict[str, object]],
    dataset_root: Path,
    output_path: Path,
    n_splits: int,
    seed: int,
) -> dict[str, object]:
    speaker_counts = Counter(
        str(row["speaker_id"])
        for row in rows
    )
    chapter_sets: dict[str, set[str]] = defaultdict(set)
    fold_counts = Counter(
        int(row["fold"])
        for row in rows
    )
    fold_speaker_counts: dict[str, dict[str, int]] = {}

    for row in rows:
        chapter_sets[str(row["speaker_id"])].add(
            str(row["chapter_id"])
        )

    for fold in range(1, n_splits + 1):
        fold_speaker_counts[str(fold)] = dict(
            sorted(
                Counter(
                    str(row["speaker_id"])
                    for row in rows
                    if int(row["fold"]) == fold
                ).items()
            )
        )

    durations = [
        float(row["duration_seconds"])
        for row in rows
    ]
    sample_rates = Counter(
        int(row["sample_rate"])
        for row in rows
    )
    channels = Counter(
        int(row["channels"])
        for row in rows
    )
    chapters_per_speaker = {
        speaker: len(chapters)
        for speaker, chapters in sorted(chapter_sets.items())
    }

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "dataset": "LibriSpeech dev-clean",
        "dataset_root": str(dataset_root.resolve()),
        "manifest_path": str(output_path.resolve()),
        "manifest_sha256": manifest_sha256(output_path),
        "splitter": "DeterministicStratifiedRoundRobin",
        "split_level": "utterance",
        "n_splits": int(n_splits),
        "random_seed": int(seed),
        "number_of_utterances": len(rows),
        "number_of_speakers": len(speaker_counts),
        "number_of_source_groups": len(
            {
                str(row["source_group"])
                for row in rows
            }
        ),
        "speaker_counts": dict(sorted(speaker_counts.items())),
        "chapters_per_speaker": chapters_per_speaker,
        "minimum_chapters_per_speaker": min(
            chapters_per_speaker.values()
        ),
        "maximum_chapters_per_speaker": max(
            chapters_per_speaker.values()
        ),
        "fold_counts": {
            str(key): int(value)
            for key, value in sorted(fold_counts.items())
        },
        "fold_speaker_counts": fold_speaker_counts,
        "duration_seconds": {
            "minimum": min(durations),
            "maximum": max(durations),
            "mean": float(np.mean(durations)),
            "total": float(np.sum(durations)),
        },
        "sample_rate_counts": {
            str(key): int(value)
            for key, value in sorted(sample_rates.items())
        },
        "channel_counts": {
            str(key): int(value)
            for key, value in sorted(channels.items())
        },
        "scientific_limitations": [
            (
                "The five-fold assignment is utterance-level rather "
                "than chapter-grouped because dev-clean contains only "
                "one to four chapters per speaker."
            ),
            (
                "Chapter identifiers are retained so a separate "
                "chapter-overlap sensitivity analysis can be reported."
            ),
        ],
    }


def write_summary(
    summary: dict[str, object],
    manifest_path: Path,
) -> Path:
    summary_path = manifest_path.with_suffix(".summary.json")

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            summary,
            summary_file,
            indent=2,
            ensure_ascii=False,
        )

    return summary_path


def main() -> None:
    arguments = parse_arguments()
    dataset_root = arguments.dataset_root.resolve()
    output_path = arguments.output.resolve()

    print("=" * 72)
    print("LIBRISPEECH RESEARCH MANIFEST")
    print("=" * 72)
    print(f"Dataset : {dataset_root}")
    print(f"Output  : {output_path}")
    print(f"Folds   : {arguments.folds}")
    print(f"Seed    : {arguments.seed}")
    print("=" * 72)

    rows = collect_rows(dataset_root)
    assign_stratified_folds(
        rows,
        n_splits=arguments.folds,
        seed=arguments.seed,
    )
    validate_fold_coverage(rows, arguments.folds)
    write_manifest(rows, output_path)

    summary = build_summary(
        rows,
        dataset_root=dataset_root,
        output_path=output_path,
        n_splits=arguments.folds,
        seed=arguments.seed,
    )
    summary_path = write_summary(summary, output_path)

    print(f"Utterances: {summary['number_of_utterances']}")
    print(f"Speakers  : {summary['number_of_speakers']}")
    print(f"Chapters  : {summary['number_of_source_groups']}")
    print(f"Fold sizes: {summary['fold_counts']}")
    print(f"Manifest  : {output_path}")
    print(f"Summary   : {summary_path}")
    print("=" * 72)
    print(
        "Important: this requested five-fold protocol is "
        "utterance-level. Chapter IDs remain in the manifest for "
        "a separate overlap-sensitivity analysis."
    )


if __name__ == "__main__":
    main()
