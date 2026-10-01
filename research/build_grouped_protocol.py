"""Build a deterministic chapter-held-out LibriSpeech sensitivity protocol.

LibriSpeech ``dev-clean`` has too few chapters per speaker for a scientifically
valid chapter-held-out five-fold experiment.  This builder therefore creates a
two-fold sensitivity protocol and keeps every audiobook chapter wholly in one
fold.  Speakers with fewer than two distinct chapters are excluded because
they cannot appear in both training and test data without chapter leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "manifests" / "librispeech_dev_clean_manifest.csv"
)
DEFAULT_MANIFEST_OUTPUT = (
    PROJECT_ROOT
    / "manifests"
    / "librispeech_dev_clean_chapter_heldout_2fold.csv"
)
DEFAULT_PROTOCOL_OUTPUT = (
    PROJECT_ROOT
    / "manifests"
    / "librispeech_dev_clean_chapter_heldout_2fold_protocol.csv"
)
DEFAULT_SUMMARY_OUTPUT = (
    PROJECT_ROOT
    / "manifests"
    / "librispeech_dev_clean_chapter_heldout_2fold.summary.json"
)
REQUIRED_COLUMNS = {
    "clip_id",
    "speaker_id",
    "speaker_label",
    "source_group",
    "fold",
}
PROTOCOL_FIELDS = [
    "clip_id",
    "speaker_label",
    "source_group",
    "fold",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic two-fold LibriSpeech protocol in which "
            "a source chapter never appears in both train and test."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_INPUT,
        help="Existing LibriSpeech manifest.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DEFAULT_MANIFEST_OUTPUT,
        help="Filtered manifest with chapter-held-out folds.",
    )
    parser.add_argument(
        "--protocol-output",
        type=Path,
        default=DEFAULT_PROTOCOL_OUTPUT,
        help="Compact clip_id-to-fold protocol CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Protocol construction and exclusion summary JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing protocol artifacts.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(
            value,
            output_file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    os.replace(temporary, path)


def read_manifest(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest was not found:\n{path}")

    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError("Manifest has no header.")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)

        if missing:
            raise ValueError(
                "Manifest is missing columns: "
                + ", ".join(sorted(missing))
            )

        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    clip_counts = Counter(row["clip_id"].strip() for row in rows)
    duplicates = sorted(
        clip_id for clip_id, count in clip_counts.items() if count > 1
    )

    if duplicates:
        raise ValueError(
            "Manifest contains duplicate clip IDs: "
            + ", ".join(duplicates[:10])
        )

    for row_number, row in enumerate(rows, start=2):
        for field in ("clip_id", "speaker_label", "source_group"):
            if not row[field].strip():
                raise ValueError(
                    f"Manifest row {row_number} has an empty {field}."
                )

    return fieldnames, rows


def best_chapter_partition(
    chapter_counts: dict[str, int],
) -> tuple[set[str], set[str], int]:
    """Return the minimum-utterance-imbalance nonempty chapter partition.

    Complementary partitions are equivalent, so the lexicographically first
    chapter is anchored in the first set.  Ties are resolved by the sorted
    chapter tuple, making the result independent of filesystem enumeration.
    """

    chapters = sorted(chapter_counts)

    if len(chapters) < 2:
        raise ValueError("At least two chapters are required.")

    anchor = chapters[0]
    remaining = chapters[1:]
    total = sum(chapter_counts.values())
    candidates: list[tuple[int, tuple[str, ...], set[str]]] = []

    for subset_size in range(0, len(remaining) + 1):
        for chosen in itertools.combinations(remaining, subset_size):
            fold_one = {anchor, *chosen}

            if len(fold_one) == len(chapters):
                continue

            fold_one_count = sum(
                chapter_counts[group] for group in fold_one
            )
            imbalance = abs(total - 2 * fold_one_count)
            candidates.append(
                (imbalance, tuple(sorted(fold_one)), fold_one)
            )

    _, _, fold_one = min(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    fold_two = set(chapters) - fold_one
    imbalance = abs(
        sum(chapter_counts[group] for group in fold_one)
        - sum(chapter_counts[group] for group in fold_two)
    )
    return fold_one, fold_two, imbalance


def build_protocol(
    rows: list[dict[str, str]],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, object]],
    dict[str, object],
]:
    by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        by_speaker[row["speaker_label"].strip()].append(row)

    included_rows: list[dict[str, str]] = []
    speaker_summaries: list[dict[str, object]] = []
    excluded_speakers: list[dict[str, object]] = []

    for speaker_label in sorted(by_speaker):
        speaker_rows = by_speaker[speaker_label]
        chapter_counts = Counter(
            row["source_group"].strip() for row in speaker_rows
        )
        speaker_id = speaker_rows[0]["speaker_id"].strip()

        if len(chapter_counts) < 2:
            excluded_speakers.append(
                {
                    "speaker_id": speaker_id,
                    "speaker_label": speaker_label,
                    "utterances": len(speaker_rows),
                    "distinct_source_groups": len(chapter_counts),
                    "source_groups": sorted(chapter_counts),
                    "reason_code": "single_source_group",
                    "reason": (
                        "A chapter-held-out closed-set fold requires at "
                        "least two distinct chapters so the speaker is "
                        "represented in both training and test partitions."
                    ),
                }
            )
            continue

        fold_one_groups, fold_two_groups, imbalance = (
            best_chapter_partition(dict(chapter_counts))
        )
        fold_by_group = {
            **{group: 1 for group in fold_one_groups},
            **{group: 2 for group in fold_two_groups},
        }
        fold_counts = Counter()

        for source_row in speaker_rows:
            row = dict(source_row)
            source_group = row["source_group"].strip()
            fold = fold_by_group[source_group]
            row["fold"] = str(fold)
            fold_counts[fold] += 1
            included_rows.append(row)

        speaker_summaries.append(
            {
                "speaker_id": speaker_id,
                "speaker_label": speaker_label,
                "utterances": len(speaker_rows),
                "distinct_source_groups": len(chapter_counts),
                "fold_1_source_groups": sorted(fold_one_groups),
                "fold_2_source_groups": sorted(fold_two_groups),
                "fold_1_utterances": fold_counts[1],
                "fold_2_utterances": fold_counts[2],
                "absolute_utterance_imbalance": imbalance,
            }
        )

    # Preserve the source-manifest row order for deterministic clip alignment.
    source_order = {
        row["clip_id"].strip(): index for index, row in enumerate(rows)
    }
    included_rows.sort(
        key=lambda row: source_order[row["clip_id"].strip()]
    )

    protocol_rows: list[dict[str, object]] = [
        {
            "clip_id": row["clip_id"].strip(),
            "speaker_label": row["speaker_label"].strip(),
            "source_group": row["source_group"].strip(),
            "fold": int(row["fold"]),
        }
        for row in included_rows
    ]
    fold_counts = Counter(
        int(row["fold"]) for row in included_rows
    )
    fold_group_sets = {
        fold: {
            row["source_group"].strip()
            for row in included_rows
            if int(row["fold"]) == fold
        }
        for fold in (1, 2)
    }
    overlap = fold_group_sets[1] & fold_group_sets[2]

    if overlap:
        raise RuntimeError(
            "Protocol construction produced source-group leakage: "
            + ", ".join(sorted(overlap)[:10])
        )

    for speaker_label in sorted(
        {row["speaker_label"].strip() for row in included_rows}
    ):
        speaker_folds = {
            int(row["fold"])
            for row in included_rows
            if row["speaker_label"].strip() == speaker_label
        }

        if speaker_folds != {1, 2}:
            raise RuntimeError(
                f"Speaker {speaker_label} is not represented in both folds."
            )

    summary = {
        "input_rows": len(rows),
        "input_speakers": len(by_speaker),
        "included_rows": len(included_rows),
        "included_speakers": len(speaker_summaries),
        "eligible_speakers": len(speaker_summaries),
        "excluded_rows": len(rows) - len(included_rows),
        "excluded_utterances": len(rows) - len(included_rows),
        "excluded_speakers": len(excluded_speakers),
        "exclusion_rule": "minimum_2_source_groups_per_speaker",
        "fold_utterance_counts": {
            str(fold): fold_counts[fold] for fold in (1, 2)
        },
        "fold_source_group_counts": {
            str(fold): len(fold_group_sets[fold]) for fold in (1, 2)
        },
        "cross_fold_source_group_overlap_count": len(overlap),
        "speaker_summaries": speaker_summaries,
        "exclusions": excluded_speakers,
    }
    return included_rows, protocol_rows, summary


def ensure_outputs_available(
    paths: list[Path],
    overwrite: bool,
) -> None:
    existing = [path for path in paths if path.exists()]

    if existing and not overwrite:
        raise FileExistsError(
            "Protocol output already exists:\n"
            + "\n".join(str(path) for path in existing)
            + "\nUse --overwrite to replace it."
        )


def main() -> int:
    arguments = parse_arguments()
    manifest_path = arguments.manifest.resolve()
    manifest_output = arguments.output_manifest.resolve()
    protocol_output = arguments.protocol_output.resolve()
    summary_output = arguments.summary_output.resolve()
    ensure_outputs_available(
        [manifest_output, protocol_output, summary_output],
        arguments.overwrite,
    )
    fieldnames, rows = read_manifest(manifest_path)
    included_rows, protocol_rows, summary = build_protocol(rows)

    if not included_rows:
        raise ValueError(
            "No speakers have at least two distinct source groups."
        )

    write_csv(manifest_output, fieldnames, included_rows)
    write_csv(protocol_output, PROTOCOL_FIELDS, protocol_rows)
    summary_payload = {
        "schema_version": 1,
        "protocol_name": "librispeech_dev_clean_chapter_heldout_2fold",
        "protocol_type": "closed_set_chapter_held_out_two_fold",
        "created_at_utc": utc_now(),
        "assignment": {
            "number_of_folds": 2,
            "group_column": "source_group",
            "speaker_column": "speaker_label",
            "method": (
                "Per-speaker exhaustive minimum-utterance-imbalance "
                "partition of whole chapters; lexicographic deterministic "
                "tie-break."
            ),
            "speaker_inclusion_rule": (
                "At least two distinct source_group values."
            ),
            "speaker_exclusion_rule": (
                "Speakers with one source_group cannot support closed-set "
                "chapter-held-out evaluation and are excluded."
            ),
        },
        "input_manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
        },
        "output_manifest": {
            "path": str(manifest_output),
            "sha256": file_sha256(manifest_output),
        },
        "protocol_file": {
            "path": str(protocol_output),
            "sha256": file_sha256(protocol_output),
            "columns": PROTOCOL_FIELDS,
        },
        **summary,
    }
    write_json(summary_output, summary_payload)

    print()
    print("=" * 72)
    print("CHAPTER-HELD-OUT PROTOCOL CREATED")
    print("=" * 72)
    print(f"Included speakers : {summary['included_speakers']}")
    print(f"Excluded speakers : {summary['excluded_speakers']}")
    print(f"Included clips    : {summary['included_rows']}")
    print(
        "Fold clips        : "
        f"1={summary['fold_utterance_counts']['1']}, "
        f"2={summary['fold_utterance_counts']['2']}"
    )
    print("Chapter overlap   : 0")
    print(f"Protocol          : {protocol_output}")
    print(f"Summary           : {summary_output}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
