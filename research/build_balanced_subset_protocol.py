"""Create a deterministic balanced subset of an existing evaluation protocol.

The subset keeps the original fold assignments and therefore preserves the
source-group separation guarantees of the parent protocol.  It is intended for
computationally expensive audio-domain robustness and ablation experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "manifests"
    / "librispeech_dev_clean_chapter_heldout_2fold_protocol.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "manifests"
    / "librispeech_dev_clean_chapter_heldout_balanced_subset_protocol.csv"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clips-per-speaker-fold", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.clips_per_speaker_fold < 1:
        raise ValueError("--clips-per-speaker-fold must be positive.")
    with args.input.resolve().open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The input protocol is empty.")

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["speaker_label"], row["fold"])].append(row)

    rng = np.random.default_rng(args.seed)
    selected: list[dict[str, str]] = []
    for key in sorted(groups):
        candidates = sorted(groups[key], key=lambda row: row["clip_id"])
        if len(candidates) < args.clips_per_speaker_fold:
            raise ValueError(
                f"{key} contains {len(candidates)} clips, fewer than requested "
                f"{args.clips_per_speaker_fold}."
            )
        indices = sorted(
            rng.choice(
                len(candidates),
                size=args.clips_per_speaker_fold,
                replace=False,
            ).tolist()
        )
        selected.extend(candidates[index] for index in indices)

    selected.sort(key=lambda row: (int(row["fold"]), row["speaker_label"], row["clip_id"]))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    speakers = len({row["speaker_label"] for row in selected})
    folds = sorted({int(row["fold"]) for row in selected})
    print("=" * 72)
    print("BALANCED ROBUSTNESS SUBSET CREATED")
    print("=" * 72)
    print(f"Rows       : {len(selected)}")
    print(f"Speakers   : {speakers}")
    print(f"Folds      : {folds}")
    print(f"Per cell   : {args.clips_per_speaker_fold}")
    print(f"Seed       : {args.seed}")
    print(f"SHA-256    : {digest}")
    print(f"Output     : {output}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
