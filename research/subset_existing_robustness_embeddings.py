"""Reuse completed full-protocol robustness embeddings for a subset protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT_PROTOCOL_ID = "librispeech_dev_clean_chapter_heldout_2fold__01ff6bcaaebb"
DEFAULT_SUBSET = (
    PROJECT_ROOT
    / "manifests"
    / "librispeech_dev_clean_chapter_heldout_balanced_subset_protocol.csv"
)
ROOT = PROJECT_ROOT / "research_results" / "robustness_embeddings"
MODEL_IDS = (
    "speechbrain_ecapa",
    "speechbrain_xvector",
    "wavlm_base_plus_sv",
    "unispeech_sat_base_plus_sv",
)
DEFAULT_CONDITIONS = ("short_0p5s", "short_1s", "short_2s", "short_3s")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-file", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--tag", default="full")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_arguments()
    protocol_path = args.protocol_file.resolve()
    protocol_sha = file_sha256(protocol_path)
    protocol_id = f"{protocol_path.stem.removesuffix('_protocol')}__{protocol_sha[:12]}"
    with protocol_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    clip_order = [row["clip_id"] for row in rows]
    label_order = np.asarray([row["speaker_label"] for row in rows])
    fold_order = np.asarray([int(row["fold"]) for row in rows], dtype=np.int16)

    completed = 0
    for model_id in MODEL_IDS:
        for condition in args.conditions:
            source = ROOT / PARENT_PROTOCOL_ID / model_id / condition / args.tag
            destination = ROOT / protocol_id / model_id / condition / args.tag
            if destination.exists() and not args.overwrite:
                if (destination / "metadata.json").exists():
                    completed += 1
                    continue
                raise FileExistsError(f"Incomplete destination exists: {destination}")
            x = np.load(source / "X.npy", allow_pickle=False)
            y = np.load(source / "y.npy", allow_pickle=True).astype(str)
            clip_ids = np.load(source / "clip_ids.npy", allow_pickle=True).astype(str)
            with (source / "metadata.json").open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            lookup = {clip_id: index for index, clip_id in enumerate(clip_ids)}
            missing = [clip_id for clip_id in clip_order if clip_id not in lookup]
            if missing:
                raise ValueError(f"{model_id}/{condition} lacks {len(missing)} subset clips.")
            indices = np.asarray([lookup[clip_id] for clip_id in clip_order], dtype=int)
            selected_x = np.asarray(x[indices], dtype=np.float32)
            selected_y = np.asarray(y[indices], dtype=str)
            if not np.array_equal(selected_y, label_order):
                raise ValueError(f"Label mismatch for {model_id}/{condition}.")

            destination.mkdir(parents=True, exist_ok=True)
            np.save(destination / "X.npy", selected_x)
            np.save(destination / "y.npy", selected_y)
            np.save(destination / "clip_ids.npy", np.asarray(clip_order))
            np.save(destination / "folds.npy", fold_order)
            metadata.update(
                {
                    "schema_version": 2,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "requested_rows": len(rows),
                    "successful_rows": len(rows),
                    "failed_rows": 0,
                    "coverage_percent": 100.0,
                    "number_of_speakers": len(set(label_order)),
                    "number_of_folds": len(set(fold_order.tolist())),
                    "protocol": {
                        "path": str(protocol_path),
                        "sha256": protocol_sha,
                        "protocol_id": protocol_id,
                        "selected_row_count": len(rows),
                        "fold_values": sorted(set(fold_order.tolist())),
                        "speaker_count": len(set(label_order)),
                        "source_group_count": len({row["source_group"] for row in rows}),
                        "source_group_disjoint": True,
                    },
                    "derived_from_full_protocol_artifact": str(source),
                    "output_directory": str(destination),
                }
            )
            metadata["artifact_sha256"] = {
                name: file_sha256(destination / name)
                for name in ("X.npy", "y.npy", "clip_ids.npy", "folds.npy")
            }
            (destination / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            completed += 1
            print(f"Created {model_id}/{condition}: {len(rows)} clips")

    print(f"Subset artifacts completed: {completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
