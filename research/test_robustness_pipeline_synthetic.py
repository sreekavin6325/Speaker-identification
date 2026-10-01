"""Small end-to-end schema test for the robustness benchmark."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from research.run_robustness_benchmark import run_model


def save_artifacts(
    directory: Path,
    X: np.ndarray,
    labels: np.ndarray,
    clip_ids: np.ndarray,
    folds: np.ndarray,
    metadata: dict[str, object],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "X.npy", X.astype(np.float32))
    np.save(directory / "y.npy", labels.astype(str))
    np.save(directory / "clip_ids.npy", clip_ids.astype(str))
    np.save(directory / "folds.npy", folds.astype(np.int16))
    with (directory / "metadata.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output)


def main() -> None:
    rng = np.random.default_rng(42)
    model_id = "speechbrain_ecapa"
    condition = "short_1s"
    tag = "synthetic"
    protocol_id = "synthetic_protocol__0123456789ab"
    protocol_hash = "0123456789abcdef"
    speakers = ["speaker_a", "speaker_b", "speaker_c", "speaker_d"]
    rows: list[dict[str, str]] = []
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    clip_ids: list[str] = []
    protocol_folds: list[int] = []
    dimension = 12

    for speaker_index, speaker in enumerate(speakers):
        center = np.zeros(dimension, dtype=np.float32)
        center[speaker_index] = 5.0
        for fold in (1, 2):
            group = f"{speaker}/chapter_{fold}"
            for utterance in range(3):
                clip_id = f"{speaker}-{fold}-{utterance}"
                rows.append(
                    {
                        "clip_id": clip_id,
                        "speaker_label": speaker,
                        "source_group": group,
                        "fold": str(fold),
                    }
                )
                vectors.append(center + rng.normal(0, 0.08, dimension))
                labels.append(speaker)
                clip_ids.append(clip_id)
                protocol_folds.append(fold)

    X = np.asarray(vectors, dtype=np.float32)
    y = np.asarray(labels, dtype=str)
    ids = np.asarray(clip_ids, dtype=str)
    strict_folds = np.asarray(protocol_folds, dtype=np.int16)
    condition_X = X + rng.normal(0, 0.03, X.shape).astype(np.float32)
    protocol = {
        "protocol_id": protocol_id,
        "sha256": protocol_hash,
        "ordered_assignment_sha256": "synthetic-assignment",
        "full_row_count": len(rows),
        "selected_row_count": len(rows),
        "fold_values": [1, 2],
        "speaker_count": len(speakers),
        "source_group_count": len(speakers) * 2,
        "source_group_disjoint": True,
    }

    with tempfile.TemporaryDirectory(prefix="robustness_test_") as temporary:
        root = Path(temporary)
        clean_root = root / "clean"
        condition_root = root / "conditions"
        output_root = root / "results"
        # Clean artifacts deliberately carry unrelated original manifest folds;
        # the external strict protocol must replace them.
        original_folds = np.asarray(
            [(index % 5) + 1 for index in range(len(X))], dtype=np.int16
        )
        save_artifacts(
            clean_root / model_id / tag,
            X,
            y,
            ids,
            original_folds,
            {"experiment": "synthetic_clean"},
        )
        save_artifacts(
            condition_root / protocol_id / model_id / condition / tag,
            condition_X,
            y,
            ids,
            strict_folds,
            {
                "experiment": "robustness_embedding_extraction",
                "condition": condition,
                "protocol": {"sha256": protocol_hash},
            },
        )

        result = run_model(
            model_id=model_id,
            conditions=[condition],
            protocol_rows=rows,
            protocol=protocol,
            clean_root=clean_root,
            condition_root=condition_root,
            output_root=output_root,
            tag=tag,
            c_value=1.0,
            max_iter=1000,
            seed=42,
            overwrite=True,
        )
        output_directory = Path(str(result["output_directory"]))
        required = {
            "fold_metrics.csv",
            "aggregate_metrics.csv",
            "aggregate_metrics.json",
            "all_predictions.csv",
            "leakage_audit.csv",
            "run_metadata.json",
        }
        missing = sorted(
            filename
            for filename in required
            if not (output_directory / filename).is_file()
        )
        if missing:
            raise AssertionError("Missing result files: " + ", ".join(missing))
        with (output_directory / "aggregate_metrics.json").open(
            "r", encoding="utf-8"
        ) as input_file:
            aggregate_rows = json.load(input_file)
        if [row["condition"] for row in aggregate_rows] != [
            "clean_full",
            condition,
        ]:
            raise AssertionError("Condition order is incorrect.")
        if any(float(row["accuracy_mean"]) < 0.99 for row in aggregate_rows):
            raise AssertionError("Synthetic separable data was not classified.")

    print("Synthetic robustness pipeline test passed.")


if __name__ == "__main__":
    main()
