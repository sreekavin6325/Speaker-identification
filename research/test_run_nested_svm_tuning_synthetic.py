"""Small end-to-end synthetic test for nested SVM tuning."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.run_baseline_cv import file_sha256, load_evaluation_protocol
from research.run_nested_svm_tuning import run_model


class NestedSVMTuningSyntheticTest(unittest.TestCase):
    def test_strict_outer_protocol_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.csv"
            protocol_path = root / "protocol.csv"
            embeddings_root = root / "embeddings"
            artifact_directory = (
                embeddings_root / "speechbrain_ecapa" / "synthetic"
            )
            artifact_directory.mkdir(parents=True)
            rows: list[dict[str, object]] = []
            vectors: list[np.ndarray] = []
            labels: list[str] = []
            clip_ids: list[str] = []
            folds: list[int] = []
            random = np.random.default_rng(123)

            for speaker_index in range(3):
                label = f"speaker_{speaker_index}"
                centre = np.zeros(6, dtype=np.float64)
                centre[speaker_index] = 5.0

                for fold in (1, 2):
                    source_group = f"{label}/chapter_{fold}"

                    for utterance_index in range(4):
                        clip_id = (
                            f"{label}-chapter-{fold}-"
                            f"{utterance_index:02d}"
                        )
                        rows.append(
                            {
                                "clip_id": clip_id,
                                "speaker_label": label,
                                "source_group": source_group,
                                "fold": fold,
                            }
                        )
                        vectors.append(
                            centre
                            + random.normal(0.0, 0.15, size=6)
                        )
                        labels.append(label)
                        clip_ids.append(clip_id)
                        folds.append(fold)

            fields = (
                "clip_id",
                "speaker_label",
                "source_group",
                "fold",
            )

            for path in (manifest_path, protocol_path):
                with path.open(
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as output_file:
                    writer = csv.DictWriter(
                        output_file,
                        fieldnames=list(fields),
                    )
                    writer.writeheader()
                    writer.writerows(rows)

            X = np.asarray(vectors, dtype=np.float64)
            y = np.asarray(labels, dtype=str)
            ids = np.asarray(clip_ids, dtype=str)
            fold_array = np.asarray(folds, dtype=np.int64)
            np.save(artifact_directory / "X.npy", X)
            np.save(artifact_directory / "y.npy", y)
            np.save(artifact_directory / "clip_ids.npy", ids)
            np.save(artifact_directory / "folds.npy", fold_array)
            metadata = {
                "schema_version": 1,
                "model_id": "speechbrain_ecapa",
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": file_sha256(manifest_path),
                "successful_rows": len(X),
                "embedding_dimension": X.shape[1],
            }
            with (
                artifact_directory / "metadata.json"
            ).open("w", encoding="utf-8") as output_file:
                json.dump(metadata, output_file)

            protocol = load_evaluation_protocol(protocol_path)
            result = run_model(
                model_id="speechbrain_ecapa",
                tag="synthetic",
                embeddings_root=embeddings_root,
                output_root=root / "results",
                protocol=protocol,
                seed=42,
                inner_folds=3,
                c_values=[0.1, 1.0],
                gamma_values=["scale", 0.01],
                n_jobs=1,
                overwrite=False,
            )
            output_directory = Path(
                str(result["output_directory"])
            )

            self.assertTrue(
                (output_directory / "run_metadata.json").is_file()
            )
            self.assertTrue(
                (output_directory / "class_labels.npy").is_file()
            )
            self.assertTrue(
                (output_directory / "grid_search_results.csv").is_file()
            )

            with (
                output_directory / "fold_metrics.csv"
            ).open("r", encoding="utf-8", newline="") as input_file:
                fold_rows = list(csv.DictReader(input_file))

            self.assertEqual(len(fold_rows), 2)
            self.assertEqual(
                {row["outer_fold"] for row in fold_rows},
                {"1", "2"},
            )

            with (
                output_directory / "all_predictions.csv"
            ).open("r", encoding="utf-8", newline="") as input_file:
                prediction_rows = list(csv.DictReader(input_file))

            self.assertEqual(len(prediction_rows), len(X))
            self.assertEqual(
                len({row["clip_id"] for row in prediction_rows}),
                len(X),
            )

            with (
                output_directory / "leakage_audit.csv"
            ).open("r", encoding="utf-8", newline="") as input_file:
                leakage_rows = list(csv.DictReader(input_file))

            self.assertTrue(
                all(
                    row["clip_overlap_count"] == "0"
                    and row["source_group_overlap_count"] == "0"
                    and row["outer_test_used_in_tuning"] == "False"
                    for row in leakage_rows
                )
            )

            with (
                output_directory / "run_metadata.json"
            ).open("r", encoding="utf-8") as input_file:
                run_metadata = json.load(input_file)

            self.assertEqual(
                run_metadata["tuning"]["selection_metric"],
                "f1_macro",
            )
            self.assertFalse(
                run_metadata["tuning"]["outer_test_used_in_tuning"]
            )
            self.assertTrue(
                run_metadata["protocol"][
                    "source_group_disjoint_enforced"
                ]
            )

            for fold in (1, 2):
                fold_directory = (
                    output_directory / f"fold_{fold:02d}"
                )
                with (
                    fold_directory / "cv_results.json"
                ).open("r", encoding="utf-8") as input_file:
                    cv_results = json.load(input_file)
                self.assertEqual(len(cv_results["params"]), 6)


if __name__ == "__main__":
    unittest.main()
