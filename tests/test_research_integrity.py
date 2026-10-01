import json
import unittest
from pathlib import Path

from model_config import MODEL_CONFIGS
from ui.dashboard_page import Dashboard


ROOT = Path(__file__).resolve().parents[1]


class ProbabilityDistributionTests(unittest.TestCase):
    def test_known_and_unknown_distribution_sums_to_one_hundred(self):
        closed_set = [
            ("A", 50.0),
            ("B", 20.0),
            ("C", 15.0),
            ("D", 10.0),
            ("E", 5.0),
        ]

        displayed = Dashboard.compose_probability_distribution(
            closed_set,
            known_speaker_probability=0.72,
        )

        self.assertEqual(len(displayed), 6)
        self.assertEqual(displayed[-1][0], "Unknown Speaker")
        self.assertAlmostEqual(displayed[-1][1], 28.0, places=6)
        self.assertAlmostEqual(
            sum(probability for _, probability in displayed),
            100.0,
            places=6,
        )

    def test_unknown_row_is_not_hardcoded(self):
        closed_set = [(str(index), 20.0) for index in range(5)]

        known = Dashboard.compose_probability_distribution(
            closed_set,
            known_speaker_probability=0.95,
        )
        unknown = Dashboard.compose_probability_distribution(
            closed_set,
            known_speaker_probability=0.04,
        )

        self.assertAlmostEqual(known[-1][1], 5.0, places=6)
        self.assertAlmostEqual(unknown[-1][1], 96.0, places=6)


class ArtifactIntegrityTests(unittest.TestCase):
    def test_every_enabled_model_has_both_classifiers_and_detector(self):
        for model_id, config in MODEL_CONFIGS.items():
            if not config.get("enabled", True):
                continue

            model_directory = Path(config["model_directory"])
            with self.subTest(model_id=model_id):
                for filename in (
                    "svm_model.pkl",
                    "logistic_model.pkl",
                    "scaler.pkl",
                    "label_encoder.pkl",
                    "open_set_detector.pkl",
                    "open_set_detector_metrics.json",
                ):
                    self.assertTrue(
                        (model_directory / filename).is_file(),
                        f"Missing {filename} for {model_id}",
                    )

    def test_open_set_unknown_speakers_are_disjoint(self):
        for model_id, config in MODEL_CONFIGS.items():
            if not config.get("enabled", True):
                continue

            metrics_path = (
                Path(config["model_directory"])
                / "open_set_detector_metrics.json"
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            train = set(metrics["unknown_train_speakers"])
            validation = set(metrics["unknown_validation_speakers"])
            test = set(metrics["unknown_test_speakers"])

            with self.subTest(model_id=model_id):
                self.assertEqual(metrics["unknown_split_unit"], "speaker_identity")
                self.assertEqual(metrics["unknown_identity_overlap_count"], 0)
                self.assertTrue(train.isdisjoint(validation))
                self.assertTrue(train.isdisjoint(test))
                self.assertTrue(validation.isdisjoint(test))

    def test_quality_filter_report_is_present_and_consistent(self):
        for model_id, config in MODEL_CONFIGS.items():
            if not config.get("enabled", True):
                continue

            report_path = (
                Path(config["embedding_directory"])
                / "quality_filter_report.json"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            before = int(report["samples_before_filter"])
            after = int(report["samples_after_filter"])
            removed = int(report["samples_removed_total"])

            with self.subTest(model_id=model_id):
                self.assertEqual(before - after, removed)
                self.assertGreater(after, 0)
                self.assertTrue(report["filter_applied"])
                self.assertIn("utterance-level", report["important_split_limitation"])


if __name__ == "__main__":
    unittest.main()
