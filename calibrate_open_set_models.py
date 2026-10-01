"""Build unknown-speaker detectors for every trained embedding model.

Known examples come from the enrolled five-speaker corpus.  LibriSpeech
embeddings provide external, non-enrolled speakers.  The saved detector uses
only similarity features, so it can be applied beside the existing classifier
without retraining the pre-trained embedding network.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from model_config import MODEL_CONFIGS
from train_model_multi import load_embeddings


PROJECT_ROOT = Path(__file__).resolve().parent
UNKNOWN_ROOT = (
    PROJECT_ROOT
    / "research_results"
    / "embeddings"
    / "librispeech_dev_clean"
)
FEATURE_NAMES = (
    "maximum_centroid_cosine",
    "centroid_margin",
    "maximum_reference_cosine",
    "top3_reference_cosine_mean",
    "top10_reference_cosine_mean",
    "top1_minus_top3_reference_margin",
)


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def similarity_features(queries, centroids, references, batch_size=256):
    queries = normalize_rows(queries)
    centroid_scores = queries @ centroids.T
    sorted_centroids = np.sort(centroid_scores, axis=1)
    parts = []

    for start in range(0, len(queries), batch_size):
        query_batch = queries[start : start + batch_size]
        reference_scores = query_batch @ references.T
        top_count = min(10, reference_scores.shape[1])
        top_scores = np.partition(
            reference_scores,
            -top_count,
            axis=1,
        )[:, -top_count:]
        top_scores.sort(axis=1)
        top3_count = min(3, top_count)
        top3_mean = top_scores[:, -top3_count:].mean(axis=1)

        centroid_batch = sorted_centroids[start : start + len(query_batch)]
        parts.append(
            np.column_stack(
                [
                    centroid_batch[:, -1],
                    centroid_batch[:, -1] - centroid_batch[:, -2],
                    top_scores[:, -1],
                    top3_mean,
                    top_scores.mean(axis=1),
                    top_scores[:, -1] - top3_mean,
                ]
            )
        )

    return np.vstack(parts).astype(np.float32)


def select_balanced_threshold(known_scores, unknown_scores):
    candidates = np.linspace(0.0, 1.0, 4001)
    false_accept = np.array(
        [(unknown_scores >= value).mean() for value in candidates]
    )
    false_reject = np.array(
        [(known_scores < value).mean() for value in candidates]
    )
    index = int(np.argmin(np.abs(false_accept - false_reject)))
    return (
        float(candidates[index]),
        float(false_accept[index]),
        float(false_reject[index]),
    )


def calibrate_model(model_id, config):
    embedding_directory = Path(config["embedding_directory"])
    model_directory = Path(config["model_directory"])
    known_x_path = embedding_directory / "X.npy"
    known_y_path = embedding_directory / "y.npy"
    unknown_x_path = UNKNOWN_ROOT / model_id / "full" / "X.npy"
    unknown_y_path = UNKNOWN_ROOT / model_id / "full" / "y.npy"

    for required in (
        known_x_path,
        known_y_path,
        unknown_x_path,
        unknown_y_path,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Required calibration data is missing: {required}")

    known_x, known_y = load_embeddings(model_id)
    unknown_x = np.load(unknown_x_path).astype(np.float32)
    unknown_y = np.load(unknown_y_path).astype(str)
    if len(unknown_x) != len(unknown_y):
        raise ValueError(
            f"Unknown embedding/label length mismatch for {model_id}: "
            f"{len(unknown_x)} embeddings vs {len(unknown_y)} labels."
        )
    known_x = normalize_rows(known_x)
    unknown_x = normalize_rows(unknown_x)

    all_indices = np.arange(len(known_x))
    reference_indices, remaining_indices = train_test_split(
        all_indices,
        test_size=0.50,
        random_state=42,
        stratify=known_y,
    )
    detector_train_indices, validation_and_test = train_test_split(
        remaining_indices,
        test_size=0.60,
        random_state=43,
        stratify=known_y[remaining_indices],
    )
    validation_indices, test_indices = train_test_split(
        validation_and_test,
        test_size=0.50,
        random_state=44,
        stratify=known_y[validation_and_test],
    )

    # Split unknown data by SPEAKER ID, not by utterance.  This prevents the
    # same unknown identity from appearing in detector training, threshold
    # selection and final evaluation.
    unknown_speakers = np.unique(unknown_y)
    if len(unknown_speakers) < 6:
        raise ValueError(
            "At least six unknown speaker identities are required for "
            "speaker-disjoint train/validation/test calibration."
        )

    unknown_train_speakers, unknown_validation_and_test_speakers = (
        train_test_split(
            unknown_speakers,
            test_size=0.50,
            random_state=42,
        )
    )
    unknown_validation_speakers, unknown_test_speakers = train_test_split(
        unknown_validation_and_test_speakers,
        test_size=0.50,
        random_state=43,
    )

    unknown_train = np.flatnonzero(
        np.isin(unknown_y, unknown_train_speakers)
    )
    unknown_validation = np.flatnonzero(
        np.isin(unknown_y, unknown_validation_speakers)
    )
    unknown_test = np.flatnonzero(
        np.isin(unknown_y, unknown_test_speakers)
    )

    overlap_pairs = {
        "train_validation": set(unknown_train_speakers)
        & set(unknown_validation_speakers),
        "train_test": set(unknown_train_speakers)
        & set(unknown_test_speakers),
        "validation_test": set(unknown_validation_speakers)
        & set(unknown_test_speakers),
    }
    if any(overlap_pairs.values()):
        raise RuntimeError(
            "Unknown-speaker identity leakage was detected during calibration."
        )

    references = known_x[reference_indices]
    classes = np.unique(known_y)
    centroids = np.stack(
        [
            references[known_y[reference_indices] == speaker].mean(axis=0)
            for speaker in classes
        ]
    )
    centroids = normalize_rows(centroids)

    known_train_features = similarity_features(
        known_x[detector_train_indices], centroids, references
    )
    unknown_train_features = similarity_features(
        unknown_x[unknown_train], centroids, references
    )
    detector = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    detector.fit(
        np.vstack([known_train_features, unknown_train_features]),
        np.concatenate(
            [
                np.ones(len(known_train_features), dtype=int),
                np.zeros(len(unknown_train_features), dtype=int),
            ]
        ),
    )

    validation_known_scores = detector.predict_proba(
        similarity_features(known_x[validation_indices], centroids, references)
    )[:, 1]
    validation_unknown_scores = detector.predict_proba(
        similarity_features(unknown_x[unknown_validation], centroids, references)
    )[:, 1]
    threshold, validation_far, validation_frr = select_balanced_threshold(
        validation_known_scores,
        validation_unknown_scores,
    )

    test_known_scores = detector.predict_proba(
        similarity_features(known_x[test_indices], centroids, references)
    )[:, 1]
    test_unknown_scores = detector.predict_proba(
        similarity_features(unknown_x[unknown_test], centroids, references)
    )[:, 1]
    test_labels = np.concatenate(
        [
            np.ones(len(test_known_scores), dtype=int),
            np.zeros(len(test_unknown_scores), dtype=int),
        ]
    )
    test_scores = np.concatenate([test_known_scores, test_unknown_scores])
    test_predictions = (test_scores >= threshold).astype(int)
    test_far = float((test_unknown_scores >= threshold).mean())
    test_frr = float((test_known_scores < threshold).mean())

    artifact = {
        "schema_version": 1,
        "model_id": model_id,
        "detector": detector,
        "threshold": threshold,
        "centroids": centroids.astype(np.float32),
        "reference_embeddings": references.astype(np.float32),
        "speaker_names": classes.tolist(),
        "feature_names": list(FEATURE_NAMES),
        "calibration_source": "enrolled five-speaker embeddings vs LibriSpeech dev-clean",
        "unknown_split_unit": "speaker_identity",
        "unknown_train_speakers": sorted(unknown_train_speakers.tolist()),
        "unknown_validation_speakers": sorted(
            unknown_validation_speakers.tolist()
        ),
        "unknown_test_speakers": sorted(unknown_test_speakers.tolist()),
        "metrics": {
            "validation_far": validation_far,
            "validation_frr": validation_frr,
            "test_far": test_far,
            "test_frr": test_frr,
            "test_balanced_accuracy": float(
                balanced_accuracy_score(test_labels, test_predictions)
            ),
            "test_roc_auc": float(roc_auc_score(test_labels, test_scores)),
            "known_test_samples": int(len(test_known_scores)),
            "unknown_test_samples": int(len(test_unknown_scores)),
            "unknown_train_speaker_count": int(
                len(unknown_train_speakers)
            ),
            "unknown_validation_speaker_count": int(
                len(unknown_validation_speakers)
            ),
            "unknown_test_speaker_count": int(
                len(unknown_test_speakers)
            ),
            "unknown_identity_overlap_count": 0,
        },
    }

    model_directory.mkdir(parents=True, exist_ok=True)
    artifact_path = model_directory / "open_set_detector.pkl"
    metadata_path = model_directory / "open_set_detector_metrics.json"
    joblib.dump(artifact, artifact_path, compress=3)
    metadata_path.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "threshold": threshold,
                **artifact["metrics"],
                "calibration_source": artifact["calibration_source"],
                "unknown_split_unit": artifact["unknown_split_unit"],
                "unknown_train_speakers": artifact[
                    "unknown_train_speakers"
                ],
                "unknown_validation_speakers": artifact[
                    "unknown_validation_speakers"
                ],
                "unknown_test_speakers": artifact[
                    "unknown_test_speakers"
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"{config['display_name']}: threshold={threshold:.3f}, "
        f"test FAR={test_far * 100:.2f}%, test FRR={test_frr * 100:.2f}%, "
        f"AUC={artifact['metrics']['test_roc_auc']:.3f}"
    )
    return artifact_path


def main():
    print("Building unknown-speaker detectors...")
    completed = 0
    for model_id, config in MODEL_CONFIGS.items():
        calibrate_model(model_id, config)
        completed += 1
    print(f"Completed {completed} model-specific open-set detectors.")


if __name__ == "__main__":
    main()
