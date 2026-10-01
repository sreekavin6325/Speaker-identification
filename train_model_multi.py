# ==========================================================
# train_model_multi.py
# Train a separate classifier for each embedding model
# ==========================================================

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)
from sklearn.svm import SVC

from model_config import (
    MODEL_CONFIGS,
    create_model_directories,
    get_model_config,
)


# ==========================================================
# Arguments
# ==========================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Train an SVM classifier for one speaker "
            "embedding model."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_CONFIGS.keys())
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20
    )

    parser.add_argument(
        "--c-value",
        type=float,
        default=10.0
    )

    parser.add_argument(
        "--kernel",
        choices=["linear", "rbf"],
        default="rbf"
    )

    return parser.parse_args()


# ==========================================================
# Load embeddings
# ==========================================================

def load_embeddings(
    model_id,
    minimum_duration_seconds=0.25,
    remove_exact_duplicates=True,
):

    config = get_model_config(model_id)

    embedding_directory = Path(
        config["embedding_directory"]
    )

    x_path = embedding_directory / "X.npy"
    y_path = embedding_directory / "y.npy"
    paths_path = embedding_directory / "audio_paths.npy"

    if not x_path.exists():
        raise FileNotFoundError(
            f"Embeddings were not found:\n{x_path}"
        )

    if not y_path.exists():
        raise FileNotFoundError(
            f"Labels were not found:\n{y_path}"
        )

    X = np.load(x_path)
    y = np.load(y_path)

    if X.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional embedding "
            f"array, received {X.shape}."
        )

    if y.ndim != 1:
        y = y.reshape(-1)

    if len(X) != len(y):
        raise ValueError(
            "Embedding and label counts do not match."
        )

    if len(X) == 0:
        raise ValueError(
            "The embedding array is empty."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Embeddings contain invalid values."
        )

    quality_report = {
        "model_id": model_id,
        "samples_before_filter": int(len(X)),
        "minimum_duration_seconds": float(minimum_duration_seconds),
        "short_files_removed": 0,
        "exact_duplicate_files_removed": 0,
        "missing_or_unreadable_files_removed": 0,
        "filter_applied": False,
        "important_split_limitation": (
            "Original source-recording/session identifiers are unavailable. "
            "The train/test split remains utterance-level and may be optimistic "
            "when adjacent clips came from the same source recording."
        ),
    }

    if paths_path.exists():
        audio_paths = np.load(paths_path, allow_pickle=True).astype(str)
        if len(audio_paths) != len(X):
            raise ValueError(
                "Embedding and audio-path counts do not match."
            )

        keep = np.ones(len(X), dtype=bool)
        seen_hashes = set()
        project_root = Path(__file__).resolve().parent

        for index, raw_path in enumerate(audio_paths):
            audio_path = Path(raw_path)
            if not audio_path.is_absolute():
                audio_path = project_root / audio_path
            try:
                duration = float(sf.info(str(audio_path)).duration)
                if duration < float(minimum_duration_seconds):
                    keep[index] = False
                    quality_report["short_files_removed"] += 1
                    continue

                if remove_exact_duplicates:
                    digest = hashlib.sha256(audio_path.read_bytes()).digest()
                    if digest in seen_hashes:
                        keep[index] = False
                        quality_report["exact_duplicate_files_removed"] += 1
                        continue
                    seen_hashes.add(digest)
            except (OSError, RuntimeError, ValueError):
                keep[index] = False
                quality_report["missing_or_unreadable_files_removed"] += 1

        X = X[keep]
        y = y[keep]
        quality_report["filter_applied"] = True

    quality_report["samples_after_filter"] = int(len(X))
    quality_report["samples_removed_total"] = (
        quality_report["samples_before_filter"]
        - quality_report["samples_after_filter"]
    )
    (embedding_directory / "quality_filter_report.json").write_text(
        json.dumps(quality_report, indent=2),
        encoding="utf-8",
    )
    print(
        "Quality filter: "
        f"{quality_report['samples_before_filter']} -> "
        f"{quality_report['samples_after_filter']} samples "
        f"({quality_report['samples_removed_total']} removed)."
    )

    return X.astype(np.float32), y.astype(str)


# ==========================================================
# Train model
# ==========================================================

def train_model(
    model_id,
    test_size=0.20,
    c_value=10.0,
    kernel="rbf",
):

    create_model_directories()

    config = get_model_config(model_id)

    model_directory = Path(
        config["model_directory"]
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("MULTI-MODEL CLASSIFIER TRAINING")
    print("=" * 70)
    print(f"Embedding model: {config['display_name']}")
    print(f"Kernel         : {kernel}")
    print(f"C value        : {c_value}")
    print("=" * 70)

    X, y = load_embeddings(model_id)

    print(f"Embeddings shape: {X.shape}")
    print(f"Labels shape    : {y.shape}")

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    print()
    print("Speakers:")

    for index, speaker in enumerate(
        label_encoder.classes_
    ):
        speaker_count = int(
            np.sum(y == speaker)
        )

        print(
            f"  {index}: {speaker} "
            f"({speaker_count} samples)"
        )

        if speaker_count < 5:
            raise ValueError(
                f"Speaker '{speaker}' has fewer than "
                f"five samples. Calibration requires "
                f"at least five samples per speaker."
            )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=42,
        stratify=y_encoded,
    )

    # Fit the scaler using only training data.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print()
    print(f"Training samples: {len(X_train_scaled)}")
    print(f"Testing samples : {len(X_test_scaled)}")

    # Base SVM without the deprecated probability=True.
    base_svm = SVC(
        kernel=kernel,
        C=c_value,
        gamma="scale",
        probability=False,
        class_weight=None,
        random_state=42,
    )

    # Produce calibrated per-speaker probabilities.
    classifier = CalibratedClassifierCV(
        estimator=base_svm,
        method="sigmoid",
        cv=5,
        ensemble=False,
        n_jobs=-1,
    )

    print()
    print("Training calibrated SVM...")

    start_time = time.perf_counter()

    classifier.fit(
        X_train_scaled,
        y_train
    )

    training_seconds = (
        time.perf_counter() - start_time
    )

    print(
        f"Training completed in "
        f"{training_seconds:.2f} seconds."
    )

    y_prediction = classifier.predict(
        X_test_scaled
    )
    training_accuracy = float(
        classifier.score(X_train_scaled, y_train)
    )

    probabilities = classifier.predict_proba(
        X_test_scaled
    )

    probability_sums = probabilities.sum(axis=1)

    if not np.allclose(
        probability_sums,
        1.0,
        atol=1e-5
    ):
        raise RuntimeError(
            "Classifier probabilities do not sum to one."
        )

    accuracy = accuracy_score(
        y_test,
        y_prediction
    )

    precision = precision_score(
        y_test,
        y_prediction,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_prediction,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_prediction,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        y_prediction,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        y_prediction,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_prediction
    )

    print()
    print("=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)
    print(f"Accuracy          : {accuracy * 100:.2f}%")
    print(f"Weighted precision: {precision:.4f}")
    print(f"Weighted recall   : {recall:.4f}")
    print(f"Weighted F1       : {weighted_f1:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print()
    print("Classification Report")
    print(report)
    print("Confusion Matrix")
    print(matrix)

    svm_path = model_directory / "svm_model.pkl"
    scaler_path = model_directory / "scaler.pkl"
    encoder_path = (
        model_directory / "label_encoder.pkl"
    )

    joblib.dump(
        classifier,
        svm_path
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    joblib.dump(
        label_encoder,
        encoder_path
    )

    np.save(
        model_directory / "confusion_matrix.npy",
        matrix
    )

    with (
        model_directory /
        "classification_report.txt"
    ).open(
        "w",
        encoding="utf-8"
    ) as report_file:

        report_file.write(report)

    metrics = {
        "model_id": model_id,
        "display_name": config["display_name"],
        "provider": config["provider"],
        "architecture": config["architecture"],
        "embedding_dimension": int(X.shape[1]),
        "number_of_samples": int(len(X)),
        "number_of_speakers": int(
            len(label_encoder.classes_)
        ),
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "kernel": kernel,
        "c_value": c_value,
        "split_unit": "utterance",
        "split_limitation": (
            "Source recording/session IDs are unavailable for the original "
            "five-speaker corpus; adjacent fragments may cross partitions."
        ),
        "training_accuracy": training_accuracy,
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(weighted_f1),
        "f1_macro": float(macro_f1),
        "training_seconds": float(
            training_seconds
        ),
        "speakers": (
            label_encoder.classes_.tolist()
        ),
    }

    with (
        model_directory / "metrics.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as metrics_file:

        json.dump(
            metrics,
            metrics_file,
            indent=4
        )

    print()
    print("=" * 70)
    print("TRAINED FILES SAVED")
    print("=" * 70)
    print(f"Classifier   : {svm_path}")
    print(f"Scaler       : {scaler_path}")
    print(f"Label encoder: {encoder_path}")
    print("=" * 70)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    arguments = parse_arguments()

    train_model(
        model_id=arguments.model,
        test_size=arguments.test_size,
        c_value=arguments.c_value,
        kernel=arguments.kernel,
    )
