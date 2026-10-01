"""Train Logistic Regression identity classifiers for every embedding model."""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from model_config import MODEL_CONFIGS, get_model_config
from train_model_multi import load_embeddings


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train Logistic Regression speaker classifiers from existing "
            "pre-computed embeddings."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["all", *MODEL_CONFIGS.keys()],
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=5000)
    return parser.parse_args()


def train_logistic_classifier(
    model_id,
    test_size=0.20,
    c_value=1.0,
    max_iter=5000,
):
    config = get_model_config(model_id)
    model_directory = Path(config["model_directory"])
    model_directory.mkdir(parents=True, exist_ok=True)

    X, y = load_embeddings(model_id)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=42,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifier = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=max_iter,
        class_weight=None,
        random_state=42,
    )

    print()
    print("=" * 72)
    print("LOGISTIC REGRESSION CLASSIFIER TRAINING")
    print("=" * 72)
    print(f"Embedding model : {config['display_name']}")
    print(f"Embeddings      : {X.shape}")
    print(f"Speakers        : {len(label_encoder.classes_)}")
    print(f"C value         : {c_value}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    start_time = time.perf_counter()
    classifier.fit(X_train_scaled, y_train)
    training_seconds = time.perf_counter() - start_time

    train_prediction = classifier.predict(X_train_scaled)
    test_prediction = classifier.predict(X_test_scaled)
    test_probabilities = classifier.predict_proba(X_test_scaled)
    if not np.allclose(test_probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Logistic Regression probabilities do not sum to one.")

    train_accuracy = accuracy_score(y_train, train_prediction)
    test_accuracy = accuracy_score(y_test, test_prediction)
    precision = precision_score(
        y_test, test_prediction, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, test_prediction, average="weighted", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, test_prediction, average="weighted", zero_division=0
    )
    macro_f1 = f1_score(
        y_test, test_prediction, average="macro", zero_division=0
    )
    report = classification_report(
        y_test,
        test_prediction,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, test_prediction)

    artifact_paths = {
        "classifier": model_directory / "logistic_model.pkl",
        "scaler": model_directory / "logistic_scaler.pkl",
        "label_encoder": model_directory / "logistic_label_encoder.pkl",
        "metrics": model_directory / "logistic_metrics.json",
        "report": model_directory / "logistic_classification_report.txt",
        "confusion_matrix": model_directory / "logistic_confusion_matrix.npy",
    }

    joblib.dump(classifier, artifact_paths["classifier"])
    joblib.dump(scaler, artifact_paths["scaler"])
    joblib.dump(label_encoder, artifact_paths["label_encoder"])
    np.save(artifact_paths["confusion_matrix"], matrix)
    artifact_paths["report"].write_text(report, encoding="utf-8")

    metrics = {
        "model_id": model_id,
        "display_name": config["display_name"],
        "classifier_id": "logistic_regression",
        "classifier_name": "Logistic Regression",
        "embedding_dimension": int(X.shape[1]),
        "number_of_samples": int(len(X)),
        "number_of_speakers": int(len(label_encoder.classes_)),
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "c_value": float(c_value),
        "max_iter": int(max_iter),
        "split_unit": "utterance",
        "split_limitation": (
            "Source recording/session IDs are unavailable for the original "
            "five-speaker corpus; adjacent fragments may cross partitions."
        ),
        "training_accuracy": float(train_accuracy),
        "testing_accuracy": float(test_accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(weighted_f1),
        "f1_macro": float(macro_f1),
        "training_seconds": float(training_seconds),
        "speakers": label_encoder.classes_.tolist(),
    }
    artifact_paths["metrics"].write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing accuracy : {test_accuracy * 100:.2f}%")
    print(f"Weighted F1      : {weighted_f1:.4f}")
    print(f"Training time    : {training_seconds:.2f} seconds")
    print(f"Saved classifier : {artifact_paths['classifier']}")
    return metrics


def main():
    arguments = parse_arguments()
    model_ids = (
        list(MODEL_CONFIGS.keys())
        if arguments.model == "all"
        else [arguments.model]
    )

    failures = []
    for model_id in model_ids:
        try:
            train_logistic_classifier(
                model_id=model_id,
                test_size=arguments.test_size,
                c_value=arguments.c_value,
                max_iter=arguments.max_iter,
            )
        except Exception as error:
            failures.append((model_id, str(error)))
            print(f"FAILED {model_id}: {error}")

    if failures:
        failure_text = "\n".join(
            f"{model_id}: {error}" for model_id, error in failures
        )
        raise SystemExit(f"\nLogistic training failures:\n{failure_text}")

    print()
    print(f"Completed Logistic Regression training for {len(model_ids)} models.")


if __name__ == "__main__":
    main()
