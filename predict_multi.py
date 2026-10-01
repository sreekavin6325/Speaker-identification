# ==========================================================
# predict_multi.py
# Unified prediction with selectable embedding models
# ==========================================================

import time
from pathlib import Path

import joblib
import numpy as np

from embedding_extractors import (
    clear_model_cache,
    extract_embedding,
)
from model_config import (
    DEFAULT_MODEL_ID,
    MODEL_CONFIGS,
    get_model_config,
)


# Identity-classifier choices.  The embedding model and the classifier are
# deliberately separate so the same speaker embedding can be evaluated with
# more than one classifier without extracting the audio again.
DEFAULT_CLASSIFIER_ID = "rbf_svm"
CLASSIFIER_CONFIGS = {
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "classifier_file": "logistic_model.pkl",
        "scaler_file": "logistic_scaler.pkl",
        "label_encoder_file": "logistic_label_encoder.pkl",
        "training_command": "train_logistic_multi.py",
    },
    "rbf_svm": {
        "display_name": "Calibrated RBF SVM",
        "classifier_file": "svm_model.pkl",
        "scaler_file": "scaler.pkl",
        "label_encoder_file": "label_encoder.pkl",
        "training_command": "train_model_multi.py",
    },
}


# Trained identity classifier, scaler and encoder cache
_CLASSIFIER_CACHE = {}
_OPEN_SET_CACHE = {}
_OPEN_SET_GATE_CACHE = {}


# ==========================================================
# Artifact paths
# ==========================================================

def get_classifier_config(classifier_id):
    """Return a validated identity-classifier configuration."""

    if classifier_id not in CLASSIFIER_CONFIGS:
        available = ", ".join(CLASSIFIER_CONFIGS)
        raise ValueError(
            f"Unknown classifier: {classifier_id}. "
            f"Available classifiers: {available}"
        )

    return CLASSIFIER_CONFIGS[classifier_id]


def get_artifact_paths(
    model_id,
    classifier_id=DEFAULT_CLASSIFIER_ID,
):

    config = get_model_config(model_id)

    model_directory = Path(
        config["model_directory"]
    )

    classifier_config = get_classifier_config(
        classifier_id
    )

    return {
        "classifier": (
            model_directory /
            classifier_config["classifier_file"]
        ),
        "scaler": (
            model_directory /
            classifier_config["scaler_file"]
        ),
        "label_encoder": (
            model_directory /
            classifier_config["label_encoder_file"]
        ),
    }


def get_open_set_artifact_path(model_id):
    """Return the optional model-specific unknown-speaker detector path."""

    config = get_model_config(model_id)
    return Path(config["model_directory"]) / "open_set_detector.pkl"


# ==========================================================
# Check model availability
# ==========================================================

def is_model_trained(model_id, classifier_id=None):

    if classifier_id is None:
        return any(
            is_model_trained(model_id, candidate_id)
            for candidate_id in CLASSIFIER_CONFIGS
        )

    paths = get_artifact_paths(
        model_id,
        classifier_id,
    )

    return all(
        path.exists()
        for path in paths.values()
    )


def get_missing_artifacts(
    model_id,
    classifier_id=DEFAULT_CLASSIFIER_ID,
):

    paths = get_artifact_paths(
        model_id,
        classifier_id,
    )

    return [
        path
        for path in paths.values()
        if not path.exists()
    ]


def get_available_classifiers(model_id):
    """Return the classifier choices trained for an embedding model."""

    return [
        {
            "classifier_id": classifier_id,
            "display_name": config["display_name"],
        }
        for classifier_id, config in CLASSIFIER_CONFIGS.items()
        if is_model_trained(model_id, classifier_id)
    ]


def get_trained_models(classifier_id=None):

    trained_models = []

    for model_id, config in MODEL_CONFIGS.items():

        if is_model_trained(model_id, classifier_id):

            trained_models.append({
                "model_id": model_id,
                "display_name": config["display_name"],
                "provider": config["provider"],
                "architecture": config["architecture"],
            })

    return trained_models


# ==========================================================
# Load classifier files
# ==========================================================

def load_classifier_artifacts(
    model_id,
    classifier_id=DEFAULT_CLASSIFIER_ID,
):

    cache_key = (model_id, classifier_id)

    if cache_key in _CLASSIFIER_CACHE:
        return _CLASSIFIER_CACHE[cache_key]

    config = get_model_config(model_id)
    classifier_config = get_classifier_config(
        classifier_id
    )

    missing_files = get_missing_artifacts(
        model_id,
        classifier_id,
    )

    if missing_files:

        missing_text = "\n".join(
            str(path)
            for path in missing_files
        )

        raise FileNotFoundError(
            f"The selected model is not trained:\n"
            f"{config['display_name']}\n\n"
            f"Missing files:\n{missing_text}\n\n"
            f"Extract its embeddings and run:\n"
            f"python "
            f"{classifier_config['training_command']} "
            f"--model {model_id}"
        )

    paths = get_artifact_paths(
        model_id,
        classifier_id,
    )

    print(
        f"Loading {classifier_config['display_name']} for "
        f"{config['display_name']}..."
    )

    try:
        classifier = joblib.load(
            paths["classifier"]
        )

        scaler = joblib.load(
            paths["scaler"]
        )

        label_encoder = joblib.load(
            paths["label_encoder"]
        )

    except Exception as error:

        raise RuntimeError(
            f"Unable to load the trained files for "
            f"{config['display_name']}.\n\n"
            f"{error}"
        ) from error

    if not hasattr(classifier, "predict_proba"):

        raise TypeError(
            f"The classifier for "
            f"{config['display_name']} does not support "
            f"predict_proba(). Retrain "
            f"{classifier_config['display_name']}."
        )

    artifacts = {
        "classifier": classifier,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }

    _CLASSIFIER_CACHE[cache_key] = artifacts

    print(
        f"{classifier_config['display_name']} loaded successfully for "
        f"{config['display_name']}."
    )

    return artifacts


def load_open_set_artifact(model_id):
    """Load the calibrated unknown-speaker detector when available."""

    if model_id in _OPEN_SET_CACHE:
        return _OPEN_SET_CACHE[model_id]

    path = get_open_set_artifact_path(model_id)
    if not path.exists():
        _OPEN_SET_CACHE[model_id] = None
        return None

    try:
        artifact = joblib.load(path)
    except Exception as error:
        raise RuntimeError(
            f"Unable to load the unknown-speaker detector for {model_id}.\n\n"
            f"{error}"
        ) from error

    required = {
        "detector",
        "threshold",
        "centroids",
        "reference_embeddings",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(
            "The unknown-speaker detector is incomplete. Missing: "
            + ", ".join(sorted(missing))
        )

    _OPEN_SET_CACHE[model_id] = artifact
    return artifact


def calculate_open_set_score(embedding, model_id):
    """Return calibrated probability that an embedding is an enrolled voice.

    The detector combines similarity to enrolled speaker centroids with
    similarity to reference recordings.  ``None`` is returned for older model
    folders that do not yet contain an open-set detector.
    """

    artifact = load_open_set_artifact(model_id)
    if artifact is None:
        return None, None

    query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    centroids = np.asarray(artifact["centroids"], dtype=np.float32)
    references = np.asarray(
        artifact["reference_embeddings"],
        dtype=np.float32,
    )

    centroid_scores = np.sort(query @ centroids.T, axis=1)
    reference_scores = query @ references.T
    top_count = min(10, reference_scores.shape[1])
    top_scores = np.partition(
        reference_scores,
        -top_count,
        axis=1,
    )[:, -top_count:]
    top_scores.sort(axis=1)
    top3_count = min(3, top_count)
    top3_mean = top_scores[:, -top3_count:].mean(axis=1)

    features = np.column_stack(
        [
            centroid_scores[:, -1],
            centroid_scores[:, -1] - centroid_scores[:, -2],
            top_scores[:, -1],
            top3_mean,
            top_scores.mean(axis=1),
            top_scores[:, -1] - top3_mean,
        ]
    ).astype(np.float32)

    detector = artifact["detector"]
    detector_classes = list(detector.classes_)
    try:
        known_index = detector_classes.index(1)
    except ValueError as error:
        raise ValueError(
            "The unknown-speaker detector does not contain the known class."
        ) from error

    known_probability = float(
        detector.predict_proba(features)[0, known_index]
    )
    return known_probability, float(artifact["threshold"])


def calculate_open_set_gate(audio_path, embedding, model_id):
    """Apply the detector calibrated for the currently selected model."""

    path = Path(audio_path)
    try:
        cache_key = (
            model_id,
            str(path.resolve()),
            path.stat().st_mtime_ns,
        )
    except OSError:
        cache_key = (model_id, str(path), None)

    if cache_key in _OPEN_SET_GATE_CACHE:
        return _OPEN_SET_GATE_CACHE[cache_key]

    gate_result = calculate_open_set_score(
        embedding,
        model_id,
    )
    _OPEN_SET_GATE_CACHE[cache_key] = gate_result
    return gate_result


# ==========================================================
# Convert class IDs to speaker names
# ==========================================================

def decode_classifier_classes(
    classifier,
    label_encoder
):

    classifier_classes = np.asarray(
        classifier.classes_
    )

    try:
        encoded_classes = classifier_classes.astype(
            int
        )

        speaker_names = (
            label_encoder.inverse_transform(
                encoded_classes
            )
        )

        return [
            str(name)
            for name in speaker_names
        ]

    except (ValueError, TypeError):

        return [
            str(class_value)
            for class_value in classifier_classes
        ]


# ==========================================================
# Validate embedding dimension
# ==========================================================

def validate_embedding_dimension(
    embedding,
    scaler,
    model_id
):

    config = get_model_config(model_id)

    expected_dimension = getattr(
        scaler,
        "n_features_in_",
        None
    )

    if expected_dimension is None:
        return

    received_dimension = int(
        embedding.shape[0]
    )

    if received_dimension != expected_dimension:

        raise ValueError(
            f"Embedding dimension mismatch for "
            f"{config['display_name']}.\n\n"
            f"Expected: {expected_dimension}\n"
            f"Received: {received_dimension}\n\n"
            f"The selected classifier was trained using "
            f"a different embedding model."
        )


# ==========================================================
# Prediction
# ==========================================================

def predict_speaker(
    audio_path,
    model_id=DEFAULT_MODEL_ID,
    classifier_id=DEFAULT_CLASSIFIER_ID,
    rejection_threshold=None,
    return_diagnostics=False,
):
    """
    Predict a speaker using the selected model.

    Returns
    -------
    tuple
        speaker_name
        confidence_percentage
        inference_time_ms
        all_speaker_probabilities
    """

    config = get_model_config(model_id)
    classifier_config = get_classifier_config(
        classifier_id
    )

    start_time = time.perf_counter()

    artifacts = load_classifier_artifacts(
        model_id,
        classifier_id,
    )

    classifier = artifacts["classifier"]
    scaler = artifacts["scaler"]
    label_encoder = artifacts["label_encoder"]

    # Extract the embedding using the selected backend.
    embedding = extract_embedding(
        audio_path,
        model_id
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    ).reshape(-1)

    if embedding.size == 0:
        raise ValueError(
            "The extracted embedding is empty."
        )

    if not np.isfinite(embedding).all():
        raise ValueError(
            "The extracted embedding contains "
            "invalid values."
        )

    validate_embedding_dimension(
        embedding,
        scaler,
        model_id
    )

    # Apply the scaler belonging to this model.
    scaled_embedding = scaler.transform(
        embedding.reshape(1, -1)
    )

    # Probabilities are returned in the same order
    # as classifier.classes_.
    probability_values = classifier.predict_proba(
        scaled_embedding
    )[0]

    probability_values = np.asarray(
        probability_values,
        dtype=float
    )

    probability_total = float(
        probability_values.sum()
    )

    if probability_total <= 0:
        raise ValueError(
            "The classifier returned invalid "
            "probabilities."
        )

    # Protect against small floating-point differences.
    probability_values = (
        probability_values /
        probability_total
    )

    speaker_names = decode_classifier_classes(
        classifier,
        label_encoder
    )

    if len(speaker_names) != len(
        probability_values
    ):
        raise ValueError(
            "Speaker labels and probability counts "
            "do not match."
        )

    probability_distribution = [
        (
            speaker_name,
            float(probability) * 100.0
        )
        for speaker_name, probability in zip(
            speaker_names,
            probability_values
        )
    ]

    probability_distribution.sort(
        key=lambda item: item[1],
        reverse=True
    )

    top_candidate = (
        probability_distribution[0][0]
    )

    confidence_percentage = (
        probability_distribution[0][1]
    )

    if rejection_threshold is not None:
        threshold_percentage = float(rejection_threshold)
        if 0.0 <= threshold_percentage <= 1.0:
            threshold_percentage *= 100.0
        if not 0.0 <= threshold_percentage <= 100.0:
            raise ValueError(
                "The unknown-speaker rejection threshold must be between "
                "0 and 100 percent."
            )
    else:
        threshold_percentage = None

    # A closed-set classifier can be highly confident even for an unknown
    # person.  The model-specific detector compares the voice embedding with
    # enrolled reference voices and provides an independent known/unknown
    # decision.
    open_set_score, open_set_threshold = calculate_open_set_gate(
        audio_path,
        embedding,
        model_id,
    )

    predicted_speaker = top_candidate
    # Classifier confidence is conditional on the enrolled identities and is
    # not, by itself, an unknown-speaker probability.  Use it only as a
    # fallback for legacy models that do not have a calibrated open-set
    # detector.  When a detector exists, its independently calibrated score
    # controls the known/unknown decision.
    rejected_by_confidence = (
        open_set_score is None
        and threshold_percentage is not None
        and confidence_percentage < threshold_percentage
    )
    rejected_by_voice_match = (
        open_set_score is not None
        and open_set_threshold is not None
        and open_set_score < open_set_threshold
    )
    if rejected_by_confidence or rejected_by_voice_match:
        predicted_speaker = "Unknown Speaker"

    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000.0

    print()
    print("=" * 60)
    print("SPEAKER PREDICTION")
    print("=" * 60)
    print(
        f"Model      : {config['display_name']}"
    )
    print(
        f"Classifier : {classifier_config['display_name']}"
    )
    print(
        f"Speaker    : {predicted_speaker}"
    )
    if predicted_speaker == "Unknown Speaker":
        print(f"Top match  : {top_candidate}")
        if threshold_percentage is not None:
            print(f"Confidence threshold: {threshold_percentage:.2f}%")
    if open_set_score is not None:
        print(f"Known-speaker score: {open_set_score * 100:.2f}%")
        print(f"Open-set threshold: {open_set_threshold * 100:.2f}%")
    print(
        f"Confidence : "
        f"{confidence_percentage:.2f}%"
    )
    print(
        f"Inference  : "
        f"{inference_time_ms:.2f} ms"
    )

    result = (
        predicted_speaker,
        confidence_percentage,
        inference_time_ms,
        probability_distribution,
    )

    if not return_diagnostics:
        return result

    diagnostics = {
        "model_id": model_id,
        "model_name": config["display_name"],
        "classifier_id": classifier_id,
        "classifier_name": classifier_config["display_name"],
        "top_candidate": top_candidate,
        "closed_set_confidence_percentage": confidence_percentage,
        "known_speaker_probability": open_set_score,
        "unknown_speaker_probability": (
            None
            if open_set_score is None
            else max(0.0, min(1.0, 1.0 - open_set_score))
        ),
        "open_set_threshold": open_set_threshold,
        "open_set_detector_model_id": model_id,
        "open_set_detector_model_name": config["display_name"],
        "rejected_by_confidence": rejected_by_confidence,
        "rejected_by_voice_match": rejected_by_voice_match,
        "embedding": embedding.copy(),
    }
    return result + (diagnostics,)


# ==========================================================
# Model information
# ==========================================================

def get_model_information(
    model_id,
    classifier_id=DEFAULT_CLASSIFIER_ID,
):

    config = get_model_config(model_id)

    return {
        "model_id": model_id,
        "display_name": config["display_name"],
        "provider": config["provider"],
        "architecture": config["architecture"],
        "pretrained_name": config["pretrained_name"],
        "classifier_id": classifier_id,
        "classifier_name": get_classifier_config(
            classifier_id
        )["display_name"],
        "trained": is_model_trained(
            model_id,
            classifier_id,
        ),
    }


def get_speaker_classes(
    model_id=DEFAULT_MODEL_ID,
    classifier_id=DEFAULT_CLASSIFIER_ID,
):

    artifacts = load_classifier_artifacts(
        model_id,
        classifier_id,
    )

    classifier = artifacts["classifier"]
    label_encoder = artifacts["label_encoder"]

    return decode_classifier_classes(
        classifier,
        label_encoder
    )


# ==========================================================
# Cache management
# ==========================================================

def clear_prediction_cache():

    _CLASSIFIER_CACHE.clear()
    _OPEN_SET_CACHE.clear()
    _OPEN_SET_GATE_CACHE.clear()

    clear_model_cache()
