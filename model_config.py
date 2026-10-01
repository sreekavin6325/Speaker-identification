# ==========================================================
# model_config.py
# Configuration for all speaker-identification models
# ==========================================================

from pathlib import Path


# Project directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Directory containing trained classifiers
MODELS_ROOT = PROJECT_ROOT / "models"

# Directory containing extracted embeddings
EMBEDDINGS_ROOT = PROJECT_ROOT / "embeddings"


MODEL_CONFIGS = {
    # ------------------------------------------------------
    # SpeechBrain ECAPA-TDNN
    # ------------------------------------------------------
    "speechbrain_ecapa": {
        "display_name": "SpeechBrain — ECAPA-TDNN",
        "provider": "SpeechBrain",
        "architecture": "ECAPA-TDNN",
        "extractor": "speechbrain",
        "pretrained_name": (
            "speechbrain/spkrec-ecapa-voxceleb"
        ),
        "sample_rate": 16000,
        "model_directory": (
            MODELS_ROOT / "speechbrain_ecapa"
        ),
        "embedding_directory": (
            EMBEDDINGS_ROOT / "speechbrain_ecapa"
        ),
    },

    # ------------------------------------------------------
    # SpeechBrain X-Vector
    # ------------------------------------------------------
    "speechbrain_xvector": {
        "display_name": "SpeechBrain — X-Vector",
        "provider": "SpeechBrain",
        "architecture": "X-Vector TDNN",
        "extractor": "speechbrain",
        "pretrained_name": (
            "speechbrain/spkrec-xvect-voxceleb"
        ),
        "sample_rate": 16000,
        "model_directory": (
            MODELS_ROOT / "speechbrain_xvector"
        ),
        "embedding_directory": (
            EMBEDDINGS_ROOT / "speechbrain_xvector"
        ),
    },

    # ------------------------------------------------------
    # Microsoft WavLM
    # ------------------------------------------------------
    "wavlm_base_plus_sv": {
        "display_name": "Microsoft — WavLM Base Plus SV",
        "provider": "Microsoft",
        "architecture": "WavLM Transformer",
        "extractor": "wavlm",
        "pretrained_name": (
            "microsoft/wavlm-base-plus-sv"
        ),
        "sample_rate": 16000,
        "model_directory": (
            MODELS_ROOT / "wavlm_base_plus_sv"
        ),
        "embedding_directory": (
            EMBEDDINGS_ROOT / "wavlm_base_plus_sv"
        ),
    },

    # ------------------------------------------------------
    # unispeech
    # ------------------------------------------------------
    "unispeech_sat_base_plus_sv": {
        "display_name": "Microsoft — UniSpeech-SAT Base Plus SV",
        "provider": "Microsoft / Hugging Face",
        "architecture": "UniSpeech-SAT + X-Vector Head",

        "source": "microsoft/unispeech-sat-base-plus-sv",
        "pretrained_name": (
            "microsoft/unispeech-sat-base-plus-sv"
        ),

        "embedding_dim": 512,
        "sample_rate": 16000,
        "extractor": "unispeech_sat",
        "enabled": True,

        "embedding_directory": (
            Path(__file__).resolve().parent
            / "embeddings"
            / "unispeech_sat_base_plus_sv"
        ),

        "model_directory": (
            Path(__file__).resolve().parent
            / "models"
            / "unispeech_sat_base_plus_sv"
        ),
    },
}


# Default model used when the application opens
DEFAULT_MODEL_ID = "speechbrain_ecapa"


def get_model_config(model_id):
    """
    Return the configuration for one model.
    """

    if model_id not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown speaker model: {model_id}"
        )

    return MODEL_CONFIGS[model_id]


def get_model_display_names():
    """
    Return all names displayed in the Settings dropdown.
    """

    return [
        config["display_name"]
        for config in MODEL_CONFIGS.values()
    ]


def get_model_id_from_display_name(display_name):
    """
    Convert a displayed model name to its internal ID.
    """

    for model_id, config in MODEL_CONFIGS.items():

        if config["display_name"] == display_name:
            return model_id

    raise ValueError(
        f"Unknown model name: {display_name}"
    )


def create_model_directories():
    """
    Create model and embedding folders if they do not exist.
    """

    MODELS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    EMBEDDINGS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    for config in MODEL_CONFIGS.values():

        config["model_directory"].mkdir(
            parents=True,
            exist_ok=True
        )

        config["embedding_directory"].mkdir(
            parents=True,
            exist_ok=True
        )


if __name__ == "__main__":

    create_model_directories()

    print("Model directories created successfully.")

    for model_id, config in MODEL_CONFIGS.items():

        print(
            f"{model_id}: "
            f"{config['display_name']}"
        )