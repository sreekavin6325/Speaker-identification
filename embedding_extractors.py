# ==========================================================
# embedding_extractors.py
# Unified multi-model speaker-embedding extraction
# ==========================================================

import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from model_config import get_model_config


# ==========================================================
# Model cache
# ==========================================================

# Loaded models are stored here so that they are not loaded
# again for every audio file.
_MODEL_CACHE = {}


# ==========================================================
# Configuration helpers
# ==========================================================

def get_pretrained_source(config):
    """
    Return the pretrained model name from the configuration.

    Both `pretrained_name` and `source` are supported.
    """

    source = (
        config.get("pretrained_name")
        or config.get("source")
    )

    if not source:
        raise KeyError(
            "Model configuration must contain either "
            "'pretrained_name' or 'source'."
        )

    return source


def get_device():
    """
    Use CUDA when available; otherwise use CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ==========================================================
# Audio loading and validation
# ==========================================================

def load_audio(audio_path, sample_rate=16000):
    """
    Load audio as mono, resample and validate it.

    Parameters
    ----------
    audio_path:
        Path to the audio file.

    sample_rate:
        Required output sample rate.

    Returns
    -------
    numpy.ndarray
        One-dimensional float32 waveform.
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file was not found:\n{audio_path}"
        )

    if not audio_path.is_file():
        raise ValueError(
            f"Audio path is not a file:\n{audio_path}"
        )

    # SoundFile reads WAV/FLAC directly and avoids librosa's
    # optional SciPy code path.  This is important on Windows
    # systems where Application Control can block a lazily loaded
    # SciPy DLL on the first call to librosa.load().
    try:
        waveform, original_sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=True,
        )
        waveform = waveform.mean(
            axis=1,
            dtype=np.float32,
        )

    except Exception as soundfile_error:
        # Keep librosa as a fallback for formats unsupported by
        # the locally installed libsndfile build.
        try:
            import librosa

            waveform, original_sample_rate = librosa.load(
                str(audio_path),
                sr=None,
                mono=True,
            )

        except Exception as librosa_error:
            raise RuntimeError(
                f"Unable to load audio file:\n"
                f"{audio_path}\n\n"
                f"SoundFile: {soundfile_error}\n"
                f"librosa: {librosa_error}"
            ) from librosa_error

    original_sample_rate = int(original_sample_rate)

    if original_sample_rate <= 0:
        raise ValueError(
            "The audio file has an invalid sample rate: "
            f"{original_sample_rate}"
        )

    if original_sample_rate != sample_rate:
        try:
            import librosa

            waveform = librosa.resample(
                np.asarray(waveform, dtype=np.float32),
                orig_sr=original_sample_rate,
                target_sr=sample_rate,
            )

        except Exception as error:
            raise RuntimeError(
                "Unable to resample audio from "
                f"{original_sample_rate} Hz to "
                f"{sample_rate} Hz:\n{audio_path}\n\n{error}"
            ) from error

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    ).reshape(-1)

    if waveform.size == 0:
        raise ValueError(
            f"The audio file is empty:\n{audio_path}"
        )

    if not np.isfinite(waveform).all():
        raise ValueError(
            "The audio contains NaN or infinite values."
        )

    # Very short audio does not contain enough speaker
    # information and can also cause convolution kernel errors.
    minimum_duration_seconds = 0.5

    duration_seconds = (
        waveform.size / float(sample_rate)
    )

    if duration_seconds < minimum_duration_seconds:
        raise ValueError(
            "Audio is too short for reliable speaker "
            "identification. "
            f"Duration: {duration_seconds:.3f} seconds. "
            f"Minimum: {minimum_duration_seconds:.1f} seconds."
        )

    rms_energy = float(
        np.sqrt(
            np.mean(
                np.square(
                    waveform,
                    dtype=np.float64,
                )
            )
        )
    )

    if not np.isfinite(rms_energy):
        raise ValueError(
            "Unable to calculate valid audio energy."
        )

    if rms_energy < 1e-6:
        raise ValueError(
            "Audio is silent or contains insufficient "
            "voice energy."
        )

    return waveform


# ==========================================================
# Embedding validation
# ==========================================================

def validate_embedding(embedding, config):
    """
    Convert an embedding to a valid one-dimensional
    float32 NumPy array and verify its dimension.
    """

    if isinstance(embedding, torch.Tensor):
        embedding = (
            embedding
            .detach()
            .cpu()
            .numpy()
        )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(-1)

    if embedding.size == 0:
        raise ValueError(
            "The model returned an empty embedding."
        )

    if not np.isfinite(embedding).all():
        raise ValueError(
            "Embedding contains invalid values."
        )

    expected_dimension = config.get(
        "embedding_dim"
    )

    if expected_dimension is not None:
        expected_dimension = int(
            expected_dimension
        )

        if embedding.size != expected_dimension:
            raise ValueError(
                "Unexpected embedding dimension. "
                f"Received: {embedding.size}. "
                f"Expected: {expected_dimension}."
            )

    return embedding


# ==========================================================
# SpeechBrain Windows compatibility
# ==========================================================

def apply_speechbrain_windows_lazy_import_fix():
    """
    Prevent Python's inspect module from accidentally loading
    optional SpeechBrain integrations such as K2 on Windows.
    """

    if os.name != "nt":
        return

    import importlib
    import inspect
    import sys
    import warnings

    from speechbrain.utils.importutils import LazyModule

    if getattr(
        LazyModule,
        "_speaker_project_windows_fix",
        False,
    ):
        return

    def windows_ensure_module(self, stacklevel):

        importer_frame = None

        try:
            importer_frame = inspect.getframeinfo(
                sys._getframe(stacklevel + 1)
            )

        except (AttributeError, ValueError):
            warnings.warn(
                "Unable to inspect the lazy-import frame."
            )

        if importer_frame is not None:
            importer_filename = os.path.basename(
                importer_frame.filename
            ).lower()

            if importer_filename == "inspect.py":
                raise AttributeError()

        if self.lazy_module is None:

            try:
                if self.package is None:
                    self.lazy_module = (
                        importlib.import_module(
                            self.target
                        )
                    )

                else:
                    self.lazy_module = (
                        importlib.import_module(
                            f".{self.target}",
                            self.package,
                        )
                    )

            except Exception as error:
                raise ImportError(
                    f"Lazy import of {repr(self)} failed"
                ) from error

        return self.lazy_module

    LazyModule.ensure_module = windows_ensure_module

    LazyModule._speaker_project_windows_fix = True


def apply_transformers_windows_regex_fix():
    """Use the standard-library regex engine if Windows blocks regex.pyd.

    Some managed Windows systems reject the optional native ``regex`` DLL
    inside a virtual environment.  Transformers imports that package while
    building documentation helpers even though the WavLM and UniSpeech audio
    inference paths used here do not require its extended syntax.  Falling
    back to :mod:`re` keeps those audio-only imports usable.  A healthy native
    ``regex`` installation is always preferred and left untouched.
    """

    if os.name != "nt":
        return

    import re
    import sys

    if "regex" in sys.modules:
        return

    try:
        import regex  # noqa: F401
    except (ImportError, OSError):
        sys.modules["regex"] = re


# ==========================================================
# SpeechBrain model loading
# ==========================================================

def load_speechbrain_model(model_id):
    """
    Load and cache a SpeechBrain speaker model.
    """

    cache_key = f"speechbrain:{model_id}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    apply_speechbrain_windows_lazy_import_fix()

    config = get_model_config(model_id)
    pretrained_source = get_pretrained_source(
        config
    )

    device = get_device()

    try:
        from speechbrain.inference.speaker import (
            EncoderClassifier,
        )

        from speechbrain.utils.fetching import (
            LocalStrategy,
        )

        # Explicitly importing these classes prevents the
        # SpeechBrain X-Vector HyperPyYAML lazy-import problem.
        if model_id == "speechbrain_xvector":
            from speechbrain.lobes.models.Xvector import (
                Classifier,
                Xvector,
            )

            # Referencing the classes prevents code-analysis
            # tools from treating them as unused imports.
            _ = Xvector
            _ = Classifier

    except ImportError as error:
        raise ImportError(
            "SpeechBrain is unavailable.\n\n"
            "Install it using:\n"
            "python -m pip install speechbrain"
        ) from error

    saved_directory = (
        Path("pretrained_models") / model_id
    )

    saved_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Loading {config['display_name']} "
        f"on {device}..."
    )

    try:
        model = EncoderClassifier.from_hparams(
            source=pretrained_source,
            savedir=str(saved_directory),
            run_opts={
                "device": str(device),
            },
            local_strategy=LocalStrategy.COPY,
        )

    except Exception as error:
        raise RuntimeError(
            f"Unable to load "
            f"{config['display_name']}.\n\n"
            f"{error}"
        ) from error

    _MODEL_CACHE[cache_key] = model

    print(
        f"{config['display_name']} "
        f"loaded successfully."
    )

    return model


# ==========================================================
# SpeechBrain embedding extraction
# ==========================================================

def extract_speechbrain_embedding(
    audio_path,
    model_id,
):
    """
    Extract a SpeechBrain ECAPA-TDNN or X-Vector embedding.
    """

    config = get_model_config(model_id)

    waveform = load_audio(
        audio_path,
        sample_rate=int(config["sample_rate"]),
    )

    model = load_speechbrain_model(model_id)
    device = get_device()

    signal = torch.from_numpy(
        waveform
    ).unsqueeze(0)

    signal = signal.to(device)

    try:
        with torch.inference_mode():
            embedding = model.encode_batch(
                signal
            )

    except Exception as error:
        raise RuntimeError(
            "SpeechBrain embedding extraction "
            f"failed.\n\n{error}"
        ) from error

    embedding = validate_embedding(
        embedding.squeeze(),
        config,
    )

    return embedding


# ==========================================================
# WavLM model loading
# ==========================================================

def load_wavlm_model(model_id):
    """
    Load and cache Microsoft WavLM.
    """

    cache_key = f"wavlm:{model_id}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    config = get_model_config(model_id)
    pretrained_source = get_pretrained_source(
        config
    )

    device = get_device()

    apply_transformers_windows_regex_fix()

    try:
        from transformers import (
            Wav2Vec2FeatureExtractor,
            WavLMForXVector,
        )

    except ImportError as error:
        raise ImportError(
            "Transformers is unavailable.\n\n"
            "Install it using:\n"
            "python -m pip install transformers"
        ) from error

    print(
        f"Loading {config['display_name']} "
        f"on {device}..."
    )

    try:
        feature_extractor = (
            Wav2Vec2FeatureExtractor.from_pretrained(
                pretrained_source
            )
        )

        model = WavLMForXVector.from_pretrained(
            pretrained_source
        )

        model.to(device)
        model.eval()

    except Exception as error:
        raise RuntimeError(
            f"Unable to load "
            f"{config['display_name']}.\n\n"
            f"{error}"
        ) from error

    cached_value = {
        "feature_extractor": feature_extractor,
        "model": model,
        "device": device,
    }

    _MODEL_CACHE[cache_key] = cached_value

    print(
        f"{config['display_name']} "
        f"loaded successfully."
    )

    return cached_value


# ==========================================================
# WavLM embedding extraction
# ==========================================================

def extract_wavlm_embedding(
    audio_path,
    model_id,
):
    """
    Extract a normalized WavLM speaker embedding.
    """

    config = get_model_config(model_id)

    waveform = load_audio(
        audio_path,
        sample_rate=int(config["sample_rate"]),
    )

    loaded_model = load_wavlm_model(model_id)

    feature_extractor = (
        loaded_model["feature_extractor"]
    )

    model = loaded_model["model"]
    device = loaded_model["device"]

    try:
        inputs = feature_extractor(
            waveform,
            sampling_rate=int(
                config["sample_rate"]
            ),
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        model_arguments = {
            name: value.to(device)
            for name, value in inputs.items()
        }

        with torch.inference_mode():
            output = model(
                **model_arguments
            )

            embedding = output.embeddings

            embedding = (
                torch.nn.functional.normalize(
                    embedding,
                    p=2,
                    dim=-1,
                )
            )

    except Exception as error:
        raise RuntimeError(
            "WavLM embedding extraction failed.\n\n"
            f"{error}"
        ) from error

    embedding = validate_embedding(
        embedding.squeeze(),
        config,
    )

    return embedding


# ==========================================================
# UniSpeech-SAT model loading
# ==========================================================

def load_unispeech_sat_model(
    model_id="unispeech_sat_base_plus_sv",
):
    """
    Load and cache Microsoft UniSpeech-SAT.
    """

    cache_key = f"unispeech_sat:{model_id}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    config = get_model_config(model_id)
    pretrained_source = get_pretrained_source(
        config
    )

    device = get_device()

    apply_transformers_windows_regex_fix()

    try:
        from transformers import (
            UniSpeechSatForXVector,
            Wav2Vec2FeatureExtractor,
        )

    except ImportError as error:
        raise ImportError(
            "The installed Transformers version does not "
            "provide UniSpeech-SAT.\n\n"
            "Install or update Transformers using:\n"
            "python -m pip install transformers"
        ) from error

    print(
        f"Loading {config['display_name']} "
        f"on {device}..."
    )

    try:
        feature_extractor = (
            Wav2Vec2FeatureExtractor.from_pretrained(
                pretrained_source
            )
        )

        model = UniSpeechSatForXVector.from_pretrained(
            pretrained_source
        )

        model.to(device)
        model.eval()

    except Exception as error:
        raise RuntimeError(
            f"Unable to load "
            f"{config['display_name']}.\n\n"
            f"{error}"
        ) from error

    cached_value = {
        "feature_extractor": feature_extractor,
        "model": model,
        "device": device,
    }

    _MODEL_CACHE[cache_key] = cached_value

    print(
        f"{config['display_name']} "
        f"loaded successfully."
    )

    return cached_value


# ==========================================================
# UniSpeech-SAT embedding extraction
# ==========================================================

def extract_unispeech_sat_embedding(
    audio_path,
    model_id="unispeech_sat_base_plus_sv",
):
    """
    Extract a normalized UniSpeech-SAT speaker embedding.
    """

    config = get_model_config(model_id)

    waveform = load_audio(
        audio_path,
        sample_rate=int(config["sample_rate"]),
    )

    loaded_model = load_unispeech_sat_model(
        model_id
    )

    feature_extractor = (
        loaded_model["feature_extractor"]
    )

    model = loaded_model["model"]
    device = loaded_model["device"]

    try:
        inputs = feature_extractor(
            waveform,
            sampling_rate=int(
                config["sample_rate"]
            ),
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        model_arguments = {
            name: value.to(device)
            for name, value in inputs.items()
        }

        with torch.inference_mode():
            output = model(
                **model_arguments
            )

            embedding = output.embeddings

            embedding = (
                torch.nn.functional.normalize(
                    embedding,
                    p=2,
                    dim=-1,
                )
            )

    except Exception as error:
        raise RuntimeError(
            "UniSpeech-SAT embedding extraction "
            f"failed.\n\n{error}"
        ) from error

    embedding = validate_embedding(
        embedding.squeeze(),
        config,
    )

    return embedding


# ==========================================================
# NVIDIA NeMo model loading
# ==========================================================

def load_nemo_model(model_id):
    """
    Load and cache NVIDIA NeMo TitaNet.

    NeMo is optional and is not needed by the Windows
    SpeechBrain, WavLM or UniSpeech-SAT pipelines.
    """

    cache_key = f"nemo:{model_id}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    config = get_model_config(model_id)
    pretrained_source = get_pretrained_source(
        config
    )

    device = get_device()

    try:
        import nemo.collections.asr as nemo_asr

    except ImportError as error:
        raise ImportError(
            "NVIDIA NeMo is not installed.\n\n"
            "NeMo is optional. On Windows, WSL2 is "
            "generally recommended for NeMo."
        ) from error

    print(
        f"Loading {config['display_name']} "
        f"on {device}..."
    )

    try:
        model = (
            nemo_asr.models
            .EncDecSpeakerLabelModel
            .from_pretrained(
                model_name=pretrained_source
            )
        )

        model.to(device)
        model.eval()

    except Exception as error:
        raise RuntimeError(
            f"Unable to load "
            f"{config['display_name']}.\n\n"
            f"{error}"
        ) from error

    _MODEL_CACHE[cache_key] = model

    print(
        f"{config['display_name']} "
        f"loaded successfully."
    )

    return model


# ==========================================================
# NVIDIA NeMo embedding extraction
# ==========================================================

def extract_nemo_embedding(
    audio_path,
    model_id,
):
    """
    Extract a NeMo TitaNet embedding.

    NeMo expects a 16-kHz mono WAV. A temporary compatible
    WAV file is therefore created.
    """

    config = get_model_config(model_id)

    waveform = load_audio(
        audio_path,
        sample_rate=int(config["sample_rate"]),
    )

    model = load_nemo_model(model_id)

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name

        sf.write(
            temporary_path,
            waveform,
            int(config["sample_rate"]),
            subtype="PCM_16",
        )

        with torch.inference_mode():
            embedding = model.get_embedding(
                temporary_path
            )

    except Exception as error:
        raise RuntimeError(
            "NeMo embedding extraction failed.\n\n"
            f"{error}"
        ) from error

    finally:
        if (
            temporary_path
            and os.path.exists(temporary_path)
        ):
            try:
                os.remove(temporary_path)

            except OSError:
                pass

    embedding = validate_embedding(
        embedding,
        config,
    )

    return embedding


# ==========================================================
# Unified extraction function
# ==========================================================

def extract_embedding(audio_path, model_id):
    """
    Extract an embedding using the selected model.

    This is the main function used by dataset extraction
    and real-time prediction.
    """

    config = get_model_config(model_id)

    extractor = config.get("extractor")

    if not extractor:
        raise KeyError(
            f"Model '{model_id}' does not contain an "
            "'extractor' configuration value."
        )

    if extractor == "speechbrain":
        return extract_speechbrain_embedding(
            audio_path,
            model_id,
        )

    if extractor == "wavlm":
        return extract_wavlm_embedding(
            audio_path,
            model_id,
        )

    if extractor == "unispeech_sat":
        return extract_unispeech_sat_embedding(
            audio_path,
            model_id,
        )

    if extractor == "nemo":
        return extract_nemo_embedding(
            audio_path,
            model_id,
        )

    raise ValueError(
        f"Unsupported embedding extractor: {extractor}"
    )


# ==========================================================
# Cache management
# ==========================================================

def clear_model_cache():
    """
    Remove all loaded models from memory.
    """

    _MODEL_CACHE.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
