# ==========================================================
# extract_embeddings_multi.py
# Extract dataset embeddings using a selected model
# ==========================================================

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from embedding_extractors import (
    clear_model_cache,
    extract_embedding,
    get_device,
)
from model_config import (
    MODEL_CONFIGS,
    create_model_directories,
    get_model_config,
)


# ==========================================================
# Configuration
# ==========================================================

DEFAULT_DATASET_DIRECTORY = "processed_dataset"

SKIP_CLASSES = {
    "_background_noise_",
    "other",
}


# ==========================================================
# Command-line arguments
# ==========================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Extract speaker embeddings using a selected "
            "speaker-recognition model."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model used to extract embeddings."
    )

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_DIRECTORY,
        help="Processed speaker dataset directory."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing embedding files."
    )

    return parser.parse_args()


# ==========================================================
# Find dataset files
# ==========================================================

def collect_audio_files(dataset_directory):

    dataset_directory = Path(dataset_directory)

    if not dataset_directory.exists():
        raise FileNotFoundError(
            f"Dataset directory was not found:\n"
            f"{dataset_directory.resolve()}"
        )

    if not dataset_directory.is_dir():
        raise NotADirectoryError(
            f"Dataset path is not a directory:\n"
            f"{dataset_directory.resolve()}"
        )

    dataset_items = []

    speaker_directories = sorted(
        directory
        for directory in dataset_directory.iterdir()
        if directory.is_dir()
    )

    for speaker_directory in speaker_directories:

        speaker_name = speaker_directory.name

        if speaker_name.casefold() in {
            name.casefold()
            for name in SKIP_CLASSES
        }:
            print(
                f"Skipping non-speaker folder: "
                f"{speaker_name}"
            )
            continue

        wav_files = sorted(
            speaker_directory.rglob("*.wav")
        )

        if not wav_files:
            print(
                f"Warning: no WAV files found for "
                f"{speaker_name}"
            )
            continue

        for audio_path in wav_files:
            dataset_items.append(
                (audio_path, speaker_name)
            )

    if not dataset_items:
        raise ValueError(
            "No valid speaker audio files were found."
        )

    return dataset_items


# ==========================================================
# Check existing output
# ==========================================================

def check_output_files(
    embedding_directory,
    overwrite
):

    x_path = embedding_directory / "X.npy"
    y_path = embedding_directory / "y.npy"

    existing_files = [
        path
        for path in (x_path, y_path)
        if path.exists()
    ]

    if existing_files and not overwrite:

        existing_text = "\n".join(
            str(path)
            for path in existing_files
        )

        raise FileExistsError(
            "Embedding files already exist:\n\n"
            f"{existing_text}\n\n"
            "Use --overwrite only if you want to "
            "replace them."
        )

    return x_path, y_path


# ==========================================================
# Save extraction metadata
# ==========================================================

def save_metadata(
    output_directory,
    model_id,
    embeddings,
    labels,
    successful_files,
    failed_files,
):

    config = get_model_config(model_id)

    unique_speakers, speaker_counts = np.unique(
        labels,
        return_counts=True
    )

    speakers = {
        str(speaker): int(count)
        for speaker, count in zip(
            unique_speakers,
            speaker_counts
        )
    }

    metadata = {
        "model_id": model_id,
        "display_name": config["display_name"],
        "provider": config["provider"],
        "architecture": config["architecture"],
        "pretrained_name": config["pretrained_name"],
        "sample_rate": config["sample_rate"],
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "number_of_embeddings": int(
            embeddings.shape[0]
        ),
        "number_of_speakers": int(
            len(unique_speakers)
        ),
        "speakers": speakers,
        "successful_files": int(successful_files),
        "failed_files": int(len(failed_files)),
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    metadata_path = (
        output_directory / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as metadata_file:

        json.dump(
            metadata,
            metadata_file,
            indent=4,
            ensure_ascii=False
        )


# ==========================================================
# Save failed-file report
# ==========================================================

def save_failed_files(
    output_directory,
    failed_files
):

    failed_path = (
        output_directory / "failed_files.txt"
    )

    if not failed_files:

        if failed_path.exists():
            failed_path.unlink()

        return

    with failed_path.open(
        "w",
        encoding="utf-8"
    ) as failed_file:

        for audio_path, error_message in failed_files:

            failed_file.write(
                f"File: {audio_path}\n"
            )

            failed_file.write(
                f"Error: {error_message}\n"
            )

            failed_file.write(
                "-" * 70 + "\n"
            )


# ==========================================================
# Extract embeddings
# ==========================================================

def extract_dataset_embeddings(
    model_id,
    dataset_directory,
    overwrite=False
):

    create_model_directories()

    config = get_model_config(model_id)

    output_directory = Path(
        config["embedding_directory"]
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    x_path, y_path = check_output_files(
        output_directory,
        overwrite
    )

    dataset_items = collect_audio_files(
        dataset_directory
    )

    print()
    print("=" * 70)
    print("MULTI-MODEL EMBEDDING EXTRACTION")
    print("=" * 70)
    print(
        f"Model       : {config['display_name']}"
    )
    print(
        f"Provider    : {config['provider']}"
    )
    print(
        f"Architecture: {config['architecture']}"
    )
    print(
        f"Device      : {get_device()}"
    )
    print(
        f"Audio files : {len(dataset_items)}"
    )
    print(
        f"Output      : {output_directory}"
    )
    print("=" * 70)
    print()

    print("Running model compatibility check...")

    first_audio_path = dataset_items[0][0]

    try:
        test_embedding = extract_embedding(
            str(first_audio_path),
            model_id
        )

    except Exception as error:
        raise RuntimeError(
            "The selected model could not be initialized. "
            "Extraction was stopped before processing the "
            "complete dataset.\n\n"
            f"{error}"
        ) from error

    expected_dimension = int(
        np.asarray(test_embedding).reshape(-1).shape[0]
    )

    print(
        f"Model check passed. Embedding dimension: "
        f"{expected_dimension}"
    )

    embeddings = []
    labels = []
    audio_paths = []
    failed_files = []

    progress = tqdm(
        dataset_items,
        desc=config["display_name"],
        unit="file"
    )

    for audio_path, speaker_name in progress:

        progress.set_postfix_str(
            speaker_name
        )

        try:
            embedding = extract_embedding(
                str(audio_path),
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
                    "Embedding contains invalid values."
                )

            if expected_dimension is None:
                expected_dimension = embedding.shape[0]

                print(
                    f"\nDetected embedding dimension: "
                    f"{expected_dimension}"
                )

            elif embedding.shape[0] != expected_dimension:

                raise ValueError(
                    "Embedding dimension mismatch. "
                    f"Expected {expected_dimension}, "
                    f"received {embedding.shape[0]}."
                )

            embeddings.append(embedding)
            labels.append(speaker_name)
            audio_paths.append(str(audio_path))

        except Exception as error:

            failed_files.append(
                (
                    str(audio_path),
                    str(error)
                )
            )

            tqdm.write(
                f"Failed: {audio_path}\n"
                f"Reason: {error}"
            )

    if not embeddings:
        raise RuntimeError(
            "Embedding extraction failed for every file."
        )

    X = np.stack(
        embeddings,
        axis=0
    ).astype(np.float32)

    y = np.asarray(
        labels,
        dtype=str
    )

    paths = np.asarray(
        audio_paths,
        dtype=str
    )

    if len(X) != len(y):
        raise RuntimeError(
            "Embedding and label counts do not match."
        )

    np.save(
        x_path,
        X
    )

    np.save(
        y_path,
        y
    )

    np.save(
        output_directory / "audio_paths.npy",
        paths
    )

    save_metadata(
        output_directory=output_directory,
        model_id=model_id,
        embeddings=X,
        labels=y,
        successful_files=len(X),
        failed_files=failed_files,
    )

    save_failed_files(
        output_directory,
        failed_files
    )

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETED")
    print("=" * 70)
    print(f"Model             : {config['display_name']}")
    print(f"Embeddings shape  : {X.shape}")
    print(f"Labels shape      : {y.shape}")
    print(f"Successful files  : {len(X)}")
    print(f"Failed files      : {len(failed_files)}")
    print(f"Number of speakers: {len(np.unique(y))}")
    print(f"Saved X           : {x_path}")
    print(f"Saved y           : {y_path}")
    print("=" * 70)

    return X, y


# ==========================================================
# Main
# ==========================================================

def main():

    arguments = parse_arguments()

    try:
        extract_dataset_embeddings(
            model_id=arguments.model,
            dataset_directory=arguments.dataset,
            overwrite=arguments.overwrite,
        )

    except KeyboardInterrupt:

        print(
            "\nEmbedding extraction was cancelled."
        )

        sys.exit(130)

    except Exception as error:

        print()
        print("=" * 70)
        print("EXTRACTION FAILED")
        print("=" * 70)
        print(error)
        print()

        traceback.print_exc()

        sys.exit(1)

    finally:
        clear_model_cache()


if __name__ == "__main__":
    main()