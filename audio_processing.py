import os
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import noisereduce as nr



#############################################
# CONFIGURATION
#############################################

INPUT_DATASET = "dataset"

OUTPUT_DATASET = "processed_dataset"

TARGET_SR = 16000

REMOVE_SILENCE = True

REDUCE_NOISE = False

NORMALIZE_AUDIO = True



#############################################
# Utility Functions
#############################################

def normalize_audio(audio):

    """
    Peak normalization.
    """

    max_amp = np.max(np.abs(audio))

    if max_amp == 0:
        return audio

    return audio / max_amp



def remove_silence(audio, sr):

    """
    Remove leading and trailing silence.
    """

    audio_trimmed, _ = librosa.effects.trim(
        audio,
        top_db=20
    )

    return audio_trimmed



def reduce_noise(audio, sr):

    """
    Noise reduction using noisereduce.
    """

    reduced = nr.reduce_noise(
        y=audio,
        sr=sr
    )

    return reduced



#############################################
# Main preprocessing function
#############################################

def preprocess_audio(input_file, output_file):

    try:

        #####################################
        # Load Audio
        #####################################

        audio, sr = librosa.load(
            input_file,
            sr=TARGET_SR,
            mono=True
        )

        #####################################
        # Silence Removal
        #####################################

        if REMOVE_SILENCE:

            audio = remove_silence(audio, sr)

        #####################################
        # Noise Reduction
        #####################################

        if REDUCE_NOISE:

            audio = reduce_noise(audio, sr)

        #####################################
        # Normalize
        #####################################

        if NORMALIZE_AUDIO:

            audio = normalize_audio(audio)

        #####################################
        # Save
        #####################################

        sf.write(
            output_file,
            audio,
            TARGET_SR
        )

    except Exception as e:

        print(f"Error processing {input_file}")

        print(e)



#############################################
# Scan Dataset
#############################################

def process_dataset():

    input_root = Path(INPUT_DATASET)

    output_root = Path(OUTPUT_DATASET)

    output_root.mkdir(exist_ok=True)

    supported = [
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a"
    ]

    speaker_folders = sorted(input_root.iterdir())

    for speaker in speaker_folders:

        if not speaker.is_dir():
            continue

        output_speaker = output_root / speaker.name

        output_speaker.mkdir(exist_ok=True)

        files = []

        for ext in supported:

            files.extend(
                speaker.glob(f"*{ext}")
            )

        print(f"\nProcessing {speaker.name}")

        for audio_file in tqdm(files):

            output_file = (
                output_speaker /
                (audio_file.stem + ".wav")
            )

            preprocess_audio(
                str(audio_file),
                str(output_file)
            )



#############################################
# Main
#############################################

if __name__ == "__main__":

    process_dataset()

    print("\nDataset preprocessing completed.")