import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

from tqdm import tqdm

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


############################################################
# CONFIGURATION
############################################################

DATASET_PATH = "processed_dataset"
EMBEDDING_PATH = "embeddings"
MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
TARGET_SAMPLE_RATE = 16000


############################################################
# CREATE OUTPUT DIRECTORY
############################################################

os.makedirs(EMBEDDING_PATH, exist_ok=True)


############################################################
# LOAD PRETRAINED MODEL
############################################################

print("=" * 60)
print("Loading ECAPA-TDNN Model...")
print("=" * 60)

classifier = EncoderClassifier.from_hparams(
    source=MODEL_NAME,
    savedir="pretrained_models/ecapa",
    local_strategy=LocalStrategy.COPY
)

print("Model Loaded Successfully!\n")


############################################################
# STORAGE
############################################################

X = []
y = []


############################################################
# PROCESS EACH SPEAKER
############################################################

dataset = Path(DATASET_PATH)

SKIP_CLASSES = {
    "_background_noise_",
    "other"
}

for speaker_dir in sorted(dataset.iterdir()):

    if not speaker_dir.is_dir():
        continue

    if speaker_dir.name in SKIP_CLASSES:
        print(f"Skipping folder: {speaker_dir.name}")
        continue

    print(f"Processing Speaker : {speaker_dir.name}")

    wav_files = sorted(speaker_dir.glob("*.wav"))

    for wav_file in tqdm(wav_files):

        try:

            ####################################################
            # LOAD AUDIO
            ####################################################

            waveform, sample_rate = torchaudio.load(str(wav_file))

            ####################################################
            # CONVERT TO MONO
            ####################################################

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            ####################################################
            # RESAMPLE TO 16kHz
            ####################################################

            if sample_rate != TARGET_SAMPLE_RATE:

                resampler = torchaudio.transforms.Resample(
                    sample_rate,
                    TARGET_SAMPLE_RATE
                )

                waveform = resampler(waveform)

            ####################################################
            # EXTRACT EMBEDDING
            ####################################################

            with torch.no_grad():

                embedding = classifier.encode_batch(waveform)

            ####################################################
            # CONVERT TO NUMPY
            ####################################################

            embedding = embedding.squeeze().cpu().numpy()

            X.append(embedding)

            y.append(speaker_dir.name)

        except Exception as e:

            print(f"\nFailed : {wav_file}")
            print(e)


############################################################
# SAVE EMBEDDINGS
############################################################

X = np.array(X)
y = np.array(y)

np.save(os.path.join(EMBEDDING_PATH, "X.npy"), X)
np.save(os.path.join(EMBEDDING_PATH, "y.npy"), y)


############################################################
# SUMMARY
############################################################

print("\n" + "=" * 60)
print("Embedding Extraction Completed")
print("=" * 60)

print("Embeddings :", X.shape)
print("Labels     :", y.shape)

print("\nSaved Files")
print("------------------------")
print("embeddings/X.npy")
print("embeddings/y.npy")