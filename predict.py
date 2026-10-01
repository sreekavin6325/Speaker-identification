import time
import joblib
import numpy as np

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

print("Loading Models...")

svm_model = joblib.load("models/svm_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/ecapa",
    local_strategy=LocalStrategy.COPY
)

print("Models Loaded Successfully")



def predict_speaker(audio_path):

    start = time.perf_counter()

    signal = classifier.load_audio(audio_path)

    embedding = classifier.encode_batch(signal.unsqueeze(0))

    embedding = embedding.squeeze().cpu().detach().numpy().reshape(1, -1)

    embedding = scaler.transform(embedding)

    prediction = svm_model.predict(embedding)[0]

    probabilities = svm_model.predict_proba(embedding)[0]

    confidence = np.max(probabilities) * 100

    speaker = label_encoder.inverse_transform([prediction])[0]

    inference_time = (time.perf_counter() - start) * 1000

    class_probabilities = []

    for class_id, probability in zip(
        svm_model.classes_,
        probabilities
    ):
        class_name = label_encoder.inverse_transform(
            [class_id]
        )[0]

        class_probabilities.append(
            (
                class_name,
                float(probability) * 100
            )
        )

    class_probabilities.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return (
        speaker,
        confidence,
        inference_time,
        class_probabilities
    )
def get_speaker_classes():
    return list(label_encoder.classes_)