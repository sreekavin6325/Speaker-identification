import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

############################################################
# CONFIGURATION
############################################################

EMBEDDING_PATH = "embeddings"
MODEL_PATH = "models"

os.makedirs(MODEL_PATH, exist_ok=True)

############################################################
# LOAD EMBEDDINGS
############################################################

print("=" * 60)
print("Loading Embeddings...")
print("=" * 60)

X = np.load(os.path.join(EMBEDDING_PATH, "X.npy"))
y = np.load(os.path.join(EMBEDDING_PATH, "y.npy"))

print("Embeddings Shape :", X.shape)
print("Labels Shape     :", y.shape)

############################################################
# FEATURE SCALING
############################################################

print("\nScaling ECAPA Embeddings...")

scaler = StandardScaler()

X = scaler.fit_transform(X)

print("Scaling Completed.")

############################################################
# CHECK DATA
############################################################

if len(X) == 0:
    raise ValueError("No embeddings found. Run extract_embeddings.py first.")

############################################################
# LABEL ENCODER
############################################################

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nSpeakers Found:")
print(label_encoder.classes_)

############################################################
# TRAIN / TEST SPLIT
############################################################

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)
np.save("models/X_test.npy", X_test)
np.save("models/y_test.npy", y_test)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

############################################################
# TRAIN SVM
############################################################

print("\nTraining SVM...")

from sklearn.model_selection import GridSearchCV

print("\nSearching Best Hyperparameters...")

param_grid = {
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.1, 0.01]
}

grid = GridSearchCV(
    SVC(probability=True),
    param_grid,
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

print("\nBest Parameters")
print(grid.best_params_)

print("\nBest Cross Validation Accuracy")
print(grid.best_score_)

print("Training Completed.")

############################################################
# PREDICTION
############################################################

y_pred = model.predict(X_test)

############################################################
# EVALUATION
############################################################

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

############################################################
# CLASSIFICATION REPORT
############################################################

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
)

############################################################
# CONFUSION MATRIX
############################################################

print("Confusion Matrix\n")

print(confusion_matrix(y_test, y_pred))

############################################################
# SAVE MODEL
############################################################
joblib.dump(
    model,
    os.path.join(MODEL_PATH, "svm_model.pkl"),
)

joblib.dump(
    scaler,
    os.path.join(MODEL_PATH, "scaler.pkl"),
)

joblib.dump(
    label_encoder,
    os.path.join(MODEL_PATH, "label_encoder.pkl"),
)

print("\nModel Saved Successfully.")

print("models/svm_model.pkl")
print("models/label_encoder.pkl")