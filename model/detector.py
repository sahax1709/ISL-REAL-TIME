"""
Sign Language Classifier using TensorFlow/Keras.
Architecture: Dense NN on MediaPipe hand landmark features.
Supports A-Z, 0-9, and common words.
"""

import os
import json
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils.hindi_mapping import (
    NUM_CLASSES, CLASS_TO_INDEX, INDEX_TO_CLASS, to_display
)
from utils.landmarks import TWO_HAND_FEATURES


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.keras")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")


def build_model(input_dim=TWO_HAND_FEATURES, num_classes=NUM_CLASSES):
    """Build the classifier architecture."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class SignLanguageDetector:
    """High-level wrapper: load model, predict, return bilingual output."""

    def __init__(self):
        self.model = None
        self.is_loaded = False

    def load(self):
        """Load saved model or build a fresh one."""
        if os.path.exists(MODEL_PATH):
            self.model = keras.models.load_model(MODEL_PATH)
            self.is_loaded = True
            print(f"[Model] Loaded from {MODEL_PATH}")
        else:
            print("[Model] No saved model found. Building fresh architecture.")
            self.model = build_model()
            self.is_loaded = False

    def predict(self, landmarks):
        """
        Predict sign from landmark array.
        Args:
            landmarks: np.ndarray of shape (126,) or (63,)
        Returns:
            dict with english, hindi, category, confidence, top_3
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        features = np.array(landmarks, dtype=np.float32)

        # Pad single-hand input to two-hand size
        if features.shape[0] == 63:
            features = np.concatenate([features, np.zeros(63, dtype=np.float32)])

        features = features.reshape(1, -1)
        preds = self.model.predict(features, verbose=0)[0]

        top_idx = np.argmax(preds)
        confidence = float(preds[top_idx])
        label = INDEX_TO_CLASS.get(top_idx, "unknown")

        # Top 3 predictions
        top3_indices = np.argsort(preds)[-3:][::-1]
        top_3 = []
        for idx in top3_indices:
            lbl = INDEX_TO_CLASS.get(idx, "unknown")
            display = to_display(lbl)
            display["confidence"] = round(float(preds[idx]) * 100, 1)
            top_3.append(display)

        result = to_display(label)
        result["confidence"] = round(confidence * 100, 1)
        result["top_3"] = top_3
        return result

    def save(self):
        """Save the current model."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model.save(MODEL_PATH)
        meta = {
            "num_classes": NUM_CLASSES,
            "input_dim": TWO_HAND_FEATURES,
            "classes": {str(k): v for k, v in INDEX_TO_CLASS.items()},
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Model] Saved to {MODEL_PATH}")
