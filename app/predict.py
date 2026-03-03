"""
Prediction Module
Loads the trained model + label map and runs inference on
images or raw landmarks.
"""

import json
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from tensorflow import keras


class SignLanguagePredictor:
    """Wraps model loading, landmark extraction, and prediction."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.labels = []
        self._load()

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )

    # ── Loading ───────────────────────────────────────────────────────

    def _load(self):
        model_path = self.model_dir / "sign_model.keras"
        labels_path = self.model_dir / "labels.json"

        if model_path.exists() and labels_path.exists():
            self.model = keras.models.load_model(str(model_path))
            with open(labels_path) as f:
                self.labels = json.load(f)
            print(f"Loaded model with {len(self.labels)} classes: {self.labels}")
        else:
            print(f"No model found at {model_path}. Train the model first.")
            self.model = None
            self.labels = []

    def is_ready(self) -> bool:
        return self.model is not None and len(self.labels) > 0

    def get_labels(self) -> list:
        return self.labels

    # ── Landmark Extraction ───────────────────────────────────────────

    def _extract_landmarks(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Extract 21 hand landmarks (63 values) from a BGR image."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        return np.array(landmarks, dtype=np.float32)

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks relative to the wrist (landmark 0)."""
        lm = landmarks.reshape(21, 3)
        wrist = lm[0].copy()
        lm = lm - wrist  # center on wrist

        # Scale to unit bounding box
        max_val = np.max(np.abs(lm))
        if max_val > 0:
            lm = lm / max_val

        return lm.flatten()

    # ── Prediction ────────────────────────────────────────────────────

    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        """Predict from raw image bytes."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Could not decode image."}

        landmarks = self._extract_landmarks(image)
        if landmarks is None:
            return {"error": "No hand detected in image."}

        return self._run_prediction(landmarks)

    def predict_from_landmarks(self, landmarks: list) -> dict:
        """Predict from pre-extracted landmark list (63 values)."""
        lm = np.array(landmarks, dtype=np.float32)
        return self._run_prediction(lm)

    def _run_prediction(self, landmarks: np.ndarray) -> dict:
        """Run model inference on normalized landmarks."""
        normalized = self._normalize_landmarks(landmarks)
        input_data = normalized.reshape(1, -1)

        predictions = self.model.predict(input_data, verbose=0)[0]
        class_idx = int(np.argmax(predictions))
        confidence = float(predictions[class_idx])

        return {
            "prediction": self.labels[class_idx],
            "confidence": round(confidence, 4),
            "all_probabilities": {
                label: round(float(predictions[i]), 4)
                for i, label in enumerate(self.labels)
            },
        }
