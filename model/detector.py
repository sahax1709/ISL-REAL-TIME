"""
Sign Language Classifier.

Architecture: sklearn RandomForest on engineered landmark features.
  - No TensorFlow needed (zero GPU/CUDA conflicts)
  - Trains in seconds, not minutes
  - Handles 66 classes well with 200+ samples each

False-positive reduction:
  - TemporalSmoother requires N consecutive same predictions
  - Confidence threshold rejects low-certainty guesses
  - Engineered features make similar signs (Q/P, Please/Sorry) more separable
"""

import os
import json
import time
import numpy as np
from collections import deque
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils.hindi_mapping import (
    NUM_CLASSES, CLASS_TO_INDEX, INDEX_TO_CLASS, to_display
)
from utils.landmarks import TOTAL_FEATURES


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")


def build_model():
    """Build a sklearn pipeline: StandardScaler + RandomForest."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])


class TemporalSmoother:
    """
    Require N consecutive identical predictions before accepting.
    This kills single-frame false positives (the main culprit for
    random 'please' / 'Q' ghosts).
    """

    def __init__(self, window=4, agreement_ratio=0.6):
        self.window = window
        self.agreement_ratio = agreement_ratio
        self.buffer = deque(maxlen=window)
        self.last_stable = None
        self.last_stable_time = 0

    def update(self, label, confidence):
        self.buffer.append((label, confidence))

        if len(self.buffer) < 2:
            return None, 0.0

        # Count most common label in buffer
        labels = [b[0] for b in self.buffer]
        confs = [b[1] for b in self.buffer]

        from collections import Counter
        counts = Counter(labels)
        best_label, count = counts.most_common(1)[0]

        ratio = count / len(self.buffer)
        if ratio >= self.agreement_ratio:
            avg_conf = np.mean([c for l, c in self.buffer if l == best_label])
            self.last_stable = best_label
            self.last_stable_time = time.time()
            return best_label, float(avg_conf)

        return None, 0.0

    def reset(self):
        self.buffer.clear()
        self.last_stable = None


class SignLanguageDetector:
    """Load model, predict with temporal smoothing, return bilingual output."""

    def __init__(self, confidence_threshold=40.0, smoothing_window=4):
        self.model = None
        self.is_loaded = False
        self.confidence_threshold = confidence_threshold
        self.smoother = TemporalSmoother(
            window=smoothing_window,
            agreement_ratio=0.6,
        )

    def load(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.is_loaded = True
            print(f"[Model] Loaded from {MODEL_PATH}")
        else:
            print("[Model] No saved model found. Run: python pretrain.py")
            self.model = None
            self.is_loaded = False

    def predict_raw(self, features):
        """
        Raw prediction without smoothing.
        Returns (label, confidence, top_3_list) or None.
        """
        if self.model is None:
            return None

        x = np.array(features, dtype=np.float32).reshape(1, -1)

        # Pad or trim to expected size
        expected = TOTAL_FEATURES
        if x.shape[1] < expected:
            x = np.pad(x, ((0, 0), (0, expected - x.shape[1])))
        elif x.shape[1] > expected:
            x = x[:, :expected]

        proba = self.model.predict_proba(x)[0]
        top_idx = np.argmax(proba)
        confidence = float(proba[top_idx]) * 100

        label = INDEX_TO_CLASS.get(top_idx, "unknown")

        # Top 3
        top3_idx = np.argsort(proba)[-3:][::-1]
        top_3 = []
        for idx in top3_idx:
            lbl = INDEX_TO_CLASS.get(idx, "unknown")
            d = to_display(lbl)
            d["confidence"] = round(float(proba[idx]) * 100, 1)
            top_3.append(d)

        return label, confidence, top_3

    def predict(self, features):
        """
        Smoothed prediction with confidence gating.
        Returns dict with english, hindi, category, confidence, top_3
        or a 'no detection' result.
        """
        raw = self.predict_raw(features)
        if raw is None:
            return {"detected": False, "message": "Model not loaded"}

        label, confidence, top_3 = raw

        # Apply temporal smoothing
        stable_label, stable_conf = self.smoother.update(label, confidence)

        if stable_label is None or stable_conf < self.confidence_threshold:
            return {
                "detected": False,
                "message": "Low confidence or unstable",
                "raw_top_3": top_3,  # still show what it sees
            }

        result = to_display(stable_label)
        result["confidence"] = round(stable_conf, 1)
        result["top_3"] = top_3
        result["detected"] = True
        return result

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        meta = {
            "num_classes": NUM_CLASSES,
            "input_dim": TOTAL_FEATURES,
            "model_type": "RandomForest",
            "classes": {str(k): v for k, v in INDEX_TO_CLASS.items()},
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Model] Saved to {MODEL_PATH}")
