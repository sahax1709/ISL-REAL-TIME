"""
Sign language classifier: sklearn RandomForest + temporal smoothing.
"""

import os
import json
import time
import numpy as np
from collections import deque, Counter
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils.hindi_mapping import NUM_CLASSES, CLASS_TO_INDEX, INDEX_TO_CLASS, to_display
from utils.landmarks import TOTAL_FEATURES

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")


def build_model():
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
    """Require N frames of agreement before marking 'stable'."""

    def __init__(self, window=3, agreement_ratio=0.5):
        self.window = window
        self.agreement_ratio = agreement_ratio
        self.buffer = deque(maxlen=window)

    def update(self, label, confidence):
        self.buffer.append((label, confidence))
        if len(self.buffer) < 2:
            return None, 0.0

        labels = [b[0] for b in self.buffer]
        counts = Counter(labels)
        best, count = counts.most_common(1)[0]

        if count / len(self.buffer) >= self.agreement_ratio:
            avg = np.mean([c for l, c in self.buffer if l == best])
            return best, float(avg)
        return None, 0.0

    def reset(self):
        self.buffer.clear()


class SignLanguageDetector:

    def __init__(self, confidence_threshold=30.0, smoothing_window=3):
        self.model = None
        self.is_loaded = False
        self.confidence_threshold = confidence_threshold
        self.smoother = TemporalSmoother(window=smoothing_window, agreement_ratio=0.5)

    def load(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.is_loaded = True
            print(f"[Model] Loaded from {MODEL_PATH}")
        else:
            print("[Model] No saved model. Run: python pretrain.py")
            self.is_loaded = False

    def _get_raw(self, features):
        """Get raw prediction + top 3. Always works if model is loaded."""
        if self.model is None:
            return None

        x = np.array(features, dtype=np.float32).reshape(1, -1)

        # Ensure correct width
        if x.shape[1] < TOTAL_FEATURES:
            x = np.pad(x, ((0, 0), (0, TOTAL_FEATURES - x.shape[1])))
        elif x.shape[1] > TOTAL_FEATURES:
            x = x[:, :TOTAL_FEATURES]

        proba = self.model.predict_proba(x)[0]
        top3_idx = np.argsort(proba)[-3:][::-1]

        top_3 = []
        for idx in top3_idx:
            lbl = INDEX_TO_CLASS.get(int(idx), "unknown")
            d = to_display(lbl)
            d["confidence"] = round(float(proba[idx]) * 100, 1)
            top_3.append(d)

        best_idx = top3_idx[0]
        label = INDEX_TO_CLASS.get(int(best_idx), "unknown")
        confidence = float(proba[best_idx]) * 100

        return label, confidence, top_3

    def predict(self, features):
        """
        Returns dict. Always includes raw_top_3.
        detected=True only when smoother confirms stability.
        """
        raw = self._get_raw(features)
        if raw is None:
            return {"detected": False, "message": "Model not loaded", "raw_top_3": []}

        label, confidence, top_3 = raw

        # Always provide raw results so the UI can show something
        base = {"raw_top_3": top_3}

        # Feed into smoother
        stable_label, stable_conf = self.smoother.update(label, confidence)

        if stable_label and stable_conf >= self.confidence_threshold:
            result = to_display(stable_label)
            result["confidence"] = round(stable_conf, 1)
            result["top_3"] = top_3
            result["detected"] = True
            result["raw_top_3"] = top_3
            return result

        base["detected"] = False
        base["message"] = "Stabilizing..."
        return base

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        meta = {
            "num_classes": NUM_CLASSES,
            "input_dim": TOTAL_FEATURES,
            "classes": {str(k): v for k, v in INDEX_TO_CLASS.items()},
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Model] Saved to {MODEL_PATH}")
