"""
train.py - Retrain on real collected data.

Usage:
    python train.py
    python train.py --augment
"""

import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.hindi_mapping import CLASS_TO_INDEX, INDEX_TO_CLASS
from utils.landmarks import TOTAL_FEATURES
from model.detector import build_model, SignLanguageDetector

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train")


def load_data(augment=False, aug_n=3):
    X, y = [], []
    rng = np.random.RandomState(42)

    if not os.path.exists(DATA_DIR):
        print(f"No data at {DATA_DIR}. Run collect.py first.")
        return None, None

    for cls in sorted(os.listdir(DATA_DIR)):
        cdir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cdir) or cls not in CLASS_TO_INDEX:
            continue

        idx = CLASS_TO_INDEX[cls]
        count = 0
        for f in sorted(os.listdir(cdir)):
            if not f.endswith(".npy"):
                continue
            try:
                feat = np.load(os.path.join(cdir, f)).astype(np.float32)
                if feat.shape[0] < TOTAL_FEATURES:
                    feat = np.pad(feat, (0, TOTAL_FEATURES - feat.shape[0]))
                feat = feat[:TOTAL_FEATURES]
                X.append(feat)
                y.append(idx)
                count += 1
                if augment:
                    for _ in range(aug_n):
                        aug = feat + rng.normal(0, 0.03, feat.shape).astype(np.float32)
                        aug *= rng.uniform(0.95, 1.05)
                        X.append(aug)
                        y.append(idx)
            except Exception as e:
                print(f"  Skip {f}: {e}")

        extra = f" (+{count*aug_n} aug)" if augment else ""
        print(f"  {cls:>12s}: {count}{extra}")

    if not X:
        print("No data.")
        return None, None
    return np.array(X), np.array(y)


def train(augment=False):
    print("=" * 50)
    print("  Training on Real Data")
    print("=" * 50 + "\n")

    X, y = load_data(augment=augment)
    if X is None:
        return

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTrain: {len(X_tr)}  Val: {len(X_val)}\n")

    model = build_model()
    model.fit(X_tr, y_tr)
    print(f"Accuracy: {model.score(X_val, y_val):.4f}\n")

    y_pred = model.predict(X_val)
    labels = sorted(set(y_val))
    names = [INDEX_TO_CLASS[i] for i in labels]
    print(classification_report(y_val, y_pred, labels=labels, target_names=names))

    det = SignLanguageDetector()
    det.model = model
    det.save()
    print("Done! Restart app.py to use new model.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--augment", action="store_true")
    train(p.parse_args().augment)
