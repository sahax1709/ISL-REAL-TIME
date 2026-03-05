"""
train.py - Train on real collected data (from collect.py).

Usage:
    python train.py
    python train.py --augment     # with data augmentation

Data: data/train/<class>/*.npy (each file = feature vector)
"""

import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(__file__))

from utils.hindi_mapping import CLASS_TO_INDEX, INDEX_TO_CLASS, NUM_CLASSES
from utils.landmarks import TOTAL_FEATURES
from model.detector import build_model, SignLanguageDetector

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "train")


def augment_sample(features, rng, noise=0.03):
    """Light augmentation: add small noise, scale slightly."""
    aug = features.copy()
    aug += rng.normal(0, noise, aug.shape).astype(np.float32)
    scale = rng.uniform(0.95, 1.05)
    aug *= scale
    return aug


def load_data(augment=False, aug_factor=3):
    X, y = [], []
    rng = np.random.RandomState(42)

    if not os.path.exists(DATA_DIR):
        print(f"No data at {DATA_DIR}. Run collect.py first.")
        return None, None

    for cls in sorted(os.listdir(DATA_DIR)):
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_dir) or cls not in CLASS_TO_INDEX:
            continue

        idx = CLASS_TO_INDEX[cls]
        count = 0

        for f in os.listdir(cls_dir):
            if not f.endswith(".npy"):
                continue
            try:
                feat = np.load(os.path.join(cls_dir, f))
                if feat.shape[0] < TOTAL_FEATURES:
                    feat = np.pad(feat, (0, TOTAL_FEATURES - feat.shape[0]))
                X.append(feat[:TOTAL_FEATURES])
                y.append(idx)
                count += 1

                if augment:
                    for _ in range(aug_factor):
                        X.append(augment_sample(feat[:TOTAL_FEATURES], rng))
                        y.append(idx)
            except Exception as e:
                print(f"  Error: {f}: {e}")

        print(f"  {cls:>12s}: {count} samples" + (f" (+{count*aug_factor} aug)" if augment else ""))

    if not X:
        print("No data found.")
        return None, None

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train(augment=False):
    print("=" * 50)
    print("  Training on Real Data")
    print("=" * 50 + "\n")

    X, y = load_data(augment=augment)
    if X is None:
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}  |  Val: {len(X_val)}\n")

    model = build_model()
    model.fit(X_train, y_train)

    acc = model.score(X_val, y_val)
    print(f"Validation Accuracy: {acc:.4f}\n")

    # Per-class report
    y_pred = model.predict(X_val)
    labels_present = sorted(set(y_val))
    names = [INDEX_TO_CLASS[i] for i in labels_present]
    print(classification_report(y_val, y_pred, labels=labels_present, target_names=names))

    det = SignLanguageDetector()
    det.model = model
    det.save()
    print("\nDone! Restart app.py to use the new model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", action="store_true", help="Augment data 3x")
    args = parser.parse_args()
    train(augment=args.augment)
