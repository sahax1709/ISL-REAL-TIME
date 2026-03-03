"""
train.py - Train the sign language model on collected data.

Usage:
    python train.py                          # Train on all data in data/train/
    python train.py --epochs 50 --batch 32   # Custom training params

Data format:
    data/train/<class_label>/sample_001.npy  # Each .npy file = (126,) landmark array
"""

import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model.detector import build_model, SignLanguageDetector, MODEL_DIR, MODEL_PATH
from utils.hindi_mapping import CLASS_TO_INDEX, NUM_CLASSES

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "train")


def load_dataset(data_dir=DATA_DIR):
    X, y = [], []
    classes_found = set()

    if not os.path.exists(data_dir):
        print(f"[!] Data directory not found: {data_dir}")
        print("    Run collect.py first to gather training samples.")
        return None, None

    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_name not in CLASS_TO_INDEX:
            print(f"[!] Skipping unknown class: {class_name}")
            continue

        label_idx = CLASS_TO_INDEX[class_name]
        classes_found.add(class_name)
        count = 0

        for fname in os.listdir(class_dir):
            if fname.endswith(".npy"):
                fpath = os.path.join(class_dir, fname)
                try:
                    landmarks = np.load(fpath)
                    if landmarks.shape[0] == 63:
                        landmarks = np.concatenate([landmarks, np.zeros(63)])
                    X.append(landmarks)
                    y.append(label_idx)
                    count += 1
                except Exception as e:
                    print(f"[!] Error loading {fpath}: {e}")

        print(f"  {class_name:>12s}: {count} samples")

    if len(X) == 0:
        print("[!] No training data found.")
        return None, None

    print(f"Total: {len(X)} samples across {len(classes_found)} classes")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train(epochs=100, batch_size=32, val_split=0.2):
    print("=" * 50)
    print("  Sign Language Model Training")
    print("=" * 50)

    X, y = load_dataset()
    if X is None:
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    model = build_model()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Val Accuracy: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

    detector = SignLanguageDetector()
    detector.model = model
    detector.save()
    print("Model saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sign language model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch, val_split=args.val_split)
