"""
pretrain.py - Generate synthetic landmark data and train a baseline model.

Creates a starter model so the app runs immediately.
Replace with real data (via collect.py) for production accuracy.

Usage:
    python pretrain.py
    python pretrain.py --samples 500
"""

import os
import argparse
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils.hindi_mapping import ALL_CLASSES, CLASS_TO_INDEX, NUM_CLASSES, to_hindi
from utils.landmarks import TWO_HAND_FEATURES
from model.detector import build_model, SignLanguageDetector


def generate_synthetic_landmarks(class_idx, num_samples=200, noise=0.15):
    rng = np.random.RandomState(class_idx * 42)
    base = rng.uniform(0.2, 0.8, size=(TWO_HAND_FEATURES,)).astype(np.float32)

    finger_ranges = [(0, 5), (5, 9), (9, 13), (13, 17), (17, 21)]
    for i, (start, end) in enumerate(finger_ranges):
        finger_state = (class_idx >> i) & 1
        for j in range(start, end):
            idx = j * 3
            if finger_state:
                base[idx + 1] = rng.uniform(0.2, 0.4)
            else:
                base[idx + 1] = rng.uniform(0.6, 0.8)

    samples = []
    for _ in range(num_samples):
        sample = base + np.random.normal(0, noise, size=base.shape).astype(np.float32)
        sample = np.clip(sample, 0.0, 1.0)
        samples.append(sample)

    return np.array(samples)


def pretrain(samples_per_class=200, epochs=50, batch_size=64):
    print("=" * 50)
    print("  Pre-training Sign Language Model")
    print("=" * 50)
    print(f"\nClasses: {NUM_CLASSES}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Total samples: {NUM_CLASSES * samples_per_class}\n")

    X_all, y_all = [], []
    for cls_name in ALL_CLASSES:
        idx = CLASS_TO_INDEX[cls_name]
        samples = generate_synthetic_landmarks(idx, samples_per_class)
        X_all.append(samples)
        y_all.extend([idx] * samples_per_class)
        hindi = to_hindi(cls_name)
        print(f"  Generated: {cls_name:>12s} ({hindi})")

    X = np.vstack(X_all)
    y = np.array(y_all, dtype=np.int32)

    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)}")

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, verbose=1
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nVal Accuracy: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

    detector = SignLanguageDetector()
    detector.model = model
    detector.save()

    data_dir = os.path.join("data", "synthetic")
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "X_synthetic.npy"), X)
    np.save(os.path.join(data_dir, "y_synthetic.npy"), y)

    print("\n" + "=" * 50)
    print("  Pre-training Complete!")
    print("  Run the app:       python app.py")
    print("  Improve accuracy:  python collect.py -> python train.py")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    pretrain(samples_per_class=args.samples, epochs=args.epochs)
