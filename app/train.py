"""
Training Script — loads data, trains model, saves to models/.
Usage: python -m app.train --epochs 50 --batch-size 32 --lr 0.001
"""

import os, json, argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from app.model import build_landmark_model

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")

def load_dataset():
    X, y, labels = [], [], []
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"No data at {DATA_DIR}. Collect data first.")
    for label_dir in sorted(DATA_DIR.iterdir()):
        if not label_dir.is_dir(): continue
        labels.append(label_dir.name)
        idx = len(labels) - 1
        samples = list(label_dir.glob("*.npy"))
        print(f"  {label_dir.name}: {len(samples)} samples")
        for sp in samples:
            X.append(np.load(str(sp))); y.append(idx)
    if not X: raise ValueError("No samples found.")
    return np.array(X), np.array(y), labels

def train(epochs=50, batch_size=32, lr=0.001, val_split=0.2, patience=10):
    print("\n" + "="*50 + "\n  Sign Language Model Training\n" + "="*50)

    print("\n[1/4] Loading dataset...")
    X, y, labels = load_dataset()
    num_classes = len(labels)
    print(f"Total: {len(X)} samples, {num_classes} classes")

    print("\n[2/4] Splitting...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42, stratify=y)
    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(cw))

    print("\n[3/4] Training...")
    model = build_landmark_model(num_classes=num_classes, input_dim=X_train.shape[1], learning_rate=lr)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        class_weight=class_weight_dict, callbacks=callbacks, verbose=1)

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n  Val Accuracy: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

    print("\n[4/4] Saving model...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_DIR / "sign_model.keras"))
    with open(MODEL_DIR / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    meta = {"num_classes": num_classes, "labels": labels, "total_samples": len(X),
            "val_accuracy": float(val_acc), "epochs_trained": len(history.history["loss"]),
            "input_dim": int(X_train.shape[1])}
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved to {MODEL_DIR}/\n  Labels: {labels}\n")
    return history

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=10)
    a = p.parse_args()
    train(a.epochs, a.batch_size, a.lr, a.val_split, a.patience)

if __name__ == "__main__":
    main()
