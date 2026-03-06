"""
pretrain.py - Synthetic data + baseline model training.

Usage:
    python pretrain.py
    python pretrain.py --samples 500
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.hindi_mapping import ALL_CLASSES, CLASS_TO_INDEX, NUM_CLASSES, to_hindi
from utils.landmarks import (
    normalize_landmarks, extract_features, FEATURES_PER_HAND, TOTAL_FEATURES
)
from model.detector import build_model, SignLanguageDetector

# Finger state encoding per sign
# [thumb, index, middle, ring, pinky] : 0=curled, 1=extended, 0.5=half
# orient: (x,y) dominant direction. y<0=up, y>0=down, x>0=sideways
SIGNS = {
    "A": ([0.6, 0, 0, 0, 0], (0, -1)),
    "B": ([0, 1, 1, 1, 1], (0, -1)),
    "C": ([.5, .5, .5, .5, .5], (0, -1)),
    "D": ([.3, 1, 0, 0, 0], (0, -1)),
    "E": ([.3, .2, .2, .2, .2], (0, -1)),
    "F": ([.3, .3, 1, 1, 1], (0, -1)),
    "G": ([.7, 1, 0, 0, 0], (1, 0)),
    "H": ([0, 1, 1, 0, 0], (1, 0)),
    "I": ([0, 0, 0, 0, 1], (0, -1)),
    "J": ([0, 0, 0, 0, 1], (0, -.5)),
    "K": ([.5, 1, 1, 0, 0], (0, -1)),
    "L": ([1, 1, 0, 0, 0], (0, -1)),
    "M": ([.15, 0, 0, 0, 0], (0, -1)),
    "N": ([.2, 0, 0, 0, 0], (0, -1)),
    "O": ([.4, .4, .4, .4, .4], (0, -1)),
    "P": ([.5, 1, 1, 0, 0], (.3, .7)),
    "Q": ([.7, 1, 0, 0, 0], (.3, .7)),
    "R": ([0, 1, 1, 0, 0], (0, -1)),
    "S": ([.3, 0, 0, 0, 0], (0, -1)),
    "T": ([.2, 0, 0, 0, 0], (0, -.9)),
    "U": ([0, 1, 1, 0, 0], (0, -1)),
    "V": ([0, 1, 1, 0, 0], (0, -1)),
    "W": ([0, 1, 1, 1, 0], (0, -1)),
    "X": ([0, .5, 0, 0, 0], (0, -1)),
    "Y": ([1, 0, 0, 0, 1], (0, -1)),
    "Z": ([0, 1, 0, 0, 0], (0, -.7)),
    "0": ([.4, .4, .4, .4, .4], (0, -1)),
    "1": ([0, 1, 0, 0, 0], (0, -1)),
    "2": ([0, 1, 1, 0, 0], (0, -1)),
    "3": ([1, 1, 1, 0, 0], (0, -1)),
    "4": ([0, 1, 1, 1, 1], (0, -1)),
    "5": ([1, 1, 1, 1, 1], (0, -1)),
    "6": ([.3, 0, 0, 0, .3], (0, -1)),
    "7": ([.3, 0, 0, .3, 0], (0, -1)),
    "8": ([.3, 0, .3, 0, 0], (0, -1)),
    "9": ([.3, .3, 0, 0, 0], (0, -1)),
    "hello": ([1, 1, 1, 1, 1], (.3, -.7)),
    "thank_you": ([0, 1, 1, 1, 1], (0, -.3)),
    "please": ([0, 1, 1, 1, 1], (0, 0)),
    "sorry": ([0, 0, 0, 0, 0], (0, 0)),
    "yes": ([0, 0, 0, 0, 0], (0, -1)),
    "no": ([.5, 1, 1, 0, 0], (1, 0)),
    "help": ([1, 1, 1, 1, 1], (0, -1)),
    "stop": ([0, 1, 1, 1, 1], (0, 0)),
    "good": ([0, 1, 1, 1, 1], (0, -.5)),
    "bad": ([0, 1, 1, 1, 1], (0, .3)),
    "love": ([0, 0, 0, 0, 0], (0, 0)),
    "friend": ([0, .5, 0, 0, 0], (0, -1)),
    "family": ([.3, 1, 0, 0, 0], (1, 0)),
    "eat": ([.3, .3, .3, .3, .3], (0, 0)),
    "drink": ([.5, .5, .5, .5, .5], (0, -.5)),
    "water": ([0, 1, 1, 1, 0], (0, -1)),
    "more": ([.3, .3, .3, .3, .3], (0, -.2)),
    "done": ([1, 1, 1, 1, 1], (0, 0)),
    "want": ([1, 1, 1, 1, 1], (0, -.8)),
    "need": ([0, .5, 0, 0, 0], (0, .5)),
    "go": ([0, 1, 0, 0, 0], (0, -.2)),
    "come": ([0, 1, 0, 0, 0], (0, .2)),
    "home": ([.3, .3, .3, .3, .3], (0, -.3)),
    "school": ([1, 1, 1, 1, 1], (0, -.1)),
    "work": ([0, 0, 0, 0, 0], (0, -.8)),
    "name": ([0, 1, 1, 0, 0], (1, 0)),
    "how": ([0, 0, 0, 0, 0], (0, -.6)),
    "what": ([1, 1, 1, 1, 1], (0, -.9)),
    "where": ([0, 1, 0, 0, 0], (0, -1)),
    "when": ([0, 1, 0, 0, 0], (0, -.9)),
}

# Which signs use two hands
TWO_HAND = {
    "help", "stop", "love", "friend", "family", "more", "done",
    "want", "go", "come", "school", "work", "name", "how", "what", "when",
}


def make_hand(finger_states, orient, rng, noise=0.06):
    """Generate synthetic 21x3 hand landmarks from finger states + orientation."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.7, 0.0]  # wrist

    ox, oy = orient
    base_angles = [-0.4, -0.15, 0.0, 0.15, 0.3]
    seg = 0.04

    fingers = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]

    for fi, (indices, ext) in enumerate(zip(fingers, finger_states)):
        ang = base_angles[fi]
        dx = ox + np.sin(ang) * 0.3
        dy = oy
        for ji, idx in enumerate(indices):
            prog = (ji + 1) / len(indices)
            base_pos = np.array([
                dx * seg * (ji + 1) * 2.5,
                dy * seg * (ji + 1) * 2.5,
                0.0
            ])
            if ext > 0.7:
                pts[idx] = pts[0] + base_pos
            elif ext < 0.3:
                curl = prog * (1.0 - ext) * 0.15
                pts[idx] = pts[0] + base_pos * 0.7 + np.array([0, curl, 0])
            else:
                pts[idx] = pts[0] + base_pos * 0.85 + np.array([0, prog * 0.04, 0])

    pts += rng.normal(0, noise, pts.shape).astype(np.float32)
    return np.clip(pts, 0.0, 1.0)


def generate_class(cls_name, n=300, noise=0.06):
    """Generate n feature vectors for a class."""
    rng = np.random.RandomState(CLASS_TO_INDEX[cls_name] * 7 + 13)

    if cls_name not in SIGNS:
        return rng.uniform(0, 1, (n, TOTAL_FEATURES)).astype(np.float32)

    finger_states, orient = SIGNS[cls_name]
    use_two = cls_name in TWO_HAND
    samples = []

    for _ in range(n):
        # Vary noise per sample for diversity
        sample_noise = noise * rng.uniform(0.5, 1.5)

        pts1 = make_hand(finger_states, orient, rng, sample_noise)
        n1 = normalize_landmarks(pts1)
        f1 = extract_features(n1)

        if use_two:
            pts2 = make_hand(finger_states, orient, rng, sample_noise * 1.2)
            pts2[:, 0] = 1.0 - pts2[:, 0]  # mirror
            n2 = normalize_landmarks(pts2)
            f2 = extract_features(n2)
        else:
            f2 = np.zeros(FEATURES_PER_HAND, dtype=np.float32)

        samples.append(np.concatenate([f1, f2]))

    return np.array(samples, dtype=np.float32)


def pretrain(samples_per_class=300):
    print("=" * 55)
    print("  Sign Language Model Pre-training")
    print("=" * 55)
    print(f"\n  Classes:     {NUM_CLASSES}")
    print(f"  Samples/cls: {samples_per_class}")
    print(f"  Features:    {TOTAL_FEATURES}")
    print()

    X_all, y_all = [], []
    for cls in ALL_CLASSES:
        idx = CLASS_TO_INDEX[cls]
        data = generate_class(cls, samples_per_class)
        X_all.append(data)
        y_all.extend([idx] * samples_per_class)
        print(f"  {cls:>12s}  {to_hindi(cls)}")

    X = np.vstack(X_all)
    y = np.array(y_all, dtype=np.int32)

    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    split = int(0.8 * len(X))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    print(f"\n  Train: {len(X_tr)}  Val: {len(X_val)}")
    print("  Training... (10-30 seconds)\n")

    model = build_model()
    model.fit(X_tr, y_tr)

    acc = model.score(X_val, y_val)
    print(f"  Validation Accuracy: {acc:.4f}")

    # Check confusing pairs
    for c1, c2 in [("P","Q"), ("G","Q"), ("S","A"), ("please","sorry"), ("U","V"), ("1","D")]:
        if c1 in CLASS_TO_INDEX and c2 in CLASS_TO_INDEX:
            m1 = y_val == CLASS_TO_INDEX[c1]
            m2 = y_val == CLASS_TO_INDEX[c2]
            if m1.sum() > 0 and m2.sum() > 0:
                a1 = (model.predict(X_val[m1]) == y_val[m1]).mean()
                a2 = (model.predict(X_val[m2]) == y_val[m2]).mean()
                print(f"    {c1:>8s}: {a1:.0%}   {c2:>8s}: {a2:.0%}")

    det = SignLanguageDetector()
    det.model = model
    det.save()

    print(f"\n  Done! Run:  python app.py")
    print("=" * 55)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=300)
    pretrain(p.parse_args().samples)
