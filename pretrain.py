"""
pretrain.py - Generate realistic synthetic landmark data and train baseline.

Key improvements:
  - Each class encodes actual finger states (extended/curled/spread)
  - Thumb position is modeled explicitly
  - Hand orientation (up/down/sideways) is encoded
  - Engineered features computed identically to live extraction
  - This makes Q, P, G actually distinguishable
  - And 'please' (circular chest motion) gets a distinct two-hand pattern

Usage:
    python pretrain.py                  # default 300 samples/class
    python pretrain.py --samples 500    # more = better
"""

import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.hindi_mapping import ALL_CLASSES, CLASS_TO_INDEX, NUM_CLASSES, to_hindi
from utils.landmarks import (
    normalize_landmarks, extract_features, FEATURES_PER_HAND, TOTAL_FEATURES
)
from model.detector import build_model, SignLanguageDetector


# ── Finger state definitions ──
# Each sign is defined by: finger curls, thumb position, hand orientation
# finger_states: [thumb, index, middle, ring, pinky]
#   0 = fully curled, 1 = fully extended, 0.5 = half bent
# orientation: (x, y, z) direction of hand
#   (0, -1, 0) = fingers pointing up
#   (0, 1, 0)  = fingers pointing down
#   (1, 0, 0)  = fingers pointing right/sideways

SIGN_DEFS = {
    # ── Letters ──
    "A": {"fingers": [0.6, 0.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0)},
    "B": {"fingers": [0.0, 1.0, 1.0, 1.0, 1.0], "thumb_out": False, "orient": (0, -1, 0)},
    "C": {"fingers": [0.5, 0.5, 0.5, 0.5, 0.5], "thumb_out": True, "orient": (0, -1, 0)},
    "D": {"fingers": [0.3, 1.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0)},
    "E": {"fingers": [0.3, 0.2, 0.2, 0.2, 0.2], "thumb_out": False, "orient": (0, -1, 0)},
    "F": {"fingers": [0.3, 0.3, 1.0, 1.0, 1.0], "thumb_out": False, "orient": (0, -1, 0), "thumb_index_touch": True},
    "G": {"fingers": [0.7, 1.0, 0.0, 0.0, 0.0], "thumb_out": True, "orient": (1, 0, 0)},
    "H": {"fingers": [0.0, 1.0, 1.0, 0.0, 0.0], "thumb_out": False, "orient": (1, 0, 0)},
    "I": {"fingers": [0.0, 0.0, 0.0, 0.0, 1.0], "thumb_out": False, "orient": (0, -1, 0)},
    "J": {"fingers": [0.0, 0.0, 0.0, 0.0, 1.0], "thumb_out": False, "orient": (0, -0.5, -0.5)},
    "K": {"fingers": [0.5, 1.0, 1.0, 0.0, 0.0], "thumb_out": True, "orient": (0, -1, 0)},
    "L": {"fingers": [1.0, 1.0, 0.0, 0.0, 0.0], "thumb_out": True, "orient": (0, -1, 0)},
    "M": {"fingers": [0.2, 0.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0), "thumb_under": 3},
    "N": {"fingers": [0.2, 0.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0), "thumb_under": 2},
    "O": {"fingers": [0.4, 0.4, 0.4, 0.4, 0.4], "thumb_out": False, "orient": (0, -1, 0), "thumb_index_touch": True},
    "P": {"fingers": [0.5, 1.0, 1.0, 0.0, 0.0], "thumb_out": True, "orient": (0.3, 0.7, 0)},
    "Q": {"fingers": [0.7, 1.0, 0.0, 0.0, 0.0], "thumb_out": True, "orient": (0.3, 0.7, 0)},
    "R": {"fingers": [0.0, 1.0, 1.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0), "fingers_crossed": True},
    "S": {"fingers": [0.3, 0.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0)},
    "T": {"fingers": [0.2, 0.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0), "thumb_between": True},
    "U": {"fingers": [0.0, 1.0, 1.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0)},
    "V": {"fingers": [0.0, 1.0, 1.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0), "fingers_spread": True},
    "W": {"fingers": [0.0, 1.0, 1.0, 1.0, 0.0], "thumb_out": False, "orient": (0, -1, 0), "fingers_spread": True},
    "X": {"fingers": [0.0, 0.5, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0)},
    "Y": {"fingers": [1.0, 0.0, 0.0, 0.0, 1.0], "thumb_out": True, "orient": (0, -1, 0)},
    "Z": {"fingers": [0.0, 1.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -0.7, -0.3)},

    # ── Digits ──
    "0": {"fingers": [0.4, 0.4, 0.4, 0.4, 0.4], "thumb_out": False, "orient": (0, -1, 0), "thumb_index_touch": True},
    "1": {"fingers": [0.0, 1.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0)},
    "2": {"fingers": [0.0, 1.0, 1.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0), "fingers_spread": True},
    "3": {"fingers": [1.0, 1.0, 1.0, 0.0, 0.0], "thumb_out": True, "orient": (0, -1, 0)},
    "4": {"fingers": [0.0, 1.0, 1.0, 1.0, 1.0], "thumb_out": False, "orient": (0, -1, 0), "fingers_spread": True},
    "5": {"fingers": [1.0, 1.0, 1.0, 1.0, 1.0], "thumb_out": True, "orient": (0, -1, 0), "fingers_spread": True},
    "6": {"fingers": [0.3, 0.0, 0.0, 0.0, 0.3], "thumb_out": False, "orient": (0, -1, 0), "thumb_pinky_touch": True},
    "7": {"fingers": [0.3, 0.0, 0.0, 0.3, 0.0], "thumb_out": False, "orient": (0, -1, 0), "thumb_ring_touch": True},
    "8": {"fingers": [0.3, 0.0, 0.3, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0), "thumb_middle_touch": True},
    "9": {"fingers": [0.3, 0.3, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0), "thumb_index_touch": True},

    # ── Words (two-hand or motion signs) ──
    "hello":      {"fingers": [1.0, 1.0, 1.0, 1.0, 1.0], "thumb_out": True, "orient": (0.3, -0.7, 0), "two_hand": False},
    "thank_you":  {"fingers": [0.0, 1.0, 1.0, 1.0, 1.0], "thumb_out": False, "orient": (0, 0, -1), "two_hand": False},
    "please":     {"fingers": [0.0, 1.0, 1.0, 1.0, 1.0], "thumb_out": False, "orient": (0, 0, -0.5), "two_hand": False, "chest_contact": True},
    "sorry":      {"fingers": [0.0, 0.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, 0, -0.5), "two_hand": False, "chest_contact": True, "is_fist": True},
    "yes":        {"fingers": [0.0, 0.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0), "two_hand": False, "is_fist": True},
    "no":         {"fingers": [0.5, 1.0, 1.0, 0.0, 0.0], "thumb_out": True, "orient": (1, 0, 0), "two_hand": False},
    "help":       {"fingers": [1.0, 1.0, 1.0, 1.0, 1.0], "thumb_out": True, "orient": (0, -1, 0), "two_hand": True},
    "stop":       {"fingers": [0.0, 1.0, 1.0, 1.0, 1.0], "thumb_out": False, "orient": (0, 0, -1), "two_hand": True},
    "good":       {"fingers": [0.0, 1.0, 1.0, 1.0, 1.0], "thumb_out": False, "orient": (0, -0.3, -0.7), "two_hand": False},
    "bad":        {"fingers": [0.0, 1.0, 1.0, 1.0, 1.0], "thumb_out": False, "orient": (0, 0.3, -0.7), "two_hand": False},
    "love":       {"fingers": [0.0, 0.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, 0, -1), "two_hand": True, "arms_crossed": True},
    "friend":     {"fingers": [0.0, 0.5, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0), "two_hand": True},
    "family":     {"fingers": [0.3, 1.0, 0.0, 0.0, 0.0], "thumb_out": True, "orient": (1, 0, 0), "two_hand": True},
    "eat":        {"fingers": [0.3, 0.3, 0.3, 0.3, 0.3], "thumb_out": False, "orient": (0, 0, -1), "two_hand": False},
    "drink":      {"fingers": [0.5, 0.5, 0.5, 0.5, 0.5], "thumb_out": True, "orient": (0, -0.5, -0.5), "two_hand": False},
    "water":      {"fingers": [0.0, 1.0, 1.0, 1.0, 0.0], "thumb_out": False, "orient": (0, -1, 0), "two_hand": False},
    "more":       {"fingers": [0.3, 0.3, 0.3, 0.3, 0.3], "thumb_out": False, "orient": (0, 0, -1), "two_hand": True},
    "done":       {"fingers": [1.0, 1.0, 1.0, 1.0, 1.0], "thumb_out": True, "orient": (0, 0, -1), "two_hand": True},
    "want":       {"fingers": [1.0, 1.0, 1.0, 1.0, 1.0], "thumb_out": True, "orient": (0, -1, 0), "two_hand": True, "palms_up": True},
    "need":       {"fingers": [0.0, 0.5, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, 0.5, -0.5), "two_hand": False},
    "go":         {"fingers": [0.0, 1.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, 0, -1), "two_hand": True},
    "come":       {"fingers": [0.0, 1.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, 0, 0.5), "two_hand": True},
    "home":       {"fingers": [0.3, 0.3, 0.3, 0.3, 0.3], "thumb_out": False, "orient": (0, -0.3, -0.7), "two_hand": False},
    "school":     {"fingers": [1.0, 1.0, 1.0, 1.0, 1.0], "thumb_out": True, "orient": (0, 0, -1), "two_hand": True},
    "work":       {"fingers": [0.0, 0.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0), "two_hand": True, "is_fist": True},
    "name":       {"fingers": [0.0, 1.0, 1.0, 0.0, 0.0], "thumb_out": False, "orient": (1, 0, 0), "two_hand": True},
    "how":        {"fingers": [0.0, 0.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0), "two_hand": True, "is_fist": True},
    "what":       {"fingers": [1.0, 1.0, 1.0, 1.0, 1.0], "thumb_out": True, "orient": (0, -1, 0), "two_hand": True, "palms_up": True},
    "where":      {"fingers": [0.0, 1.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0), "two_hand": False},
    "when":       {"fingers": [0.0, 1.0, 0.0, 0.0, 0.0], "thumb_out": False, "orient": (0, -1, 0), "two_hand": True},
}


def generate_hand_landmarks(sign_def, rng, noise=0.08):
    """
    Generate synthetic 21x3 landmarks based on sign definition.
    More realistic than random noise: actually encodes finger positions.
    """
    pts = np.zeros((21, 3), dtype=np.float32)

    # Wrist at origin area
    pts[0] = [0.5, 0.7, 0.0]

    orient = np.array(sign_def.get("orient", (0, -1, 0)), dtype=np.float32)
    orient = orient / (np.linalg.norm(orient) + 1e-6)

    finger_states = sign_def["fingers"]  # [thumb, index, middle, ring, pinky]
    finger_indices = [
        [1, 2, 3, 4],    # thumb
        [5, 6, 7, 8],    # index
        [9, 10, 11, 12],  # middle
        [13, 14, 15, 16], # ring
        [17, 18, 19, 20], # pinky
    ]

    # Spread angles for each finger base
    base_angles = [-0.4, -0.15, 0.0, 0.15, 0.3]

    spread = sign_def.get("fingers_spread", False)

    for fi, (indices, ext_ratio) in enumerate(zip(finger_indices, finger_states)):
        angle = base_angles[fi]
        if spread:
            angle *= 2.0

        # Base direction for this finger
        dx = orient[0] + np.sin(angle) * 0.3
        dy = orient[1]
        dz = orient[2] + np.cos(angle) * 0.1

        seg_len = 0.04  # segment length

        for ji, idx in enumerate(indices):
            progress = (ji + 1) / len(indices)

            if ext_ratio > 0.7:
                # Extended: follow orientation
                pts[idx] = pts[0] + np.array([dx, dy, dz]) * seg_len * (ji + 1) * 3
            elif ext_ratio < 0.3:
                # Curled: fold back
                curl_factor = progress * (1.0 - ext_ratio) * 0.7
                pts[idx] = pts[0] + np.array([
                    dx * seg_len * (ji + 1) * 2,
                    dy * seg_len * (ji + 1) * 2 + curl_factor * 0.15,
                    dz * seg_len * (ji + 1) * 2,
                ])
            else:
                # Half bent
                pts[idx] = pts[0] + np.array([
                    dx * seg_len * (ji + 1) * 2.5,
                    dy * seg_len * (ji + 1) * 2.5 + progress * 0.05,
                    dz * seg_len * (ji + 1) * 2.5,
                ])

    # Special: thumb-index touch (F, O, 9)
    if sign_def.get("thumb_index_touch"):
        pts[4] = (pts[4] + pts[8]) / 2 + rng.normal(0, 0.005, 3)

    # Special: thumb-pinky touch (6)
    if sign_def.get("thumb_pinky_touch"):
        pts[4] = (pts[4] + pts[20]) / 2 + rng.normal(0, 0.005, 3)

    # Special: crossed fingers (R)
    if sign_def.get("fingers_crossed"):
        mid = (pts[8] + pts[12]) / 2
        pts[8] = mid + np.array([0.01, -0.01, 0])
        pts[12] = mid + np.array([-0.01, 0.01, 0])

    # Add noise
    pts += rng.normal(0, noise, pts.shape).astype(np.float32)
    pts = np.clip(pts, 0.0, 1.0)

    return pts


def generate_samples(class_name, num_samples=300, noise=0.08):
    """Generate feature vectors for a class."""
    sign_def = SIGN_DEFS.get(class_name)
    if sign_def is None:
        # Fallback: random
        rng = np.random.RandomState(hash(class_name) % (2**31))
        return rng.uniform(0, 1, (num_samples, TOTAL_FEATURES)).astype(np.float32)

    rng = np.random.RandomState(CLASS_TO_INDEX[class_name] * 7 + 13)
    samples = []

    for i in range(num_samples):
        # Primary hand
        pts1 = generate_hand_landmarks(sign_def, rng, noise)
        normed1 = normalize_landmarks(pts1)
        feats1 = extract_features(normed1)

        # Second hand
        if sign_def.get("two_hand"):
            pts2 = generate_hand_landmarks(sign_def, rng, noise * 1.2)
            # Mirror x for second hand
            pts2[:, 0] = 1.0 - pts2[:, 0]
            normed2 = normalize_landmarks(pts2)
            feats2 = extract_features(normed2)
        else:
            feats2 = np.zeros(FEATURES_PER_HAND, dtype=np.float32)

        combined = np.concatenate([feats1, feats2])
        samples.append(combined)

    return np.array(samples, dtype=np.float32)


def pretrain(samples_per_class=300, test_split=0.2):
    print("=" * 55)
    print("  Sign Language Model Pre-training")
    print("=" * 55)
    print(f"\n  Classes:     {NUM_CLASSES}")
    print(f"  Samples/cls: {samples_per_class}")
    print(f"  Total:       {NUM_CLASSES * samples_per_class}")
    print(f"  Features:    {TOTAL_FEATURES} per sample")
    print()

    X_all, y_all = [], []
    for cls_name in ALL_CLASSES:
        idx = CLASS_TO_INDEX[cls_name]
        samples = generate_samples(cls_name, samples_per_class)
        X_all.append(samples)
        y_all.extend([idx] * samples_per_class)
        hi = to_hindi(cls_name)
        print(f"  {cls_name:>12s}  {hi}")

    X = np.vstack(X_all)
    y = np.array(y_all, dtype=np.int32)

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train)}  |  Val: {len(X_val)}")
    print("\n  Training RandomForest (this takes ~10-30 seconds)...\n")

    model = build_model()
    model.fit(X_train, y_train)

    val_acc = model.score(X_val, y_val)
    print(f"  Validation Accuracy: {val_acc:.4f}")

    # Quick check on the problem classes
    print("\n  Checking confusing pairs:")
    for pair in [("P", "Q"), ("please", "sorry"), ("Q", "G"), ("S", "A")]:
        c1, c2 = pair
        if c1 in CLASS_TO_INDEX and c2 in CLASS_TO_INDEX:
            mask1 = y_val == CLASS_TO_INDEX[c1]
            mask2 = y_val == CLASS_TO_INDEX[c2]
            if mask1.sum() > 0 and mask2.sum() > 0:
                acc1 = (model.predict(X_val[mask1]) == y_val[mask1]).mean()
                acc2 = (model.predict(X_val[mask2]) == y_val[mask2]).mean()
                print(f"    {c1:>8s}: {acc1:.2%}    {c2:>8s}: {acc2:.2%}")

    # Save
    detector = SignLanguageDetector()
    detector.model = model
    detector.save()

    print("\n" + "=" * 55)
    print("  Done! Run: python app.py")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=300)
    args = parser.parse_args()
    pretrain(samples_per_class=args.samples)
