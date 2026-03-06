"""
collect.py - Collect training data via webcam.

Usage:
    python collect.py                  # interactive
    python collect.py --class A        # specific class
    python collect.py --samples 100

Controls: SPACE=record, Q=quit
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from utils.hindi_mapping import ALL_CLASSES, to_hindi
from utils.landmarks import LandmarkExtractor

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train")


def collect(class_name, num_samples=50, extractor=None):
    cdir = os.path.join(DATA_DIR, class_name)
    os.makedirs(cdir, exist_ok=True)
    existing = len([f for f in os.listdir(cdir) if f.endswith(".npy")])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return 0

    recording = False
    collected = 0

    print(f"\nClass: {class_name} ({to_hindi(class_name)})")
    print(f"Existing: {existing} | Target: +{num_samples}")
    print("SPACE=record  Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        result = extractor.process_frame(frame)
        display = result["annotated"]

        color = (0, 0, 255) if recording else (180, 180, 180)
        label = "REC" if recording else "PAUSED"
        cv2.putText(display, f"{class_name} ({to_hindi(class_name)})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, label, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f"{existing+collected}/{existing+num_samples}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if recording and result["features"] is not None:
            np.save(os.path.join(cdir, f"s_{existing+collected:04d}.npy"), result["features"])
            collected += 1
            if collected >= num_samples:
                print(f"Done: {collected} samples")
                break

        cv2.imshow("Collect", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            recording = not recording
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return collected


def interactive(num_samples):
    ext = LandmarkExtractor()
    print(f"\nClasses ({len(ALL_CLASSES)}):")
    for i, c in enumerate(ALL_CLASSES):
        print(f"  {i+1:3d}. {c:>12s} {to_hindi(c)}", end="")
        if (i + 1) % 3 == 0:
            print()
    print("\n")

    while True:
        choice = input("Class (or 'quit'): ").strip()
        if choice.lower() == "quit":
            break
        c = choice if choice in ALL_CLASSES else choice.upper()
        if c not in ALL_CLASSES:
            c = choice.lower()
        if c in ALL_CLASSES:
            collect(c, num_samples, ext)
        else:
            print(f"Unknown: {choice}")
    ext.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--class-name", type=str, default=None)
    p.add_argument("--samples", type=int, default=50)
    a = p.parse_args()

    if a.class_name:
        ext = LandmarkExtractor()
        collect(a.class_name, a.samples, ext)
        ext.close()
    else:
        interactive(a.samples)
