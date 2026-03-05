"""
collect.py - Collect real training data via webcam.

Usage:
    python collect.py                    # interactive
    python collect.py --class A          # specific class
    python collect.py --samples 100      # samples per class

Controls:
    SPACE = start/stop recording
    Q     = quit current class
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import cv2
from utils.hindi_mapping import ALL_CLASSES, to_hindi
from utils.landmarks import LandmarkExtractor

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "train")


def collect_for_class(class_name, num_samples=50, extractor=None):
    class_dir = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    existing = len([f for f in os.listdir(class_dir) if f.endswith(".npy")])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return 0

    recording = False
    collected = 0
    hindi = to_hindi(class_name)

    print(f"\nCollecting: {class_name} ({hindi})")
    print(f"Existing: {existing} samples")
    print("SPACE=record  Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        result = extractor.process_frame(frame)
        display = result["annotated"]

        status = "REC" if recording else "PAUSED"
        color = (0, 0, 255) if recording else (200, 200, 200)
        cv2.putText(display, f"{class_name} ({hindi})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, status, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f"{existing+collected}/{existing+num_samples}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if recording and result["features"] is not None:
            fname = f"sample_{existing+collected:04d}.npy"
            np.save(os.path.join(class_dir, fname), result["features"])
            collected += 1
            if collected >= num_samples:
                print(f"Done: {collected} samples for {class_name}")
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


def interactive(num_samples=50):
    extractor = LandmarkExtractor()
    print("\n" + "=" * 50)
    print("  Sign Language Data Collection")
    print("=" * 50)
    print(f"\nClasses ({len(ALL_CLASSES)}):")
    for i, c in enumerate(ALL_CLASSES):
        print(f"  {i+1:3d}. {c:>12s} {to_hindi(c)}", end="")
        if (i + 1) % 3 == 0:
            print()
    print("\n")

    while True:
        choice = input("Class name (or 'all' / 'quit'): ").strip()
        if choice.lower() == "quit":
            break
        elif choice.lower() == "all":
            for c in ALL_CLASSES:
                collect_for_class(c, num_samples, extractor)
        else:
            c = choice if choice in ALL_CLASSES else choice.upper()
            if c in ALL_CLASSES:
                collect_for_class(c, num_samples, extractor)
            else:
                print(f"Unknown: {choice}")

    extractor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    if args.class_name:
        ext = LandmarkExtractor()
        collect_for_class(args.class_name, args.samples, ext)
        ext.close()
    else:
        interactive(args.samples)
