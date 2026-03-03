"""
collect.py - Collect training data using your webcam.

Usage:
    python collect.py                    # Interactive class selection
    python collect.py --class A          # Collect samples for letter A
    python collect.py --class hello      # Collect samples for word 'hello'
    python collect.py --samples 100      # Collect 100 samples per class

Controls:
    SPACE  = Start/stop recording samples
    N      = Next class
    Q      = Quit
"""

import os
import sys
import time
import argparse
import numpy as np

try:
    import cv2
    import mediapipe as mp
except ImportError:
    print("Install deps: pip install opencv-python mediapipe")
    sys.exit(1)

from utils.hindi_mapping import ALL_CLASSES, to_hindi
from utils.landmarks import LandmarkExtractor

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "train")


def collect_for_class(class_name, num_samples=50, extractor=None):
    os.makedirs(os.path.join(DATA_DIR, class_name), exist_ok=True)
    existing = len([f for f in os.listdir(os.path.join(DATA_DIR, class_name)) if f.endswith(".npy")])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    mp_draw = mp.solutions.drawing_utils
    mp_hands_style = mp.solutions.hands

    recording = False
    collected = 0
    hindi = to_hindi(class_name)

    print(f"
Collecting for: {class_name} ({hindi})")
    print(f"Existing samples: {existing}")
    print("Press SPACE to start/stop | Q to quit
")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        result = extractor.extract_from_frame(frame)

        # Draw landmarks
        if result["hand_landmarks_raw"]:
            for hand_lm in result["hand_landmarks_raw"]:
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands_style.HANDS_CONNECTIONS
                )

        # Status overlay
        status = "RECORDING" if recording else "PAUSED"
        color = (0, 0, 255) if recording else (200, 200, 200)
        cv2.putText(frame, f"Class: {class_name} ({hindi})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Samples: {existing + collected}/{existing + num_samples}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if recording and result["landmarks"] is not None:
            fname = f"sample_{existing + collected:04d}.npy"
            fpath = os.path.join(DATA_DIR, class_name, fname)
            np.save(fpath, result["landmarks"])
            collected += 1

            if collected >= num_samples:
                print(f"Collected {collected} samples for {class_name}")
                break

        cv2.imshow("Sign Language Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            recording = not recording
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return collected


def interactive_mode(num_samples=50):
    extractor = LandmarkExtractor(max_hands=2)
    print("
" + "=" * 50)
    print("  Sign Language Data Collection")
    print("=" * 50)
    print(f"
Available classes ({len(ALL_CLASSES)}):")

    for i, cls in enumerate(ALL_CLASSES):
        hindi = to_hindi(cls)
        print(f"  {i+1:3d}. {cls:>12s} -> {hindi}", end="")
        if (i + 1) % 3 == 0:
            print()
    print()

    while True:
        choice = input("
Enter class name (or 'all' / 'quit'): ").strip()
        if choice.lower() == "quit":
            break
        elif choice.lower() == "all":
            for cls in ALL_CLASSES:
                collect_for_class(cls, num_samples, extractor)
        elif choice in ALL_CLASSES or choice.upper() in ALL_CLASSES:
            cls = choice if choice in ALL_CLASSES else choice.upper()
            collect_for_class(cls, num_samples, extractor)
        else:
            print(f"Unknown class: {choice}")

    extractor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect sign language training data")
    parser.add_argument("--class-name", type=str, default=None, help="Specific class to collect")
    parser.add_argument("--samples", type=int, default=50, help="Samples per class")
    args = parser.parse_args()

    if args.class_name:
        ext = LandmarkExtractor(max_hands=2)
        collect_for_class(args.class_name, args.samples, ext)
        ext.close()
    else:
        interactive_mode(args.samples)
