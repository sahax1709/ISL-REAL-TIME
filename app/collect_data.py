"""
Data Collection Script
Capture custom sign language training data via webcam, images, or CSV.

Usage:
    python -m app.collect_data --label hello --samples 200
    python -m app.collect_data --label thank_you --from-images ./imgs/
    python -m app.collect_data --from-csv dataset.csv
    python -m app.collect_data --list
"""

import os, sys, json, argparse, csv
import numpy as np, cv2
import mediapipe as mp
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"

def setup_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

def extract_landmarks(hands, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    lm = []
    for p in hand.landmark:
        lm.extend([p.x, p.y, p.z])
    return np.array(lm, dtype=np.float32)

def normalize_landmarks(landmarks):
    lm = landmarks.reshape(21, 3)
    lm = lm - lm[0].copy()
    mx = np.max(np.abs(lm))
    if mx > 0:
        lm = lm / mx
    return lm.flatten()

def collect_from_webcam(label, num_samples=200, camera_id=0):
    setup_dirs()
    label_dir = RAW_DIR / label
    label_dir.mkdir(exist_ok=True)
    existing = len(list(label_dir.glob("*.npy")))
    print(f"\nCollecting '{label}' | Target: {num_samples} | Existing: {existing}")
    print("Press S to start, Q to quit.\n")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("ERROR: Could not open webcam."); return

    collecting, count = False, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        hand_ok = False
        if results.multi_hand_landmarks:
            hand_ok = True
            for h in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(display, h, mp_hands.HAND_CONNECTIONS)

        status = "COLLECTING" if collecting else "READY (S)"
        clr = (0,0,255) if collecting else (0,255,0)
        cv2.putText(display, f"Label: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(display, f"Status: {status}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2)
        cv2.putText(display, f"Samples: {count}/{num_samples}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        if collecting and hand_ok:
            landmarks = extract_landmarks(hands, frame)
            if landmarks is not None:
                norm = normalize_landmarks(landmarks)
                np.save(str(label_dir / f"sample_{existing+count:05d}.npy"), norm)
                count += 1
                if count >= num_samples:
                    print(f"Done! {count} samples for '{label}'"); break

        cv2.imshow("Data Collection", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        elif key == ord("s"): collecting = True; print("Started...")

    cap.release(); cv2.destroyAllWindows(); hands.close()

def collect_from_images(label, image_dir):
    setup_dirs()
    label_dir = RAW_DIR / label
    label_dir.mkdir(exist_ok=True)
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"ERROR: {image_dir} not found."); return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [f for f in image_dir.iterdir() if f.suffix.lower() in exts]
    count, existing = 0, len(list(label_dir.glob("*.npy")))
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None: continue
        lm = extract_landmarks(hands, image)
        if lm is not None:
            np.save(str(label_dir / f"sample_{existing+count:05d}.npy"), normalize_landmarks(lm))
            count += 1
    hands.close()
    print(f"Extracted {count}/{len(images)} samples for '{label}'.")

def collect_from_csv(csv_path):
    setup_dirs()
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found."); return
    label_counts = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 64: continue
            label = row[0].strip()
            vals = np.array([float(v) for v in row[1:64]], dtype=np.float32)
            norm = normalize_landmarks(vals)
            d = RAW_DIR / label; d.mkdir(parents=True, exist_ok=True)
            idx = label_counts.get(label, 0)
            np.save(str(d / f"sample_{idx:05d}.npy"), norm)
            label_counts[label] = idx + 1
    print("Loaded from CSV:")
    for lbl, cnt in label_counts.items():
        print(f"  {lbl}: {cnt} samples")

def list_collected_data():
    if not RAW_DIR.exists():
        print("No data yet."); return
    print(f"\n{'Label':<20} {'Samples':>10}")
    print("-" * 32)
    total = 0
    for d in sorted(RAW_DIR.iterdir()):
        if d.is_dir():
            c = len(list(d.glob("*.npy")))
            print(f"{d.name:<20} {c:>10}"); total += c
    print("-" * 32)
    print(f"{'TOTAL':<20} {total:>10}\n")

def main():
    p = argparse.ArgumentParser(description="Collect sign language data")
    p.add_argument("--label", type=str)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--from-images", type=str)
    p.add_argument("--from-csv", type=str)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list: list_collected_data()
    elif args.from_csv: collect_from_csv(args.from_csv)
    elif args.label and args.from_images: collect_from_images(args.label, args.from_images)
    elif args.label: collect_from_webcam(args.label, args.samples, args.camera)
    else: p.print_help()

if __name__ == "__main__":
    main()
