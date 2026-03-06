"""
Hand landmark extraction + feature engineering.
Produces 89 features per hand, 178 total (two hands, second padded if absent).
"""

import numpy as np
import mediapipe as mp
import cv2

TIPS = [4, 8, 12, 16, 20]
MCPS = [1, 5, 9, 13, 17]
FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
FEATURES_PER_HAND = 89
TOTAL_FEATURES = FEATURES_PER_HAND * 2


def normalize_landmarks(raw):
    """Wrist-origin + scale normalization. Input (21,3), output (21,3)."""
    pts = np.array(raw, dtype=np.float32).reshape(21, 3)
    pts = pts - pts[0]
    ref = np.linalg.norm(pts[9])
    if ref < 1e-6:
        ref = 1.0
    pts = pts / ref
    return pts


def finger_curls(pts):
    """5 values: 0=straight, 1=fully curled."""
    curls = []
    for finger in FINGERS:
        bone_len = sum(np.linalg.norm(pts[finger[i+1]] - pts[finger[i]]) for i in range(len(finger)-1))
        bone_len = max(bone_len, 1e-6)
        direct = np.linalg.norm(pts[finger[-1]] - pts[finger[0]])
        curls.append(1.0 - min(direct / bone_len, 1.0))
    return np.array(curls, dtype=np.float32)


def tip_distances(pts):
    """10 pairwise distances between fingertips."""
    d = []
    for i in range(len(TIPS)):
        for j in range(i+1, len(TIPS)):
            d.append(np.linalg.norm(pts[TIPS[i]] - pts[TIPS[j]]))
    return np.array(d, dtype=np.float32)


def finger_angles(pts):
    """5 angles: each finger vs palm direction."""
    palm_center = np.mean(pts[MCPS], axis=0)
    palm_vec = palm_center - pts[0]
    norm_p = np.linalg.norm(palm_vec)
    if norm_p < 1e-6:
        return np.zeros(5, dtype=np.float32)

    angles = []
    for tip, mcp in zip(TIPS, MCPS):
        fv = pts[tip] - pts[mcp]
        nf = np.linalg.norm(fv)
        if nf < 1e-6:
            angles.append(0.0)
            continue
        cos_a = np.clip(np.dot(fv, palm_vec) / (nf * norm_p), -1.0, 1.0)
        angles.append(float(np.arccos(cos_a) / np.pi))
    return np.array(angles, dtype=np.float32)


def thumb_features(pts):
    """3 values: thumb relative to index/middle."""
    t = pts[4]
    return np.array([
        np.linalg.norm(t - (pts[5] + pts[6]) / 2),
        np.linalg.norm(t - (pts[9] + pts[10]) / 2),
        t[1] - pts[5][1],
    ], dtype=np.float32)


def hand_orientation(pts):
    """3 values: direction wrist to middle tip, normalized."""
    d = pts[12] - pts[0]
    n = np.linalg.norm(d)
    if n < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (d / n).astype(np.float32)


def extract_features(pts):
    """89 features from normalized (21,3) landmarks."""
    return np.concatenate([
        pts.flatten(),          # 63
        finger_curls(pts),      # 5
        tip_distances(pts),     # 10
        finger_angles(pts),     # 5
        thumb_features(pts),    # 3
        hand_orientation(pts),  # 3
    ])


class LandmarkExtractor:
    def __init__(self, max_hands=2, detection_conf=0.7, tracking_conf=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.max_hands = max_hands

    def process_frame(self, frame):
        """Returns dict with features(178), num_hands, annotated frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        annotated = frame.copy()

        if not results.multi_hand_landmarks:
            return {"features": None, "num_hands": 0, "annotated": annotated}

        for hlm in results.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(annotated, hlm, self.mp_hands.HAND_CONNECTIONS)

        hand_feats = []
        for hlm in results.multi_hand_landmarks:
            raw = np.array([[lm.x, lm.y, lm.z] for lm in hlm.landmark])
            normed = normalize_landmarks(raw)
            hand_feats.append(extract_features(normed))

        while len(hand_feats) < self.max_hands:
            hand_feats.append(np.zeros(FEATURES_PER_HAND, dtype=np.float32))

        combined = np.concatenate(hand_feats[:self.max_hands])
        return {"features": combined, "num_hands": len(results.multi_hand_landmarks), "annotated": annotated}

    def process_bytes(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"features": None, "num_hands": 0, "annotated": None}
        return self.process_frame(frame)

    def close(self):
        self.hands.close()
