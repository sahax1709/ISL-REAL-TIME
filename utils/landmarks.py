"""
Hand landmark extraction and feature engineering.

Key improvements over raw landmarks:
  1. Wrist-origin normalization (translation invariant)
  2. Scale normalization (distance invariant)
  3. Engineered features: inter-finger angles, tip distances, finger curl ratios
  4. These extra features make Q vs P vs G much more separable
"""

import numpy as np
import mediapipe as mp
import cv2

# MediaPipe landmark indices
WRIST = 0
THUMB = [1, 2, 3, 4]
INDEX = [5, 6, 7, 8]
MIDDLE = [9, 10, 11, 12]
RING = [13, 14, 15, 16]
PINKY = [17, 18, 19, 20]
FINGERS = [THUMB, INDEX, MIDDLE, RING, PINKY]
TIPS = [4, 8, 12, 16, 20]
MCPS = [1, 5, 9, 13, 17]  # base knuckles


def normalize_landmarks(landmarks_xyz):
    """
    Translate to wrist origin, scale by hand size.
    Input:  (21, 3) raw landmarks
    Output: (21, 3) normalized landmarks
    """
    pts = np.array(landmarks_xyz, dtype=np.float32).reshape(21, 3)

    # Translate: wrist = origin
    wrist = pts[0].copy()
    pts = pts - wrist

    # Scale: normalize by distance from wrist to middle finger MCP
    ref_dist = np.linalg.norm(pts[9])  # middle MCP
    if ref_dist < 1e-6:
        ref_dist = 1.0
    pts = pts / ref_dist

    return pts


def compute_finger_curl(pts):
    """
    Curl ratio per finger: how folded is each finger (0=straight, 1=fully curled).
    Uses ratio of tip-to-MCP distance vs total finger bone length.
    """
    curls = []
    for finger in FINGERS:
        # Total bone length (sum of segment distances)
        bone_len = 0.0
        for i in range(len(finger) - 1):
            bone_len += np.linalg.norm(pts[finger[i + 1]] - pts[finger[i]])
        bone_len = max(bone_len, 1e-6)

        # Direct tip-to-base distance
        tip_base = np.linalg.norm(pts[finger[-1]] - pts[finger[0]])

        # Curl = 1 - (direct/total). Straight finger: direct ~ total -> curl ~ 0
        curls.append(1.0 - min(tip_base / bone_len, 1.0))

    return np.array(curls, dtype=np.float32)


def compute_tip_distances(pts):
    """
    Pairwise distances between all fingertips (10 values).
    Distinguishes signs like F (thumb-index touch) vs Y (thumb-pinky spread).
    """
    dists = []
    for i in range(len(TIPS)):
        for j in range(i + 1, len(TIPS)):
            d = np.linalg.norm(pts[TIPS[i]] - pts[TIPS[j]])
            dists.append(d)
    return np.array(dists, dtype=np.float32)


def compute_finger_angles(pts):
    """
    Angle of each finger relative to palm plane (5 values).
    Key for distinguishing P/Q (pointing down) vs K/G (pointing sideways/up).
    """
    angles = []
    palm_center = np.mean(pts[MCPS], axis=0)

    for tip_idx, mcp_idx in zip(TIPS, MCPS):
        finger_vec = pts[tip_idx] - pts[mcp_idx]
        palm_vec = palm_center - pts[0]  # wrist to palm center

        norm_f = np.linalg.norm(finger_vec)
        norm_p = np.linalg.norm(palm_vec)
        if norm_f < 1e-6 or norm_p < 1e-6:
            angles.append(0.0)
            continue

        cos_angle = np.dot(finger_vec, palm_vec) / (norm_f * norm_p)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles.append(float(np.arccos(cos_angle) / np.pi))  # 0-1 range

    return np.array(angles, dtype=np.float32)


def compute_thumb_position(pts):
    """
    Thumb position relative to other fingers (3 values).
    Critical for M vs N vs S vs T distinctions.
    """
    thumb_tip = pts[4]
    index_mid = (pts[5] + pts[6]) / 2  # between index MCP and PIP
    middle_mid = (pts[9] + pts[10]) / 2

    return np.array([
        np.linalg.norm(thumb_tip - index_mid),   # thumb-to-index gap
        np.linalg.norm(thumb_tip - middle_mid),   # thumb-to-middle gap
        thumb_tip[1] - pts[5][1],                 # thumb y relative to index MCP
    ], dtype=np.float32)


def compute_hand_orientation(pts):
    """
    Hand orientation features (3 values).
    Distinguishes G/H (sideways) from K/V (upward) from P/Q (downward).
    """
    # Direction from wrist to middle fingertip
    direction = pts[12] - pts[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32)
    direction = direction / norm
    return direction.astype(np.float32)


def extract_features(pts_normalized):
    """
    Build full feature vector from normalized 21x3 landmarks.

    Components:
      - Flattened normalized landmarks: 63
      - Finger curls:                    5
      - Tip-to-tip distances:           10
      - Finger angles:                   5
      - Thumb position:                  3
      - Hand orientation:                3
    Total: 89 features per hand
    """
    flat = pts_normalized.flatten()                    # 63
    curls = compute_finger_curl(pts_normalized)        # 5
    tip_dists = compute_tip_distances(pts_normalized)  # 10
    angles = compute_finger_angles(pts_normalized)     # 5
    thumb_pos = compute_thumb_position(pts_normalized) # 3
    orient = compute_hand_orientation(pts_normalized)  # 3

    return np.concatenate([flat, curls, tip_dists, angles, thumb_pos, orient])


FEATURES_PER_HAND = 89
TOTAL_FEATURES = FEATURES_PER_HAND * 2  # two hands, padded


class LandmarkExtractor:
    """Extract and process hand landmarks from camera frames."""

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
        """
        Extract engineered features from a BGR frame.
        Returns dict with:
            features:  np.ndarray(178,) or None
            num_hands: int
            annotated_frame: frame with landmarks drawn
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        annotated = frame.copy()

        if not results.multi_hand_landmarks:
            return {"features": None, "num_hands": 0, "annotated": annotated}

        # Draw landmarks on annotated frame
        for hlm in results.multi_hand_landmarks:
            self.mp_draw.draw_landmarks(
                annotated, hlm, self.mp_hands.HAND_CONNECTIONS
            )

        # Extract features per hand
        hand_features = []
        for hlm in results.multi_hand_landmarks:
            raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hlm.landmark])
            normed = normalize_landmarks(raw_pts)
            feats = extract_features(normed)
            hand_features.append(feats)

        # Pad to max_hands
        while len(hand_features) < self.max_hands:
            hand_features.append(np.zeros(FEATURES_PER_HAND, dtype=np.float32))

        combined = np.concatenate(hand_features[:self.max_hands])

        return {
            "features": combined,
            "num_hands": len(results.multi_hand_landmarks),
            "annotated": annotated,
        }

    def process_bytes(self, image_bytes):
        """Extract features from raw image bytes (for web endpoint)."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"features": None, "num_hands": 0, "annotated": None}
        return self.process_frame(frame)

    def close(self):
        self.hands.close()
