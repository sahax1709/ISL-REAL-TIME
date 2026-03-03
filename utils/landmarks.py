"""
Hand landmark extraction using MediaPipe.
Extracts 21 landmarks (x, y, z) per hand = 63 features.
For two-hand signs, concatenates to 126 features (padded if only one hand).
"""

import numpy as np

try:
    import mediapipe as mp
    import cv2
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

SINGLE_HAND_FEATURES = 63
TWO_HAND_FEATURES = 126


class LandmarkExtractor:
    """Extract normalized hand landmarks from images/frames."""

    def __init__(self, max_hands=2, min_detection_confidence=0.7):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "mediapipe and opencv-python required. "
                "pip install mediapipe opencv-python-headless"
            )
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        self.max_hands = max_hands

    def extract_from_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return {"landmarks": None, "num_hands": 0, "hand_landmarks_raw": []}

        all_landmarks = []
        for hand_lm in results.multi_hand_landmarks:
            hand_data = []
            for lm in hand_lm.landmark:
                hand_data.extend([lm.x, lm.y, lm.z])
            all_landmarks.append(hand_data)

        while len(all_landmarks) < self.max_hands:
            all_landmarks.append([0.0] * SINGLE_HAND_FEATURES)

        features = np.array(
            all_landmarks[:self.max_hands]
        ).flatten().astype(np.float32)

        return {
            "landmarks": features,
            "num_hands": len(results.multi_hand_landmarks),
            "hand_landmarks_raw": results.multi_hand_landmarks,
        }

    def extract_from_bytes(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"landmarks": None, "num_hands": 0, "hand_landmarks_raw": []}
        return self.extract_from_frame(frame)

    def close(self):
        self.hands.close()
