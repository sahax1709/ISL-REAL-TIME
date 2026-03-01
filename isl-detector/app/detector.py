"""
ISL Detector - MediaPipe hand tracking + hybrid sign classifier
25 ISL signs with English <-> Hindi mapping
"""
import numpy as np, cv2, logging
import mediapipe as mp
from typing import Dict, List, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

ISL_SIGNS = {
    "A":          {"hindi": "\u0905",    "desc": "Closed fist, thumb on the side"},
    "B":          {"hindi": "\u092c",    "desc": "Flat hand, 4 fingers up, thumb folded"},
    "C":          {"hindi": "\u0938",    "desc": "Curved hand like holding a cup"},
    "D":          {"hindi": "\u0921",    "desc": "Index finger pointing up, rest curled"},
    "I":          {"hindi": "\u0907",    "desc": "Only pinky finger extended"},
    "L":          {"hindi": "\u0932",    "desc": "L-shape: thumb + index finger extended"},
    "V":          {"hindi": "\u0935",    "desc": "Victory/peace sign"},
    "W":          {"hindi": "\u0935",    "desc": "Three fingers (index+middle+ring) up"},
    "Y":          {"hindi": "\u092f",    "desc": "Thumb + pinky extended (hang loose)"},
    "Hello":      {"hindi": "\u0928\u092e\u0938\u094d\u0924\u0947", "desc": "Open palm, all fingers spread wide"},
    "Thank You":  {"hindi": "\u0927\u0928\u094d\u092f\u0935\u093e\u0926", "desc": "Flat hand from chin forward"},
    "Yes":        {"hindi": "\u0939\u093e\u0901",   "desc": "Thumbs up gesture"},
    "No":         {"hindi": "\u0928\u0939\u0940\u0902",  "desc": "Index + middle finger closing"},
    "I Love You": {"hindi": "\u092e\u0948\u0902 \u0924\u0941\u092e\u0938\u0947 \u092a\u094d\u092f\u093e\u0930 \u0915\u0930\u0924\u093e/\u0915\u0930\u0924\u0940 \u0939\u0942\u0901", "desc": "Thumb + index + pinky extended"},
    "OK":         {"hindi": "\u0920\u0940\u0915 \u0939\u0948", "desc": "Thumb + index form a circle, rest extended"},
    "Stop":       {"hindi": "\u0930\u0941\u0915\u094b",  "desc": "Palm facing forward, fingers together"},
    "One":        {"hindi": "\u090f\u0915",   "desc": "Only index finger raised"},
    "Two":        {"hindi": "\u0926\u094b",   "desc": "Index + middle fingers raised"},
    "Three":      {"hindi": "\u0924\u0940\u0928",  "desc": "Index + middle + ring fingers raised"},
    "Four":       {"hindi": "\u091a\u093e\u0930",  "desc": "Four fingers raised, thumb folded"},
    "Five":       {"hindi": "\u092a\u093e\u0901\u091a",  "desc": "All five fingers spread open"},
    "Good":       {"hindi": "\u0905\u091a\u094d\u091b\u093e", "desc": "Thumbs up"},
    "Bad":        {"hindi": "\u092c\u0941\u0930\u093e",  "desc": "Thumbs down"},
    "Peace":      {"hindi": "\u0936\u093e\u0902\u0924\u093f", "desc": "Victory sign"},
    "Rock":       {"hindi": "\u0930\u0949\u0915",  "desc": "Index + pinky extended, rest curled"},
}


class ISLDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.6)
        self.buf: List[str] = []
        self.buf_sz = 5
        self.last = ""
        self.thresh = 0.60
        logger.info(f"ISL Detector ready - {len(ISL_SIGNS)} signs")

    def _fingers(self, lm) -> List[bool]:
        p = lm.landmark
        if p[2].x < p[17].x:
            thumb = p[4].x < p[3].x
        else:
            thumb = p[4].x > p[3].x
        return [thumb, p[8].y < p[6].y, p[12].y < p[10].y,
                p[16].y < p[14].y, p[20].y < p[18].y]

    def _d(self, lm, a, b):
        p = lm.landmark
        return np.sqrt((p[a].x-p[b].x)**2 + (p[a].y-p[b].y)**2)

    def _classify(self, lm) -> Tuple[str, float]:
        f = self._fingers(lm)
        thumb, idx, mid, ring, pink = f
        n = sum(f)
        palm = self._d(lm, 0, 9)
        pts = lm.landmark

        if n == 0: return "A", 0.85
        if n == 1:
            if thumb:
                return ("Yes", 0.88) if pts[4].y < pts[3].y else ("Bad", 0.82)
            if idx: return "D", 0.87
            if pink: return "I", 0.88
        if n == 2:
            if thumb and idx: return "L", 0.87
            if thumb and pink: return "Y", 0.90
            if idx and mid:
                sp = self._d(lm, 8, 12)
                return ("V", 0.88) if sp > palm*0.35 else ("Two", 0.80)
            if idx and pink: return "Rock", 0.85
        if n == 3:
            if thumb and idx and pink: return "I Love You", 0.90
            if idx and mid and ring: return "W", 0.85
            if thumb and idx and mid:
                if self._d(lm, 4, 8) < palm*0.3: return "OK", 0.88
        if n == 4:
            if not thumb:
                sp = self._d(lm, 8, 20)
                return ("B", 0.82) if sp < palm*0.6 else ("Four", 0.80)
        if n == 5:
            sp = self._d(lm, 8, 20)
            if sp > palm*0.8: return "Hello", 0.85
            elif sp > palm*0.5: return "Five", 0.82
            else: return "Stop", 0.80

        if thumb and idx and not pink:
            gap = self._d(lm, 4, 8)
            if 0.12 < gap < 0.35: return "C", 0.75

        m = {1:"One",2:"Two",3:"Three",4:"Four",5:"Five"}
        if n in m: return m[n], 0.55
        return "Unknown", 0.0

    def _smooth(self, sign, conf):
        self.buf.append(sign)
        if len(self.buf) > self.buf_sz: self.buf.pop(0)
        counts = Counter(self.buf)
        best, count = counts.most_common(1)[0]
        sc = (count / len(self.buf)) * conf
        if sc >= self.thresh:
            self.last = best
            return best, sc
        return (self.last or "Detecting..."), 0.0

    def detect(self, frame: np.ndarray) -> Dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        resp = {"detected": False, "sign": "", "hindi": "",
                "confidence": 0.0, "landmarks": None,
                "description": "", "finger_states": []}
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            sign, conf = self._classify(hand)
            sign, conf = self._smooth(sign, conf)
            info = ISL_SIGNS.get(sign, {"hindi": "?", "desc": ""})
            resp.update({
                "detected": True, "sign": sign, "hindi": info["hindi"],
                "confidence": round(conf, 2),
                "landmarks": [{"x": round(l.x,4), "y": round(l.y,4), "z": round(l.z,4)}
                              for l in hand.landmark],
                "description": info["desc"],
                "finger_states": self._fingers(hand)})
        else:
            self.buf.clear()
            self.last = ""
        return resp
