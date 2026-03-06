"""
app.py - Local Flask server.
Run:  python app.py
Open: http://localhost:5000
"""

import os
import sys
import base64
import logging

import numpy as np
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.detector import SignLanguageDetector, TemporalSmoother
from utils.landmarks import LandmarkExtractor
from utils.hindi_mapping import ALL_CLASSES, to_display

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
detector = SignLanguageDetector(confidence_threshold=30.0, smoothing_window=3)
extractor = None


def init():
    global extractor
    log.info("Loading model...")
    detector.load()
    extractor = LandmarkExtractor(max_hands=2, detection_conf=0.6)
    log.info(f"Ready. Model={detector.is_loaded}, Classes={len(ALL_CLASSES)}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(silent=True)
        if not body or "image" not in body:
            return jsonify({"detected": False, "message": "No image", "raw_top_3": []})

        img = body["image"]
        if "," in img:
            img = img.split(",", 1)[1]
        raw_bytes = base64.b64decode(img)

        result = extractor.process_bytes(raw_bytes)
        if result["features"] is None:
            return jsonify({"detected": False, "message": "No hand", "num_hands": 0, "raw_top_3": []})

        pred = detector.predict(result["features"])
        pred["num_hands"] = result["num_hands"]
        return jsonify(pred)

    except Exception as e:
        log.error(f"Error: {e}", exc_info=True)
        return jsonify({"detected": False, "error": str(e), "raw_top_3": []}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": detector.is_loaded, "num_classes": len(ALL_CLASSES)})


@app.route("/classes")
def classes():
    return jsonify({"total": len(ALL_CLASSES), "classes": [to_display(c) for c in ALL_CLASSES]})


@app.route("/settings", methods=["POST"])
def update_settings():
    body = request.get_json(silent=True) or {}
    if "confidence_threshold" in body:
        detector.confidence_threshold = float(body["confidence_threshold"])
    if "smoothing_window" in body:
        detector.smoother = TemporalSmoother(
            window=int(body["smoothing_window"]),
            agreement_ratio=0.5,
        )
    return jsonify({"ok": True})


if __name__ == "__main__":
    init()
    print("\n  Open http://localhost:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
