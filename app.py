"""
app.py - Local Flask server for sign language detection.

Run:  python app.py
Open: http://localhost:5000
"""

import os
import sys
import base64
import logging

import numpy as np
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(__file__))

from model.detector import SignLanguageDetector
from utils.landmarks import LandmarkExtractor
from utils.hindi_mapping import ALL_CLASSES, to_display

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

detector = SignLanguageDetector(confidence_threshold=35.0, smoothing_window=4)
extractor = None


def init():
    global extractor
    log.info("Loading model...")
    detector.load()
    extractor = LandmarkExtractor(max_hands=2, detection_conf=0.7)
    log.info(f"Ready. Model loaded: {detector.is_loaded}, Classes: {len(ALL_CLASSES)}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json()
        image_data = body.get("image", "")
        if not image_data:
            return jsonify({"error": "No image"}), 400

        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        result = extractor.process_bytes(image_bytes)
        if result["features"] is None:
            return jsonify({"detected": False, "message": "No hand detected", "num_hands": 0})

        prediction = detector.predict(result["features"])
        prediction["num_hands"] = result["num_hands"]
        return jsonify(prediction)

    except Exception as e:
        log.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": detector.is_loaded,
        "num_classes": len(ALL_CLASSES),
    })


@app.route("/classes")
def classes():
    return jsonify({
        "total": len(ALL_CLASSES),
        "classes": [to_display(c) for c in ALL_CLASSES],
    })


@app.route("/settings", methods=["POST"])
def update_settings():
    """Live-update detection settings without restart."""
    body = request.get_json()
    if "confidence_threshold" in body:
        detector.confidence_threshold = float(body["confidence_threshold"])
    if "smoothing_window" in body:
        from model.detector import TemporalSmoother
        detector.smoother = TemporalSmoother(
            window=int(body["smoothing_window"]),
            agreement_ratio=0.6,
        )
    return jsonify({"ok": True})


if __name__ == "__main__":
    init()
    print("\n  Open http://localhost:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
