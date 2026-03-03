"""
app.py - FastAPI server for Sign Language Detection.

Endpoints:
    GET  /             Web UI with webcam
    POST /predict      Send base64 frame, get prediction
    GET  /health       Health check
    GET  /classes      List all supported classes

Run locally:
    python app.py

Deploy on Render:
    See render.yaml
"""

import os
import io
import base64
import logging

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from model.detector import SignLanguageDetector
from utils.landmarks import LandmarkExtractor
from utils.hindi_mapping import ALL_CLASSES, to_display

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sign Language Detector", version="1.0.0")

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global instances
detector = SignLanguageDetector()
extractor = None


@app.on_event("startup")
async def startup():
    global extractor
    logger.info("Loading model...")
    detector.load()

    try:
        extractor = LandmarkExtractor(max_hands=2)
        logger.info("MediaPipe landmark extractor ready")
    except Exception as e:
        logger.warning(f"MediaPipe not available: {e}")
        extractor = None

    logger.info(f"Model loaded: {detector.is_loaded}")
    logger.info(f"Classes supported: {len(ALL_CLASSES)}")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()
        image_data = body.get("image", "")

        if not image_data:
            return JSONResponse({"error": "No image data"}, status_code=400)

        # Decode base64 image
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)

        if extractor is None:
            return JSONResponse(
                {"error": "MediaPipe not available on server"},
                status_code=503
            )

        # Extract landmarks
        result = extractor.extract_from_bytes(image_bytes)

        if result["landmarks"] is None:
            return JSONResponse({
                "detected": False,
                "message": "No hand detected",
                "num_hands": 0,
            })

        # Predict
        prediction = detector.predict(result["landmarks"])

        return JSONResponse({
            "detected": True,
            "num_hands": result["num_hands"],
            "prediction": prediction,
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": detector.is_loaded,
        "mediapipe_available": extractor is not None,
        "num_classes": len(ALL_CLASSES),
    }


@app.get("/classes")
async def classes():
    return {
        "total": len(ALL_CLASSES),
        "classes": [to_display(c) for c in ALL_CLASSES],
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
