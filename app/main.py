"""
Sign Language Recognition API
FastAPI server for real-time sign language prediction — deploy on Render.
"""

import os
import json
import base64
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.predict import SignLanguagePredictor

# ── App Setup ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Sign Language Recognition API",
    description="Real-time sign language detection using MediaPipe + TensorFlow",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Global Predictor ──────────────────────────────────────────────────

predictor: SignLanguagePredictor = None


@app.on_event("startup")
async def load_model():
    global predictor
    model_dir = Path(__file__).resolve().parent.parent / "models"
    predictor = SignLanguagePredictor(model_dir=str(model_dir))
    print("✅ Model loaded and ready.")


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(
        content="<h1>Sign Language API</h1><p>POST image to /predict</p>"
    )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None and predictor.is_ready(),
        "labels": predictor.get_labels() if predictor and predictor.is_ready() else [],
    }


@app.post("/predict")
async def predict_sign(file: UploadFile = File(...)):
    """Predict gesture from uploaded image (JPEG/PNG/WebP)."""
    if predictor is None or not predictor.is_ready():
        raise HTTPException(503, "Model not loaded. Train the model first.")
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(400, "Only JPEG/PNG/WebP accepted.")
    try:
        contents = await file.read()
        return JSONResponse(content=predictor.predict_from_bytes(contents))
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


@app.post("/predict/landmarks")
async def predict_from_landmarks(request: Request):
    """Predict from pre-extracted MediaPipe landmarks (63 values)."""
    if predictor is None or not predictor.is_ready():
        raise HTTPException(503, "Model not loaded.")
    try:
        body = await request.json()
        landmarks = body.get("landmarks", [])
        if len(landmarks) != 63:
            raise HTTPException(400, f"Need 63 values (21×3), got {len(landmarks)}.")
        return JSONResponse(content=predictor.predict_from_landmarks(landmarks))
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON.")


@app.post("/predict/base64")
async def predict_from_base64(request: Request):
    """Predict from base64-encoded image (webcam frames)."""
    if predictor is None or not predictor.is_ready():
        raise HTTPException(503, "Model not loaded.")
    try:
        body = await request.json()
        img_b64 = body.get("image", "")
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(img_b64)
        return JSONResponse(content=predictor.predict_from_bytes(image_bytes))
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
