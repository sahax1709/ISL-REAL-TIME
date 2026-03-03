# Sign Language Detector

Real-time sign language detection with **bilingual output** (English + Hindi/Devanagari).

Built with **MediaPipe** (hand landmarks), **TensorFlow** (classifier), and **FastAPI** (web server).

---

## Supported Signs

| Category | Count | Examples |
|----------|-------|---------|
| Letters | 26 | A-Z (with Devanagari: A→अ, B→ब, ...) |
| Digits | 10 | 0-9 (with Hindi numerals: 0→०, 1→१, ...) |
| Words | 30 | hello→नमस्ते, thank_you→धन्यवाद, water→पानी, ... |

**Total: 66 classes**

---

## Project Structure

```
sign-language-detector/
├── app.py                  # FastAPI web server
├── pretrain.py             # Generate baseline model with synthetic data
├── train.py                # Train model on real collected data
├── collect.py              # Webcam data collection tool
├── model/
│   ├── detector.py         # TF/Keras model: build, predict, save/load
│   └── __init__.py
├── utils/
│   ├── hindi_mapping.py    # English↔Hindi mappings for all 66 classes
│   ├── landmarks.py        # MediaPipe hand landmark extraction
│   └── __init__.py
├── templates/
│   └── index.html          # Web UI with webcam + bilingual output
├── saved_model/            # (generated) Trained model files
├── data/
│   └── train/              # (generated) Training data per class
├── requirements.txt
├── Dockerfile
├── render.yaml             # Render deployment config
└── .gitignore
```

---

## Quick Start (Local)

### 1. Clone and Install

```bash
git clone <your-repo-url>
cd sign-language-detector
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Baseline Model

```bash
python pretrain.py
```

This creates a model trained on synthetic data so the app works immediately.

### 3. Run the App

```bash
python app.py
```

Open **http://localhost:8000** in your browser.

### 4. (Optional) Collect Real Data and Retrain

```bash
# Collect training samples via webcam
python collect.py

# Train on collected data
python train.py --epochs 100 --batch 32
```

---

## Collecting Custom Data

The `collect.py` script opens your webcam and lets you record hand sign samples:

```bash
# Interactive mode - choose classes one by one
python collect.py

# Collect for a specific class
python collect.py --class-name A --samples 100
python collect.py --class-name hello --samples 100
```

**Controls during collection:**
- **SPACE** = Start/stop recording
- **Q** = Quit current class

Each sample is saved as a `.npy` file (126-dim landmark array) in `data/train/<class>/`.

**Recommended: 50-200 samples per class** with varied hand positions and angles.

---

## Training

After collecting data:

```bash
python train.py --epochs 100 --batch 32 --val-split 0.2
```

The trained model is saved to `saved_model/sign_model.keras`.

---

## Deploy on Render

### Option A: Docker (Recommended)

1. Push your code to GitHub/GitLab
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repository
4. Render auto-detects the `Dockerfile`
5. The Dockerfile runs `pretrain.py` during build, so the model is ready
6. Set plan to **Starter** or higher (needs ~1GB RAM for TensorFlow)

### Option B: Native Python

1. New Web Service → Connect repo
2. Build Command: `pip install -r requirements.txt && python pretrain.py`
3. Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Set Environment: Python 3.11

### Environment Variables (both options)

```
PORT=10000
TF_CPP_MIN_LOG_LEVEL=3
PYTHONUNBUFFERED=1
```

### Important Notes for Render

- **Use Starter plan or above** — Free tier may not have enough RAM for TensorFlow + MediaPipe
- The pre-trained model is built during Docker build, so first deploy takes ~5-10 minutes
- Webcam access requires HTTPS (Render provides this automatically)
- To use custom-trained model: include `saved_model/` in your repo after training locally

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI with webcam |
| POST | `/predict` | Send base64 image, get prediction |
| GET | `/health` | Health check |
| GET | `/classes` | List all supported classes |

### POST /predict

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Response:**
```json
{
  "detected": true,
  "num_hands": 1,
  "prediction": {
    "english": "A",
    "hindi": "अ",
    "category": "letter",
    "confidence": 87.3,
    "top_3": [
      {"english": "A", "hindi": "अ", "confidence": 87.3},
      {"english": "S", "hindi": "स", "confidence": 5.1},
      {"english": "E", "hindi": "इ", "confidence": 2.8}
    ]
  }
}
```

---

## Tech Stack

- **MediaPipe Hands** — 21 hand landmarks (x, y, z) per hand
- **TensorFlow/Keras** — Dense neural network classifier
- **FastAPI** — Async web framework with auto-docs at `/docs`
- **OpenCV** — Image processing
- **Jinja2** — HTML templates

---

## Improving Accuracy

1. **Collect more data**: 200+ samples per class with varied lighting and angles
2. **Data augmentation**: Add noise, slight rotations to landmark data
3. **Use both hands**: The model supports 2-hand input (126 features)
4. **Increase model complexity**: Edit `model/detector.py` to add more layers
5. **Transfer learning**: Use a pre-trained hand gesture model as base

---

## License

MIT
