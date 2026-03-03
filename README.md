# Sign Language Recognition Model

Real-time gesture recognition: **MediaPipe** + **TensorFlow/Keras** + **FastAPI**, deployable on **Render**.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Collect Custom Data (3 methods)

**Webcam:**
```bash
python -m app.collect_data --label hello --samples 200
python -m app.collect_data --label thank_you --samples 200
```
Press S to start, Q to quit.

**From images:**
```bash
python -m app.collect_data --label hello --from-images ./images/hello/
```

**From CSV** (format: label, x0, y0, z0, ... x20, y20, z20):
```bash
python -m app.collect_data --from-csv dataset.csv
```

**List data:** `python -m app.collect_data --list`

## Train

```bash
python -m app.train --epochs 50 --batch-size 32 --lr 0.001
```

Saves `models/sign_model.keras` and `models/labels.json`.

## Run Locally

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 for the webcam demo.

## API Endpoints

- `GET /` — Demo UI
- `GET /health` — Health check
- `POST /predict` — Upload image file
- `POST /predict/base64` — Base64 image JSON
- `POST /predict/landmarks` — 63 landmark values

## Deploy to Render

1. Commit trained model files (`models/sign_model.keras`, `models/labels.json`)
2. Push to GitHub
3. Render Dashboard > New Web Service > Docker > Connect repo > Deploy
4. Use Starter plan or higher (TF needs ~1GB RAM)
5. Health check auto-configured via `render.yaml`

## Adding Gestures

1. `python -m app.collect_data --label new_sign --samples 200`
2. `python -m app.train`
3. Push and redeploy
