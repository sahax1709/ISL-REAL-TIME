# Sign Language Detector (Local)

Real-time hand sign recognition: **A-Z, 0-9, 30 common words** with English + Hindi output.

## Quick Start

```bash
cd isl
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
python setup.py                 # installs deps + trains baseline model
python app.py                   # open http://localhost:5000
```

That's it. Three commands.

## What's Different (vs previous version)

- **No TensorFlow** - uses scikit-learn RandomForest (installs in seconds, zero GPU/CUDA issues)
- **No Docker/Render** - local only, Flask on port 5000
- **Better recognition** - normalized landmarks, engineered features (finger curls, angles, tip distances), temporal smoothing
- **False positive fix** - smoothing window requires 4 consecutive matching predictions before accepting; low-confidence guesses are filtered out
- **6 dependencies** instead of 9, no version conflicts

## Improving Accuracy

```bash
python collect.py          # webcam data collection (SPACE=record, Q=quit)
python train.py            # retrain on your data
python train.py --augment  # retrain with 3x data augmentation
```

## Project Structure

```
isl/
  app.py             Flask server (localhost:5000)
  setup.py           One-command install + pretrain
  pretrain.py        Synthetic data baseline model
  collect.py         Webcam data collection
  train.py           Retrain on real data
  model/
    detector.py      RandomForest classifier + temporal smoother
  utils/
    landmarks.py     MediaPipe extraction + feature engineering
    hindi_mapping.py English/Hindi mappings (66 classes)
  templates/
    index.html       Web UI
  requirements.txt   6 packages, no conflicts
```

## Settings (adjustable live in the UI)

| Setting | Default | Effect |
|---------|---------|--------|
| Confidence threshold | 35% | Below this = ignored |
| Smoothing window | 4 | Frames that must agree before accepting |
| Detection interval | 500ms | How often frames are sent to server |
