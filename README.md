# Sign Language Detector (Local)

A-Z, 0-9, 30 common words. English + Hindi output.

## Quick Start (Windows)

```
cd isl
python -m venv venv
.\venv\Scripts\Activate
python setup.py
python app.py
```

Open **http://localhost:5000**

## Quick Start (Mac/Linux)

```
cd isl
python3 -m venv venv
source venv/bin/activate
python setup.py
python app.py
```

## Improve Accuracy

```
python collect.py       # record signs via webcam (SPACE=record, Q=quit)
python train.py         # retrain on your data
python train.py --augment  # retrain with data augmentation
```

## Stack

- **MediaPipe** - hand landmark extraction (21 points per hand)
- **scikit-learn** - RandomForest classifier (no GPU needed)
- **Flask** - local web server
- **6 pip packages**, no TensorFlow, no CUDA

## Settings (adjustable live in UI)

| Setting | Default | What it does |
|---------|---------|-------------|
| Confidence | 30% | Below = ignored |
| Smoothing | 3 frames | Must agree before accepting |
| Interval | 500ms | Detection frequency |
