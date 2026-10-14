"""predict.py - predict one image (the 'doctor assistant' step)."""
import numpy as np

from .config import CLASSES
from .preprocessing import preprocess_image, read_gray


def predict_gray(model, gray):
    """gray: 2-D uint8 array -> dict with label, probability, confidence."""
    x = preprocess_image(gray)[None, ...]
    p = float(model.predict(x, verbose=0)[0][0])
    label = CLASSES[int(p >= 0.5)]
    conf = p if p >= 0.5 else 1 - p
    return {"label": label, "probability_pneumonia": round(p, 4),
            "confidence_percent": round(conf * 100, 1)}


def predict_file(model, path):
    gray = read_gray(path)
    if gray is None:
        raise ValueError(f"Cannot read image: {path}")
    return predict_gray(model, gray)
