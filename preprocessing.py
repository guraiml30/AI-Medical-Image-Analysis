"""
preprocessing.py - turn a raw medical image into model-ready input.

Steps: grayscale -> resize -> CLAHE (local contrast) -> normalise to [0,1].
The SAME function is used in training and in the API, so inputs always match.
"""
try:
    import cv2  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled by the environment setup.
    cv2 = None

import numpy as np

from .config import IMG_SIZE


def _require_cv2():
    if cv2 is None:
        raise ImportError(
            "OpenCV is required for preprocessing. Install opencv-python or opencv-python-headless."
        )


def preprocess_image(gray):
    """gray: 2-D uint8 array -> float32 array shaped (IMG_SIZE, IMG_SIZE, 1)."""
    _require_cv2()
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return (gray.astype("float32") / 255.0)[..., None]


def read_gray(path):
    """Load an image file as grayscale (None if unreadable)."""
    _require_cv2()
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
