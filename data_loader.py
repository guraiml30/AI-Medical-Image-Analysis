"""data_loader.py - load images from data/<split>/<CLASS>/ into NumPy arrays."""
import numpy as np

from .config import CLASSES, DATA_DIR, VALID_EXT
from .preprocessing import preprocess_image, read_gray


def load_split(split):
    """Return X (N,128,128,1) float32 and y (N,) int32 for 'train' or 'test'."""
    X, y = [], []
    for label, cls in enumerate(CLASSES):
        for p in sorted((DATA_DIR / split / cls).iterdir()):
            if p.suffix.lower() in VALID_EXT:
                g = read_gray(p)
                if g is not None:
                    X.append(preprocess_image(g))
                    y.append(label)
    return np.array(X, "float32"), np.array(y, "int32")
