"""
simulate_data.py - VIRTUAL SIMULATION of a hospital image archive.

No hospital access? We draw synthetic chest X-ray-style images:
  NORMAL     -> clear, dark lung fields
  PNEUMONIA  -> lung fields with hazy bright opacities (like consolidation)
This lets the whole pipeline run end-to-end. For realistic numbers, replace
with the public Kaggle "Chest X-Ray Images (Pneumonia)" dataset.
"""
try:
    import cv2  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "OpenCV is required for synthetic X-ray generation. "
        "Install it with 'pip install opencv-python' or 'pip install opencv-python-headless'."
    ) from exc

import numpy as np

from .config import CLASSES, DATA_DIR, SEED


def simulate_xray(label, rng):
    """Return one 256x256 uint8 fake chest X-ray for the given class name."""
    s = 256
    img = np.full((s, s), 40, np.uint8)
    cv2.ellipse(img, (s // 2, s // 2), (95, 120), 0, 0, 360, 90, -1)       # torso
    for cx in (88, 168):                                                   # lungs
        cv2.ellipse(img, (cx, 128), (42, 82), 0, 0, 360, 25, -1)
    for y in range(40, 220, 22):                                           # ribs
        cv2.ellipse(img, (s // 2, y), (92, 12), 0, 0, 180, 120, 2)
    cv2.line(img, (s // 2, 20), (s // 2, 240), 150, 6)                     # spine
    if label == "PNEUMONIA":
        for _ in range(int(rng.integers(2, 5))):                           # opacities
            cx = int(rng.choice([88, 168])) + int(rng.integers(-15, 15))
            cy = int(rng.integers(80, 190))
            blob = np.zeros_like(img)
            cv2.circle(blob, (cx, cy), int(rng.integers(14, 30)), 255, -1)
            blob = cv2.GaussianBlur(blob, (31, 31), 0)
            img = cv2.add(img, (blob * 0.55).astype(np.uint8))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    noise = rng.normal(0, 8, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def dataset_exists():
    """True if data/train and data/test both contain images for every class."""
    for split in ("train", "test"):
        for c in CLASSES:
            folder = DATA_DIR / split / c
            if not folder.exists() or not any(folder.iterdir()):
                return False
    return True


def generate_dataset(n_per_class=300):
    """Write simulated images to data/train and data/test."""
    rng = np.random.default_rng(SEED)
    for split, n in (("train", n_per_class), ("test", max(n_per_class // 4, 8))):
        for c in CLASSES:
            out = DATA_DIR / split / c
            out.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                cv2.imwrite(str(out / f"{c.lower()}_{i:04d}.png"), simulate_xray(c, rng))
    return DATA_DIR
