"""config.py - one place for every path and constant used by the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
IMAGES_DIR = ROOT / "images"
DOCS_DIR = ROOT / "docs"
MODEL_PATH = MODELS_DIR / "medical_cnn.keras"

IMG_SIZE = 128                       # every image is resized to 128x128
CLASSES = ["NORMAL", "PNEUMONIA"]    # label 0 / label 1
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

SEED = 42
BATCH_SIZE = 32
EPOCHS = 8
VAL_SPLIT = 0.15


def ensure_dirs():
    """Create all project folders if they do not exist."""
    for d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR, DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)
