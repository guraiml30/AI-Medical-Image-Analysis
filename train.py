"""train.py - train the model, save it, and save the accuracy/loss graph."""
import json

import numpy as np

from .config import BATCH_SIZE, EPOCHS, MODEL_PATH, OUTPUTS_DIR, VAL_SPLIT
from .model import build_model
from .visualize import plot_history


def train(X_train, y_train, epochs=EPOCHS, transfer=False):
    try:
        from tensorflow.keras import callbacks  # pyright: ignore[reportMissingModuleSource]
    except ImportError as exc:
        raise ImportError("TensorFlow is required to train the model.") from exc

    model = build_model(transfer)
    weights = len(y_train) / (2.0 * np.bincount(y_train))     # handle class imbalance
    hist = model.fit(
        X_train, y_train, validation_split=VAL_SPLIT, epochs=epochs, batch_size=BATCH_SIZE,
        class_weight={0: float(weights[0]), 1: float(weights[1])},
        callbacks=[callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
        verbose=2)
    model.save(MODEL_PATH)
    h = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    (OUTPUTS_DIR / "training_history.json").write_text(json.dumps(h, indent=2))
    plot_history(h)
    return model, h


def load_trained():
    try:
        from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingModuleSource]
    except ImportError as exc:
        raise ImportError("TensorFlow is required to load the model.") from exc
    return load_model(MODEL_PATH)
