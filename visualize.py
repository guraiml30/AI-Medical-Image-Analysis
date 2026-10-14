"""visualize.py - every graph/image used as GitHub proof."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CLASSES, OUTPUTS_DIR, SEED
from .preprocessing import preprocess_image


def plot_preprocessing(raw_gray, name="preprocessing_preview.png"):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(raw_gray, cmap="gray"); ax[0].set_title("Original")
    ax[1].imshow(preprocess_image(raw_gray)[..., 0], cmap="gray")
    ax[1].set_title("Preprocessed (resize + CLAHE)")
    for a in ax:
        a.axis("off")
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / name, dpi=130); plt.close()


def plot_sample_grid(X, y, name="sample_images.png", n=8):
    idx = np.random.default_rng(SEED).choice(len(y), size=min(n, len(y)), replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.2))
    for a, i in zip(axes.ravel(), idx):
        a.imshow(X[i, ..., 0], cmap="gray"); a.set_title(CLASSES[y[i]], fontsize=10); a.axis("off")
    plt.suptitle("Sample images from the (virtual) archive")
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / name, dpi=130); plt.close()


def plot_history(h, name="accuracy_loss_graph.png"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h["accuracy"], label="train"); ax[0].plot(h["val_accuracy"], label="val"); ax[0].set_title("Accuracy")
    ax[1].plot(h["loss"], label="train"); ax[1].plot(h["val_loss"], label="val"); ax[1].set_title("Loss")
    for a in ax:
        a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / name, dpi=130); plt.close()


def plot_confusion(cm, name="confusion_matrix.png"):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="GnBu")
    ax.set_xticks([0, 1], CLASSES); ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                    color="white" if cm[i, j] > cm.max() / 2 else "#0B2545")
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / name, dpi=130); plt.close()


def plot_predictions(X, y, prob, pred, name="prediction_results.png"):
    idx = np.random.default_rng(SEED).choice(len(y), size=min(12, len(y)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
    for a, i in zip(axes.ravel(), idx):
        a.imshow(X[i, ..., 0], cmap="gray"); a.axis("off")
        ok = pred[i] == y[i]
        a.set_title(f"Actual: {CLASSES[y[i]]}\nPred: {CLASSES[pred[i]]} ({prob[i]*100:.0f}%)",
                    fontsize=9, color="#1B9E6B" if ok else "#D64545")
    plt.suptitle("Predicted vs Actual  (green = correct, red = wrong)", fontsize=13)
    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / name, dpi=130); plt.close()
