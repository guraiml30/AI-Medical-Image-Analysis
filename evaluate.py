"""evaluate.py - accuracy, precision/recall/F1, confusion matrix on the test set."""
import json

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .config import CLASSES, OUTPUTS_DIR
from .visualize import plot_confusion


def evaluate(model, X_test, y_test, dataset_kind="unknown"):
    prob = model.predict(X_test, verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    rep = classification_report(y_test, pred, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision_pneumonia": round(rep["PNEUMONIA"]["precision"], 4),
        "recall_pneumonia": round(rep["PNEUMONIA"]["recall"], 4),
        "f1_pneumonia": round(rep["PNEUMONIA"]["f1-score"], 4),
        "test_images": int(len(y_test)),
        "dataset": dataset_kind,
        "confusion_matrix": cm.tolist(),
    }
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(classification_report(y_test, pred, target_names=CLASSES, zero_division=0))
    plot_confusion(cm)
    return metrics, prob, pred
