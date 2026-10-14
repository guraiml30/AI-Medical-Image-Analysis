#!/usr/bin/env python3
"""
rullall.py - RUN ALL: deploys every pipeline, then launches the themed API.

    python rullall.py                  # everything
    python rullall.py --epochs 10      # train longer
    python rullall.py --transfer       # MobileNetV2 transfer learning
    python rullall.py --skip-train     # reuse saved model
    python rullall.py --no-api         # pipelines only
    python rullall.py --api-only       # just the web app
    python rullall.py --port 8080

Stages: setup -> dataset -> preprocess -> model -> train -> evaluate
        -> predict -> report -> api
Uses the modules in src/ (config, simulate_data, data_loader, preprocessing,
model, train, evaluate, predict, visualize, api).

DISCLAIMER: educational project, NOT a medical device.
"""
import argparse
import json
import shutil
import sys
import time
import traceback

from src import config
from src.config import CLASSES, DATA_DIR, IMAGES_DIR, MODEL_PATH, OUTPUTS_DIR, ROOT


def stage_setup(ctx):
    config.ensure_dirs()
    return "folders ready"


def stage_dataset(ctx):
    from src.simulate_data import dataset_exists, generate_dataset
    if dataset_exists():
        ctx["dataset"] = "real"
        return "dataset found in data/ (using it)"
    generate_dataset()
    ctx["dataset"] = "simulated"
    return "no dataset found -> generated SIMULATED chest X-rays in data/"


def stage_preprocess(ctx):
    from src.data_loader import load_split
    from src.preprocessing import read_gray
    from src.visualize import plot_preprocessing, plot_sample_grid
    ctx["X_train"], ctx["y_train"] = load_split("train")
    ctx["X_test"], ctx["y_test"] = load_split("test")
    sample = next((DATA_DIR / "train" / "PNEUMONIA").iterdir())
    plot_preprocessing(read_gray(sample))
    plot_sample_grid(ctx["X_train"], ctx["y_train"])
    return f"train={len(ctx['y_train'])} test={len(ctx['y_test'])} images"


def stage_model(ctx):
    from src.model import build_model
    m = build_model(ctx["args"].transfer)
    ctx["model"] = m
    return f"{'MobileNetV2 transfer' if ctx['args'].transfer else 'custom CNN'} - {m.count_params():,} params"


def stage_train(ctx):
    a = ctx["args"]
    if a.skip_train and MODEL_PATH.exists():
        from src.train import load_trained
        ctx["model"] = load_trained()
        return "skipped training, loaded saved model"
    from src.train import train
    ctx["model"], h = train(ctx["X_train"], ctx["y_train"], epochs=a.epochs, transfer=a.transfer)
    return f"saved {MODEL_PATH.name}, best val_acc={max(h['val_accuracy']):.3f}"


def stage_evaluate(ctx):
    from src.evaluate import evaluate
    m, prob, pred = evaluate(ctx["model"], ctx["X_test"], ctx["y_test"], ctx.get("dataset", "unknown"))
    ctx["metrics"], ctx["prob"], ctx["pred"] = m, prob, pred
    return f"accuracy={m['accuracy']:.3f}  recall(pneumonia)={m['recall_pneumonia']:.3f}"


def stage_predict(ctx):
    from src.visualize import plot_predictions
    plot_predictions(ctx["X_test"], ctx["y_test"], ctx["prob"], ctx["pred"])
    return "saved outputs/prediction_results.png"


def stage_report(ctx):
    """Copy proof images to images/ and refresh the Results block in README.md."""
    names = [("preprocessing_preview.png", "01_preprocessing_preview.png"),
             ("accuracy_loss_graph.png", "02_accuracy_loss_graph.png"),
             ("confusion_matrix.png", "03_confusion_matrix.png"),
             ("prediction_results.png", "04_prediction_results.png"),
             ("sample_images.png", "00_sample_images.png")]
    for src, dst in names:
        if (OUTPUTS_DIR / src).exists():
            shutil.copy(OUTPUTS_DIR / src, IMAGES_DIR / dst)
    m = ctx.get("metrics") or json.loads((OUTPUTS_DIR / "metrics.json").read_text())
    block = (f"| Metric | Value |\n|---|---|\n| Accuracy | {m['accuracy']*100:.1f}% |\n"
             f"| Precision (pneumonia) | {m['precision_pneumonia']*100:.1f}% |\n"
             f"| Recall (pneumonia) | {m['recall_pneumonia']*100:.1f}% |\n"
             f"| F1 (pneumonia) | {m['f1_pneumonia']*100:.1f}% |\n"
             f"| Test images | {m['test_images']} |\n| Data | {m['dataset']} |")
    if m["dataset"] == "simulated":
        block += "\n\n> Simulated data: re-run with the real Kaggle dataset for realistic numbers."
    readme = ROOT / "README.md"
    txt = readme.read_text(encoding="utf-8")
    s, e = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
    if s in txt and e in txt:
        txt = txt[:txt.index(s) + len(s)] + "\n" + block + "\n" + txt[txt.index(e):]
        readme.write_text(txt, encoding="utf-8")
    return "images/ refreshed + README results updated"


def stage_api(ctx):
    from src.api import run_api
    a = ctx["args"]
    run_api(a.host, a.port)
    return "server stopped"


def parse_args():
    p = argparse.ArgumentParser(description="Run every pipeline + the API")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--transfer", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--no-api", action="store_true")
    p.add_argument("--api-only", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def main():
    args = parse_args()
    ctx = {"args": args}
    print("\n" + "=" * 64 + "\n  AI-Powered Medical Image Analysis System | RUN ALL\n" + "=" * 64)
    if args.api_only:
        stages = [("Setup", stage_setup), ("API", stage_api)]
    else:
        stages = [("Setup", stage_setup), ("Dataset", stage_dataset), ("Preprocess", stage_preprocess),
                  ("Model", stage_model), ("Train", stage_train), ("Evaluate", stage_evaluate),
                  ("Predict", stage_predict), ("Report", stage_report)]
        if not args.no_api:
            stages.append(("API", stage_api))

    report = []
    for i, (name, fn) in enumerate(stages, 1):
        print(f"\n[{i}/{len(stages)}] {name} ...")
        t0 = time.time()
        try:
            msg = fn(ctx)
            report.append({"stage": name, "status": "ok", "seconds": round(time.time() - t0, 1), "info": msg})
            print(f"    [OK] {msg}  ({time.time() - t0:.1f}s)")
        except KeyboardInterrupt:
            print("\n    stopped by user")
            break
        except ImportError as e:
            print(f"    [X] Missing package: {e}\n        Run: pip install -r requirements.txt")
            report.append({"stage": name, "status": "failed", "error": str(e)})
            break
        except Exception as e:
            traceback.print_exc()
            report.append({"stage": name, "status": "failed", "error": str(e)})
            break

    config.ensure_dirs()
    (OUTPUTS_DIR / "run_report.json").write_text(json.dumps(report, indent=2))
    print("\n" + "=" * 64 + "\n  SUMMARY\n" + "=" * 64)
    for r in report:
        print(f"  {'OK ' if r['status'] == 'ok' else 'ERR'}  {r['stage']:<11} {r.get('info', r.get('error', ''))}")
    print("\n  Proof files are in outputs/ and images/ - screenshot them for your README.\n")


if __name__ == "__main__":
    sys.exit(main())
