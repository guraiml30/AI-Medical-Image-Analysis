# AI-Powered Medical Image Analysis System

Deep-learning pipeline that classifies chest X-rays as **NORMAL** or **PNEUMONIA**, with a clinical-themed FastAPI web app for live predictions.

> Educational project built on public / simulated data. **Not a medical device. Not for diagnosis.**

## Overview
Medical image analysis uses AI to help radiologists flag suspicious scans faster and more consistently. This project implements the complete workflow: data -> preprocessing -> model -> training -> evaluation -> visualization -> web API.

## Problem Statement
Radiologists face rising scan volumes, fatigue and reader-to-reader variability. An AI "second reader" can prioritise urgent cases, reduce missed findings and speed up reporting.

## Industry Relevance
Hospitals, diagnostic labs, radiology centers and health-tech companies use similar systems for triage, second opinions and workflow automation.

## Tech Stack
Python 3.10+, OpenCV, NumPy, Matplotlib, scikit-learn, TensorFlow/Keras (custom CNN or MobileNetV2 transfer learning), FastAPI, Uvicorn.

## Dataset
- **Virtual simulation (default):** synthetic chest X-ray-style images generated in code (clear lungs vs lungs with hazy opacities), so the project runs with zero downloads.
- **Public dataset (recommended for real results):** Kaggle *Chest X-Ray Images (Pneumonia)*. Place it as `data/train/{NORMAL,PNEUMONIA}` and `data/test/{NORMAL,PNEUMONIA}`.

## Architecture
```
Medical Image -> Grayscale -> Resize 128x128 -> CLAHE -> Normalise
   -> CNN / MobileNetV2 -> P(PNEUMONIA) -> Label + Confidence
   -> Metrics, Confusion Matrix, Predicted-vs-Actual plots -> FastAPI Web UI
```
More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Installation
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux
pip install -r requirements.txt
```

## Usage
```bash
python rullall.py                 # run EVERY pipeline, then start the web app
python rullall.py --epochs 10     # train longer
python rullall.py --transfer      # MobileNetV2 transfer learning
python rullall.py --skip-train    # reuse saved model
python rullall.py --api-only      # web app only -> http://127.0.0.1:8000
```
Individual steps: `python main.py simulate | train | evaluate | predict <img> | api`

## Results
<!-- RESULTS_START -->
| Metric | Value |
|---|---|
| Accuracy | 50.0% |
| Precision (pneumonia) | 50.0% |
| Recall (pneumonia) | 100.0% |
| F1 (pneumonia) | 66.7% |
| Test images | 150 |
| Data | real |
<!-- RESULTS_END -->

## Screenshots
| Samples | Preprocessing |
|---|---|
| ![samples](images/00_sample_images.png) | ![pre](images/01_preprocessing_preview.png) |

| Training curves | Confusion matrix |
|---|---|
| ![curves](images/02_accuracy_loss_graph.png) | ![cm](images/03_confusion_matrix.png) |

| Predicted vs actual |
|---|
| ![pred](images/04_prediction_results.png) |

Add your own after running the app: `images/06_api_home.png`, `images/07_api_prediction.png`.

## Project Structure
```
AI-Medical-Image-Analysis/
├── data/          datasets (git-ignored) - real or simulated
├── notebooks/     exploration notebook
├── src/           config, simulate_data, data_loader, preprocessing, model, train, evaluate, predict, visualize, api, ui
├── models/        saved trained model
├── outputs/       generated graphs, metrics.json, run_report.json
├── images/        screenshots used in this README
├── docs/          explanation, architecture, simulation, GitHub + proof guides
├── main.py        run individual steps
├── rullall.py     run everything + start API
├── requirements.txt
└── .gitignore
```

## Learning Outcomes
Image preprocessing with OpenCV, CNNs and transfer learning, class-imbalance handling, evaluation (accuracy, recall, confusion matrix), building and styling an ML API, and professional GitHub documentation.

## Disclaimer
Educational use only. Predictions are not medical advice. Always consult a qualified radiologist.

## License
MIT
