# Project Explanation

## What is AI-Powered Medical Image Analysis?
**Simple:** a computer program learns from many labelled medical pictures (X-rays, MRI, CT) and then helps doctors spot problems in new pictures, like a very fast second pair of eyes.

**Technical:** supervised deep learning (CNNs / transfer learning) maps an image tensor to a diagnosis probability. The pipeline is: data collection -> preprocessing -> feature extraction (learned convolutional filters) -> training -> prediction -> evaluation -> visualization.

## Problems it solves
- Huge scan volumes and radiologist shortage
- Fatigue-related misses and reader variability
- Slow turnaround for urgent cases (triage)
- Lack of quick second opinions in small clinics

## Why it matters
| Setting | Value |
|---|---|
| Hospitals | prioritise critical scans, shorten wait times |
| Diagnostic labs | consistent screening, higher throughput |
| Radiology centers | second-reader checks, fewer missed findings |
| Health-tech companies | scalable products, telehealth, remote screening |

## What AI can help detect
Pneumonia, tumors (brain MRI), fractures, lung nodules, diabetic retinopathy and more. This project focuses on **pneumonia from chest X-rays**.

## Workflow in this project
1. **Data collection** - public Kaggle dataset or virtual simulation (`src/simulate_data.py`)
2. **Preprocessing** - grayscale, resize, CLAHE, normalise (`src/preprocessing.py`)
3. **Feature extraction** - convolutional layers learn edges, textures, opacities (`src/model.py`)
4. **Training** - binary classifier with augmentation and class weights (`src/train.py`)
5. **Prediction** - probability + label + confidence (`src/predict.py`)
6. **Evaluation** - accuracy, precision, recall, F1, confusion matrix (`src/evaluate.py`)
7. **Visualization** - graphs and predicted-vs-actual grid (`src/visualize.py`)
8. **Deployment demo** - FastAPI web app (`src/api.py`)

## Tech stack options
| | A: Easiest | B: Intermediate (**selected**) | C: Advanced |
|---|---|---|---|
| Tools | NumPy, OpenCV, scikit-learn | + TensorFlow/Keras, FastAPI | + PyTorch, Grad-CAM, DICOM, MLflow |
| Model | HOG/pixels + SVM | Small CNN or MobileNetV2 transfer | ResNet/EfficientNet, U-Net segmentation |
| Data | Simulated / small subset | Kaggle Chest X-ray | Multi-class + DICOM archives |
| Difficulty | Low | Medium | High |
| GPU | No | Optional (CPU fine at 128px) | Recommended |
| Output | Accuracy score | Metrics, graphs, web app | Explainability, segmentation, tracking |

**Why B:** runs on a laptop CPU, shows real deep learning, and ends with a visible web demo that looks strong on GitHub.

## Phases (checklist)
1. Setup - venv + `pip install -r requirements.txt` (verify: `python -c "import cv2, tensorflow"`)
2. Dataset - simulate or download (verify: image counts printed)
3. Preprocessing - check `outputs/preprocessing_preview.png`
4. Model - check parameter count in log
5. Training - watch val accuracy rise; common mistake: no class weights on imbalanced data
6. Evaluation - check recall for PNEUMONIA (missing disease is worse than a false alarm)
7. Prediction - run `python main.py predict <image>`
8. Visualization - review `outputs/*.png`
9. GitHub - follow `docs/GITHUB_STEPS.md`
10. Final output - README with screenshots + demo recording
