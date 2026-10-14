# Architecture

```
+-------------------+   +----------------------+   +-------------------+
| Input image       |-->| Preprocessing        |-->| CNN / MobileNetV2 |
| (X-ray PNG/JPG)   |   | gray, resize, CLAHE, |   | feature extraction|
+-------------------+   | normalise            |   | + classifier head |
                        +----------------------+   +---------+---------+
                                                             |
                      +---------------------+      +---------v---------+
                      | Visualization/API   |<-----| Prediction        |
                      | graphs, web UI      |      | label + confidence|
                      +---------------------+      +-------------------+
```

## Modules
| File | Role |
|---|---|
| `src/config.py` | paths and constants |
| `src/simulate_data.py` | virtual hospital archive generator |
| `src/data_loader.py` | read images into arrays |
| `src/preprocessing.py` | shared preprocessing (training + API) |
| `src/model.py` | CNN / transfer-learning model |
| `src/train.py` | training, saving, history plot |
| `src/evaluate.py` | metrics + confusion matrix |
| `src/predict.py` | single-image inference |
| `src/visualize.py` | all plots |
| `src/api.py`, `src/ui.py` | FastAPI app + themed UI |
| `rullall.py` | orchestrates every stage |

## Data flow
raw image -> `preprocess_image` -> tensor (128,128,1) -> model -> P(PNEUMONIA) -> threshold 0.5 -> label -> metrics/plots/API response.
