# Virtual Simulation

No hospital access is needed.

- **The dataset is the "hospital archive".** Images in `data/` stand in for scans stored in a PACS system. `NORMAL` and `PNEUMONIA` folders play the role of radiologist-confirmed reports.
- **Images represent diagnoses.** Real data: labelled X-rays (Kaggle). Simulated data: clear lung fields vs. hazy opacities, generated in `src/simulate_data.py`.
- **Predictions simulate doctor assistance.** The model outputs a label and confidence for each new scan, like an AI second reader flagging cases for a radiologist. The web app shows this as an upload-and-analyze workflow.

## Workflow
1. Scans arrive (data/ or upload in the web app)
2. Preprocess for consistent contrast and size
3. Model predicts probability of pneumonia
4. System shows label + confidence
5. Metrics and plots show how trustworthy the system is
6. A human reviews flagged cases (the app displays a disclaimer)

## Outputs to capture as proof
sample images, preprocessing before/after, training log, accuracy/loss graph, confusion matrix, predicted-vs-actual grid, web UI screenshots.
