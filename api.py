"""api.py - FastAPI app: themed web UI + /predict, /metrics, /health."""
import json

import cv2  # type: ignore[import-not-found]
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import MODEL_PATH, OUTPUTS_DIR
from .predict import predict_gray
from .ui import UI_HTML


def create_app():
    app = FastAPI(title="MedVision AI - Medical Image Analysis API",
                  description="Educational chest X-ray screening demo. Not a medical device.",
                  version="1.0.0")
    OUTPUTS_DIR.mkdir(exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
    state = {"model": None}

    def get_model():
        if state["model"] is None:
            if not MODEL_PATH.exists():
                raise HTTPException(503, "Model not trained yet. Run: python rullall.py")
            from .train import load_trained
            state["model"] = load_trained()
        return state["model"]

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home():
        return UI_HTML

    @app.get("/health")
    def health():
        return {"status": "ok", "model_ready": MODEL_PATH.exists()}

    @app.get("/metrics")
    def metrics():
        p = OUTPUTS_DIR / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else {}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        data = np.frombuffer(await file.read(), np.uint8)
        gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise HTTPException(400, "Could not read image. Upload a PNG/JPG.")
        result = predict_gray(get_model(), gray)
        result["disclaimer"] = "Educational demo. Not a medical diagnosis."
        return result

    return app


def run_api(host="127.0.0.1", port=8000):
    import uvicorn
    print(f"\n   UI    http://{host}:{port}\n   Docs  http://{host}:{port}/docs\n   Ctrl+C to stop.\n")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
