import json
import logging

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from transformers import AutoTokenizer

from config import settings
from logging_config import setup_logging
from src.explainer.llm_explainer import explain as llm_explain

setup_logging()
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.app_name, description="Detects manipulation techniques in text", version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    logger.error(
        f"Unhandled error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


try:
    tokenizer = AutoTokenizer.from_pretrained(settings.onnx_dir)
    session = ort.InferenceSession(
        f"{settings.onnx_dir}/model_quantized.onnx", providers=["CPUExecutionProvider"]
    )
    with open(settings.classes_path) as f:
        classes = json.load(f)
    logger.info("Model loaded, service is ready")

except Exception:
    tokenizer = None
    session = None
    classes = None
    logger.error("Failed to load model: {exc}")


class TextInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    label: str
    confidence: float
    all_scores: dict


class AnalysisOutput(PredictionOutput):
    explanation: str


def classify(text: str) -> tuple[str, float, dict]:
    enc = tokenizer(
        text, max_length=128, padding="max_length", truncation=True, return_tensors="np"
    )

    inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }

    logits = session.run(None, inputs)[0][0]
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    idx = int(probs.argmax())

    all_scores = {classes[i]: round(float(probs[i]), 4)
                  for i in range(len(classes))}
    return classes[idx], round(float(probs[idx]), 4), all_scores


@app.get("/")
def root():
    return FileResponse(f"{settings.static_dir}/index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/ready")
def ready():
    return {"status": "ready"}


@app.post("/predict")
@limiter.limit("90/minute")
def predict(request: Request, input: TextInput):
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    label, confidence, all_scores = classify(input.text)
    logger.info(f"Prediction: {label} ({confidence})")
    return PredictionOutput(label=label, confidence=confidence, all_scores=all_scores)


@app.post("/analyze", response_model=AnalysisOutput)
@limiter.limit("30/minute")
def analyze(request: Request, input: TextInput):
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    label, confidence, all_scores = classify(input.text)
    explanation = llm_explain(input.text, label, confidence)
    return AnalysisOutput(
        label=label, confidence=confidence, all_scores=all_scores, explanation=explanation
    )
