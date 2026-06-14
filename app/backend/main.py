"""FastAPI backend for the movie review sentiment analysis app."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.backend.schemas import (
    BatchReviewRequest,
    BatchSentimentResponse,
    HealthResponse,
    ReviewRequest,
    SentimentResponse,
)
from src.models.sentiment_predictor import SentimentPredictor


APP_NAME = "Movie Review Sentiment Analyzer API"
APP_VERSION = "0.1.0"
DEFAULT_MODEL_PATH = "models/bert-sentiment"


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="API for predicting movie review sentiment using a fine-tuned BERT model.",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_predictor() -> SentimentPredictor:
    """Load and cache the sentiment predictor for the API process."""

    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    device = os.getenv("DEVICE")

    if device is not None and device.strip().lower() in {"", "auto"}:
        device = None

    return SentimentPredictor(model_path=model_path, device=device)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return basic API health information."""

    predictor = get_predictor()

    return HealthResponse(
        status="ok",
        model_path=str(predictor.model_path),
        device=str(predictor.device),
    )


@app.post("/predict", response_model=SentimentResponse)
def predict(request: ReviewRequest) -> dict[str, Any]:
    """Predict sentiment for a single movie review."""

    try:
        return get_predictor().predict(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc


@app.post("/predict/batch", response_model=BatchSentimentResponse)
def predict_batch(request: BatchReviewRequest) -> dict[str, list[dict[str, Any]]]:
    """Predict sentiment for a batch of movie reviews."""

    try:
        predictions = get_predictor().predict_batch(request.texts)
        return {"predictions": predictions}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Batch prediction failed.") from exc
