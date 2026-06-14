"""Inference wrapper for the fine-tuned sentiment classification model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_PATH = Path("models/bert-sentiment")
LABELS = ("negative", "positive")


class SentimentPredictor:
    """Load a fine-tuned Hugging Face sentiment model and run inference."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the tokenizer, model, and inference device."""

        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict[str, Any]:
        """Predict sentiment for a single non-empty text."""

        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict sentiment for a batch of non-empty texts."""

        if not texts:
            return []

        self._validate_texts(texts)
        encoded_inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded_inputs = {
            key: value.to(self.device) for key, value in encoded_inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        return [self._format_prediction(row) for row in probabilities.cpu()]

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        """Resolve the requested device or choose CUDA when available."""

        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _validate_texts(texts: list[str]) -> None:
        """Validate that every batch item is a non-empty string."""

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"Text at index {index} must be a string.")
            if not text.strip():
                raise ValueError(f"Text at index {index} must not be empty.")

    @staticmethod
    def _format_prediction(probabilities: torch.Tensor) -> dict[str, Any]:
        """Convert class probabilities into the public response shape."""

        negative_probability = float(probabilities[0].item())
        positive_probability = float(probabilities[1].item())
        predicted_class = int(torch.argmax(probabilities).item())
        sentiment = LABELS[predicted_class]

        return {
            "sentiment": sentiment,
            "confidence": max(negative_probability, positive_probability),
            "probabilities": {
                "negative": negative_probability,
                "positive": positive_probability,
            },
        }


if __name__ == "__main__":
    predictor = SentimentPredictor()
    print(predictor.predict("This movie was excellent."))
    print(predictor.predict("This movie was boring and badly written."))
