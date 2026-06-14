"""Pydantic schemas for the sentiment analysis API."""

from pydantic import BaseModel, Field, field_validator


class ReviewRequest(BaseModel):
    """Request body for single-review sentiment prediction."""

    text: str = Field(..., min_length=1, description="Movie review text to analyze.")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject empty or whitespace-only review text."""
        if not value.strip():
            raise ValueError("Review text must not be empty.")
        return value


class BatchReviewRequest(BaseModel):
    """Request body for batch sentiment prediction."""

    texts: list[str] = Field(..., description="List of movie review texts to analyze.")

    @field_validator("texts")
    @classmethod
    def texts_must_not_contain_blank_items(cls, value: list[str]) -> list[str]:
        """Reject blank texts in a batch request."""
        for text in value:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Batch items must be non-empty strings.")
        return value


class SentimentProbabilities(BaseModel):
    """Class probabilities returned by the sentiment model."""

    negative: float = Field(..., ge=0.0, le=1.0)
    positive: float = Field(..., ge=0.0, le=1.0)


class SentimentResponse(BaseModel):
    """Response body for single-review sentiment prediction."""

    sentiment: str = Field(..., description="Predicted sentiment: positive or negative.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: SentimentProbabilities

    @field_validator("sentiment")
    @classmethod
    def sentiment_must_be_valid(cls, value: str) -> str:
        """Ensure sentiment label is one of the supported classes."""
        if value not in {"negative", "positive"}:
            raise ValueError("Sentiment must be either 'negative' or 'positive'.")
        return value


class BatchSentimentResponse(BaseModel):
    """Response body for batch sentiment prediction."""

    predictions: list[SentimentResponse]


class HealthResponse(BaseModel):
    """Response body for the health-check endpoint."""

    status: str = "ok"
