const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function predictSentiment(text) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    const message = errorData?.detail || "Prediction request failed.";
    throw new Error(message);
  }

  return response.json();
}

export async function predictSentimentBatch(texts) {
  const response = await fetch(`${API_BASE_URL}/predict/batch`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ texts }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    const message = errorData?.detail || "Batch prediction request failed.";
    throw new Error(message);
  }

  return response.json();
}