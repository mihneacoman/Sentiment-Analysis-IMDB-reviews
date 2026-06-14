import { useState } from "react";
import { predictSentiment } from "./api";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleAnalyze() {
    const trimmedText = text.trim();

    if (!trimmedText) {
      setError("Please enter a movie review.");
      setResult(null);
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);

    try {
      const prediction = await predictSentiment(trimmedText);
      setResult(prediction);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  }

  const sentimentClass =
    result?.sentiment === "positive" ? "positive" : "negative";

  return (
    <main className="app">
      <section className="card">
        <h1>Movie Review Sentiment Analyzer</h1>

        <p className="subtitle">
          Enter a movie review and classify it as positive or negative using the
          fine-tuned BERT model.
        </p>

        <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Example: This movie was boring and badly written."
          rows={8}
        />

        <button onClick={handleAnalyze} disabled={isLoading}>
          {isLoading ? "Analyzing..." : "Analyze sentiment"}
        </button>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className={`result ${sentimentClass}`}>
            <h2>{result.sentiment}</h2>

            <p>
              Confidence: <strong>{(result.confidence * 100).toFixed(2)}%</strong>
            </p>

            <div className="probabilities">
              <p>
                Negative:{" "}
                <strong>
                  {(result.probabilities.negative * 100).toFixed(2)}%
                </strong>
              </p>
              <p>
                Positive:{" "}
                <strong>
                  {(result.probabilities.positive * 100).toFixed(2)}%
                </strong>
              </p>
            </div>
          </div>
        )}
      </section>
    </main>
  );
}

export default App;