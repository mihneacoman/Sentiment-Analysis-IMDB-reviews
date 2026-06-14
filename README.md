# Movie Review Sentiment and Complaint Reason Analysis

This project studies sentiment analysis on IMDB movie reviews. It starts with binary sentiment classification, comparing classical machine learning methods with transformer-based models for distinguishing positive and negative reviews.

The project then extends beyond polarity detection by experimenting with negative review reason detection. The goal is to identify why a negative review is negative, for example because of weak plot, poor acting, slow pacing, failed expectations, or poor production quality.

The final BERT sentiment model is exposed through a small web application with a FastAPI backend and a React frontend.

## Dataset

The project uses the IMDB movie review dataset introduced by Maas et al. (2011). Reviews with IMDB ratings lower than 5 are labeled as negative, while reviews with ratings greater than or equal to 6 are labeled as positive.

For the initial sentiment classification experiments, a balanced subset is used:

| Split | Negative | Positive | Total |
|---|---:|---:|---:|
| Training | 4,000 | 4,000 | 8,000 |
| Validation | 1,000 | 1,000 | 2,000 |
| Test | 1,000 | 1,000 | 2,000 |

The raw dataset encodes sentiment as `0` for positive and `1` for negative. In this project, labels are remapped to the more conventional format: `0` for negative and `1` for positive.

## Classical Baselines

Initial experiments evaluate Bag-of-Words and TF-IDF representations with classical machine learning models.

| Model | Validation Accuracy |
|---|---:|
| Bag-of-Words + Logistic Regression | 0.8445 |
| TF-IDF + Naive Bayes | 0.8515 |
| TF-IDF + Linear SVM | 0.8580 |
| TF-IDF + Logistic Regression | 0.8670 |

TF-IDF with Logistic Regression is the strongest classical baseline. Error analysis shows that many mistakes involve mixed sentiment, contrastive phrasing, and reviews where the overall sentiment is not expressed directly.

## Transformer Models

Transformer models are fine-tuned on the same balanced training set and evaluated on the held-out test set. Training and inference times were measured on Google Colab with GPU acceleration.

| Model | Validation Accuracy | Test Accuracy | Test F1 | Training Time | Inference Time |
|---|---:|---:|---:|---:|---:|
| DistilBERT base uncased | 0.9095 | 0.8980 | 0.898 | ~10 min | — |
| BERT base uncased | 0.9175 | 0.9115 | 0.912 | ~21 min | ~1.6 ms/review |

Both transformer models outperform the classical baselines. BERT base uncased gives the strongest result and is used as the default sentiment model in the application.

The BERT model also reduces false positives on negative reviews compared with DistilBERT, which is useful because the second part of the project focuses on analyzing negative reviews in more detail.

Inference time was measured on 100 short sample reviews using batched inference with batch size 16. It should be interpreted as an approximate GPU throughput estimate rather than single-request latency.

The fine-tuned BERT model is hosted on Hugging Face Hub:

```text
mihneacoman/bert-imdb-sentiment
```

## Negative Review Reason Detection

The second part of the project explores why negative reviews are negative. Since the IMDB dataset does not provide reason labels, this part uses a rule-based weak labeling approach.

A `NegativeReasonDetector` was implemented using keyword patterns, phrase matching, and simple text normalization. The detector assigns one or more reason categories to a negative review when there is enough lexical evidence.

Current reason categories include:

| Reason Category | Description |
|---|---|
| `bad_acting` | Complaints about acting or performances. |
| `weak_plot_bad_writing` | Complaints about the story, script, or writing. |
| `boring_slow_pacing` | Complaints about boredom, pacing, or excessive length. |
| `disappointing_ending` | Complaints about the ending. |
| `failed_expectations` | The movie did not meet expectations. |
| `weak_characters` | Complaints about character development or characterization. |
| `bad_dialogue` | Complaints about dialogue or lines. |
| `confusing_story` | Complaints about confusing or unclear storytelling. |
| `poor_visuals_effects` | Complaints about visuals, CGI, or effects. |
| `generic_unoriginal` | Complaints about lack of originality. |
| `poor_direction_execution` | Complaints about direction or execution. |
| `tonal_mismatch` | Complaints about inconsistent tone. |
| `factual_inaccuracy` | Complaints about historical or factual inaccuracies. |
| `not_funny` | Complaints that a comedy was not funny. |
| `poor_production_quality` | Complaints about low-budget or poor technical quality. |
| `other_uncertain` | No confident reason detected. |

On 4,000 negative reviews, the detector currently identifies at least one reason for 1,682 reviews.

| Output Type | Count | Percentage |
|---|---:|---:|
| At least one detected reason | 1,682 | 42.05% |
| `other_uncertain` | 2,318 | 57.95% |

This method is useful as an interpretable baseline and as a weak supervision tool, but it is not treated as a final reason classification model. Many reviews require semantic understanding beyond keyword matching, especially when the complaint is implicit, sarcastic, or expressed indirectly.

## Web Application

The selected BERT sentiment model is wrapped in a reusable `SentimentPredictor` class. The class handles tokenization, inference, probability computation, and conversion from model outputs to readable sentiment labels.

A FastAPI backend exposes the model through REST endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Checks whether the API is running. |
| `/predict` | POST | Predicts sentiment for one movie review. |
| `/predict/batch` | POST | Predicts sentiment for multiple reviews. |

The backend uses Pydantic schemas for request validation and response formatting. Empty or whitespace-only reviews are rejected before inference.

A simple React frontend built with Vite allows the user to enter a movie review and view the predicted sentiment, confidence score, and class probabilities.

## Project Structure

```text
src/
  models/
    sentiment_predictor.py
    negative_reason_detector.py

app/
  backend/
    main.py
    schemas.py
  frontend/
    src/
      api.js
      App.jsx
      App.css

models/
  README.md
```

## Setup

Clone the repository:

```bash
git clone <repo-url>
cd <repo-name>
```

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

The default model is loaded from Hugging Face Hub:

```text
mihneacoman/bert-imdb-sentiment
```

Optionally, create a local `.env` file from the example:

```bash
cp .env.example .env
```

The expected variables are:

```env
MODEL_PATH=mihneacoman/bert-imdb-sentiment
DEVICE=auto
```

## Running the Application

Start the backend from the project root:

```bash
MODEL_PATH=mihneacoman/bert-imdb-sentiment uvicorn app.backend.main:app --reload
```

On first run, the model is downloaded from Hugging Face Hub and cached locally.

Start the frontend in another terminal:

```bash
cd app/frontend
npm install
npm run dev
```

Then open:

```text
http://localhost:5173/
```

The FastAPI documentation is available at:

```text
http://localhost:8000/docs
```

## Quick API Test

With the backend running, test the sentiment endpoint:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was boring and badly written."}'
```

Expected output format:

```json
{
  "sentiment": "negative",
  "confidence": 0.9978,
  "probabilities": {
    "negative": 0.9978,
    "positive": 0.0021
  }
}
```

Exact probabilities may vary slightly depending on hardware and library versions.

## Limitations

The BERT model is trained and evaluated on a balanced subset of IMDB reviews, so results may differ on other types of movie reviews.

The negative reason detector is rule-based and should be interpreted as an experimental baseline, not as a final supervised reason classifier.

## Reference

Maas, Andrew L., Daly, Raymond E., Pham, Peter T., Huang, Dan, Ng, Andrew Y., & Potts, Christopher. (2011). *Learning Word Vectors for Sentiment Analysis*. In *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies*, 142–150.

Dataset available through Kaggle: [IMDB Sentiments](https://www.kaggle.com/datasets/jcblaise/imdb-sentiments).