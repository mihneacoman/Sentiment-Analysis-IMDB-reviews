# Movie Review Sentiment and Complaint Reason Analysis

This project studies sentiment analysis on IMDB movie reviews. It starts with binary sentiment classification, comparing classical machine learning methods with transformer-based models for distinguishing positive and negative reviews.

The project then extends beyond polarity detection by focusing on negative reviews and identifying the main reasons behind the negative sentiment, such as weak plot, poor acting, slow pacing, disappointing ending, or failed expectations.

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

TF-IDF with Logistic Regression is the strongest classical baseline. Error analysis suggests that many mistakes involve mixed sentiment or contrastive phrasing, motivating the use of transformer-based models.

## Transformer Baselines

Transformer models are fine-tuned on the same balanced training set and evaluated on the held-out test set. Training and inference times were measured on Google Colab with GPU acceleration.

| Model | Validation Accuracy | Test Accuracy | Test F1 | Training Time | Inference Time |
|---|---:|---:|---:|---:|---:|
| DistilBERT base uncased | 0.9095 | 0.8980 | 0.898 | ~10 min | — |
| BERT base uncased | 0.9175 | 0.9115 | 0.912 | ~21 min | ~1.6 ms/review |

Both transformer models outperform the classical baselines. BERT base uncased gives the strongest result so far and is used as the default sentiment model for the next stage of the project.

The BERT model reduces false positives on negative reviews compared with DistilBERT, which is important because the next stage focuses on analyzing negative reviews in more detail.

Inference time was measured on 100 short sample reviews using batched inference with batch size 16, so it should be interpreted as an approximate GPU throughput estimate rather than single-request latency.

## Next Stage: Negative Review Reason Detection

The next part of the project focuses on negative reviews and attempts to identify the main reason for dissatisfaction. Since the dataset does not provide reason labels, this stage will require weak supervision, manual annotation, clustering, or a combination of these methods.

Potential reason categories include:

| Reason Category | Description |
|---|---|
| Weak plot | The story is poorly structured, illogical, or uninteresting. |
| Poor acting | The performances are unconvincing or distracting. |
| Slow pacing | The movie feels too slow, repetitive, or unnecessarily long. |
| Disappointing ending | The ending fails to satisfy expectations. |
| Failed expectations | The reviewer expected more because of the director, cast, genre, or premise. |

## Reference

Maas, Andrew L., Daly, Raymond E., Pham, Peter T., Huang, Dan, Ng, Andrew Y., & Potts, Christopher. (2011). *Learning Word Vectors for Sentiment Analysis*. In *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies*, 142–150.

Dataset available through Kaggle: [IMDB Sentiments](https://www.kaggle.com/datasets/jcblaise/imdb-sentiments).