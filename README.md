# Movie Review Sentiment and Complaint Reason Analysis

This project studies sentiment analysis on IMDB movie reviews. It starts with binary sentiment classification, comparing classical machine learning methods with transformer-based models for distinguishing positive and negative reviews.

The project will then extend beyond polarity detection by focusing on negative reviews and identifying the main reasons behind the negative sentiment, such as weak plot, poor acting, slow pacing, disappointing ending, or failed expectations.

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

TF-IDF with Logistic Regression is the strongest classical baseline so far. Error analysis suggests that many mistakes involve mixed sentiment or contrastive phrasing, motivating the next stage of transformer-based modeling.

## Reference

Maas, Andrew L., Daly, Raymond E., Pham, Peter T., Huang, Dan, Ng, Andrew Y., & Potts, Christopher. (2011). *Learning Word Vectors for Sentiment Analysis*. In *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies*, 142–150.

Dataset available through Kaggle: [IMDB Sentiments](https://www.kaggle.com/datasets/jcblaise/imdb-sentiments).