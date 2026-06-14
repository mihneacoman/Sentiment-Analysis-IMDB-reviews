# Movie Review Sentiment and Complaint Reason Analysis

This project studies sentiment analysis on IMDB movie reviews, starting from the classical binary classification task of distinguishing positive from negative reviews. The initial objective is to compare traditional machine learning methods with modern transformer-based models on movie review sentiment classification.

The project will then extend the task beyond simple polarity detection. In particular, it will focus on negative reviews and attempt to identify the main reasons behind the negative sentiment, such as weak plot, poor acting, slow pacing, disappointing ending, or failed expectations. This turns the project from a simple sentiment classifier into a more informative review analysis system.

The dataset used in this project is based on the IMDB movie review dataset introduced by Maas et al. (2011):

Maas, Andrew L., Daly, Raymond E., Pham, Peter T., Huang, Dan, Ng, Andrew Y., & Potts, Christopher. (2011). Learning Word Vectors for Sentiment Analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, 142–150.

## Dataset

The experiments use a balanced subset of the IMDB movie review dataset. Each review is associated with a binary sentiment label: negative or positive. The raw dataset encodes sentiment as `0` for positive and `1` for negative. In this project, the labels are remapped to the more conventional format: `0` for negative and `1` for positive.

For the initial sentiment classification experiments, the dataset is divided as follows:

| Split | Negative | Positive | Total |
|---|---:|---:|---:|
| Training | 4,000 | 4,000 | 8,000 |
| Validation | 1,000 | 1,000 | 2,000 |
| Test | 1,000 | 1,000 | 2,000 |

This subset is used to keep the experiments computationally manageable while preserving a balanced evaluation setting. The same split will serve as the starting point for both classical machine learning baselines and transformer-based models.

The dataset is available through Kaggle: IMDB Sentiments⁠￼.
