# Twitter Sentiment Analyzer

A machine learning web app that classifies tweets and social media text as **Positive** or **Negative** in real time.

## Live Demo

**[twitter-sentiment-analyzer-bnepjrzr5deuhg6uv6dw6x.streamlit.app](https://twitter-sentiment-analyzer-bnepjrzr5deuhg6uv6dw6x.streamlit.app/)**

<!-- Add a screenshot here — this does more for a recruiter skimming this repo than any bullet list below.
Example:
![App screenshot](assets/demo-screenshot.png)
-->

## Overview

Built an end-to-end NLP pipeline trained on 1.6 million real tweets from the [Sentiment140 dataset](http://help.sentiment140.com/for-students). Compared 3 different ML models and deployed the best one as a live web application.

## Features

- Single tweet analysis with instant results
- Bulk analysis with sentiment distribution chart
- Shows model confidence score
- Trained on 1.6 million real tweets

## Model Performance

| Model | Accuracy |
|---|---|
| Logistic Regression | **78.15%** ✅ |
| Linear SVM | 76.36% |
| Naive Bayes | 75.92% |

Logistic Regression was chosen for deployment despite the narrow margin over SVM — it's faster at inference time and its coefficients are directly interpretable per feature, which matters more for a real-time app than the ~2% accuracy gap.

## Tech Stack

- Python
- Scikit-learn
- NLTK
- TF-IDF Vectorization
- Streamlit
- Pandas

## How It Works

1. Raw tweets are cleaned — URLs, mentions, hashtags removed
2. Text is lemmatized and stopwords removed
3. TF-IDF converts text to numerical features
4. Logistic Regression classifies sentiment
5. Result displayed with confidence score

## Run Locally

```bash
git clone https://github.com/Celeste-Hephzibah/twitter-sentiment-analyzer
cd twitter-sentiment-analyzer
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
twitter-sentiment-analyzer/
├── app.py              # Streamlit web app
├── notebooks/          # EDA and model training
├── model/              # Saved model files
└── requirements.txt
```
