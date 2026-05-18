# Hotel Review Sentiment Analysis and Recommendation System Using NLP and Machine Learning

## Project Overview

This project is a Natural Language Processing (NLP) based Machine Learning application that analyzes hotel reviews and predicts whether the review sentiment is Positive or Negative.

The project uses hotel review text data, performs text preprocessing, feature extraction using TF-IDF Vectorization, and applies Machine Learning algorithms for sentiment classification.

A Streamlit web application is also developed for real-time sentiment prediction.

---

## Dataset Information

Dataset used:
- Hotel_Reviews.csv

Dataset contains:
- Positive reviews
- Negative reviews
- Reviewer scores
- Hotel information

A smaller sample dataset (`hotel_reviews_sample.csv`) is included for repository demonstration purposes.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Natural Language Processing (NLP)
- Scikit-learn
- Streamlit

---

## NLP Techniques Used

- Text Cleaning
- Lowercase Conversion
- Stopword Removal
- Stemming
- TF-IDF Vectorization

---

## Machine Learning Models Used

1. Logistic Regression
2. Naive Bayes

---

## Model Performance

| Model | Accuracy |
|-------|-----------|
| Logistic Regression | 87.9% |
| Naive Bayes | 86.5% |

Best Performing Model:
- Logistic Regression

---

## Project Structure

```bash
Capstone_Project_3/
│
├── DrAshimaMalikFinalProject.ipynb
├── app.py
├── requirements.txt
├── README.md
├── hotel_sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── hotel_reviews_sample.csv