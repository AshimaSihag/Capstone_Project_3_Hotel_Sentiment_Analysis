import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

model = pickle.load(open('hotel_sentiment_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = text.replace("donot", "do not")
    text = text.replace("dont", "do not")
    text = text.replace("didnt", "did not")
    text = text.replace("doesnt", "does not")
    text = text.replace("isnt", "is not")
    text = text.replace("wasnt", "was not")
    text = text.replace("wont", "will not")
    text = text.replace("cant", "can not")

    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

def predict_sentiment(review):
    review_lower = review.lower()

    review_lower = review_lower.replace("donot", "do not")
    review_lower = review_lower.replace("dont", "do not")
    review_lower = review_lower.replace("didnt", "did not")
    review_lower = review_lower.replace("doesnt", "does not")
    review_lower = review_lower.replace("isnt", "is not")
    review_lower = review_lower.replace("wasnt", "was not")
    review_lower = review_lower.replace("wont", "will not")
    review_lower = review_lower.replace("cant", "can not")

    negative_phrases = [
        "do not like",
        "did not like",
        "does not like",
        "not like",
        "not good",
        "not satisfied",
        "not happy",
        "not clean",
        "not comfortable",
        "bad",
        "worst",
        "dirty",
        "terrible",
        "poor",
        "awful",
        "horrible",
        "disappointing",
        "rude",
        "uncomfortable",
        "waste",
        "pathetic"
    ]

    if any(phrase in review_lower for phrase in negative_phrases):
        return "Negative Review 😔"

    cleaned_review = preprocess_text(review)
    vector_input = tfidf.transform([cleaned_review]).toarray()
    prediction = model.predict(vector_input)[0]

    if prediction == 1:
        return "Positive Review 😊"
    else:
        return "Negative Review 😔"

st.set_page_config(
    page_title="Hotel Review Sentiment Analysis",
    page_icon="🏨",
    layout="centered"
)

st.title("🏨 Hotel Review Sentiment Analysis")

st.write(
    "This app predicts whether a hotel review is Positive or Negative using NLP and Machine Learning."
)

review = st.text_area("Enter Hotel Review:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a hotel review.")
    else:
        result = predict_sentiment(review)

        if "Positive" in result:
            st.success(result)
        else:
            st.error(result)

st.markdown("---")
st.write("Final Project | Capstone 3 | Machine Learning and NLP")