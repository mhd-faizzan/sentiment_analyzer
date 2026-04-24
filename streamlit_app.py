"""
streamlit_app.py -  Web interface for the Sentiment Analyzer.

"""

import streamlit as st
from src.models.predict import (
    load_model,
    load_vectorizer,
    predict_sentiment
)

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Load model and vectorizer
@st.cache_resource
def load_resources():
    model      = load_model(path="models/sentiment_model.pkl")
    vectorizer = load_vectorizer(path="models/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_resources()

# Title
st.title("Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review and find out if it's **positive** or **negative**!")

# Input
review = st.text_area(
    "Enter your movie review:",
    height=150,
    placeholder="Type your review here..."
)

# Button
if st.button("Analyze Sentiment 🔍"):
    if review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        with st.spinner("Analyzing..."):
            result = predict_sentiment(review, model, vectorizer)

        # Show result
        if "positive" in result['label']:
            st.success(f"## {result['label']}")
        else:
            st.error(f"## {result['label']}")

        st.metric(
            label="Confidence",
            value=result['confidence']
        )

        st.markdown("---")
        st.markdown("**Your Review:**")
        st.write(review)