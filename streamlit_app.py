"""
streamlit_app.py - Web interface for the Sentiment Analyzer.

"""

import yaml
import streamlit as st
from src.models.predict import (
    load_model,
    load_vectorizer,
    predict_sentiment
)

# Load config
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Load model and vectorizer once
@st.cache_resource
def load_resources():
    model      = load_model(path=config['model']['save_path'])
    vectorizer = load_vectorizer(path=config['features']['vectorizer_path'])
    return model, vectorizer

model, vectorizer = load_resources()

# Header
st.title("Movie Review Sentiment Analyzer")
st.markdown("""
    **NLP model trained on 50,000 IMDB reviews**
    Enter any movie review and find out if it's positive or negative!
""")

st.markdown("---")

# Input
review = st.text_area(
    "Enter your movie review:",
    height=150,
    placeholder="Type your review here... e.g. This movie was absolutely amazing!"
)

col1, col2, col3 = st.columns([1,1,1])
with col2:
    analyze = st.button("Analyze Sentiment", use_container_width=True)

# Prediction
if analyze:
    if review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        with st.spinner("Analyzing your review..."):
            result = predict_sentiment(review, model, vectorizer)

        st.markdown("---")
        st.markdown("### Result:")

        if "positive" in result['label']:
            st.success(f"## {result['label']}")
        else:
            st.error(f"## {result['label']}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Confidence", value=result['confidence'])
        with col2:
            st.metric(label="Review Length", value=f"{len(review)} chars")

        st.markdown("---")
        with st.expander("See your review"):
            st.write(review)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built using Python, Scikit-learn, and Streamlit</p>
    </div>
""", unsafe_allow_html=True)