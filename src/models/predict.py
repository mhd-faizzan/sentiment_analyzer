"""
predict.py - Loads trained model and vectorizer to predicts sentiment of new reviews.

"""

import logging
import pickle
from src.data.preprocess import clean_text

# Setup logger
logger = logging.getLogger(__name__)


def load_model(path: str = "models/sentiment_model.pkl"):

    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully!!")
    return model


def load_vectorizer(path: str = "models/vectorizer.pkl"):

    with open(path, 'rb') as f:
        vectorizer = pickle.load(f)
    logger.info("Vectorizer loaded successfully :-)")
    return vectorizer


def predict_sentiment(review: str, model, vectorizer) -> dict:
    """
    Predicts sentiment of a single review.

    Args:
        review    : Raw review text
        model     : Trained model
        vectorizer: Fitted vectorizer

    Returns:
        Dictionary with prediction and confidence
    
    Example:
        predict_sentiment("This movie was amazing!", model, vectorizer)
        → {"label": "positive", "confidence": 0.95}
    """

    # Step 1 — Clean the review
    cleaned = clean_text(review)

    # Step 2 — Convert to numbers
    features = vectorizer.transform([cleaned])

    # Step 3 — Predict
    prediction   = model.predict(features)[0]
    probability  = model.predict_proba(features)[0]
    confidence   = round(max(probability) * 100, 2)

    label = "positive" if prediction == 1 else "negative"

    return {
        "review"    : review,
        "label"     : label,
        "confidence": f"{confidence}%"
    }

