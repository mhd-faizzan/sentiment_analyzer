"""
predict.py - Loads trained model and vectorizer to predicts sentiment of new reviews.

"""
import os 
import logging
import pickle
from huggingface_hub import hf_hub_download
from src.data.preprocess import clean_text

# Setup logger
logger = logging.getLogger(__name__)

# Your HF username and repo
HF_USERNAME  = "faizzan"       
MODEL_REPO   = "sentiment_analyzer"


def load_model(path: str = "models/sentiment_model.pkl"):
    """
    Loads trained model from disk or HF Hub.

    Args:
        path: Local path to model

    Returns:
        Trained model
    """
    if os.path.exists(path):
        logger.info("Loading model from local disk...")
        with open(path, 'rb') as f:
            model = pickle.load(f)
    else:
        logger.info("Downloading model from HF Hub...")
        model_path = hf_hub_download(
            repo_id=f"{HF_USERNAME}/{MODEL_REPO}",
            filename="sentiment_model.pkl"
        )
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

    logger.info("Model loaded successfully!!")
    return model


def load_vectorizer(path: str = "models/vectorizer.pkl"):
    """
    Loads vectorizer from disk or HF Hub.

    Args:
        path: Local path to vectorizer

    Returns:
        Fitted vectorizer
    """
    if os.path.exists(path):
        logger.info("Loading vectorizer from local disk...")
        with open(path, 'rb') as f:
            vectorizer = pickle.load(f)
    else:
        logger.info("Downloading vectorizer from HF Hub...")
        vectorizer_path = hf_hub_download(
            repo_id=f"{HF_USERNAME}/{MODEL_REPO}",
            filename="vectorizer.pkl"
        )
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

    logger.info("Vectorizer loaded successfully!!")
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
    """
    from src.data.preprocess import clean_text

    # Step 1 — Clean the review
    cleaned = clean_text(review)

    # Step 2 — Convert to numbers
    features = vectorizer.transform([cleaned])

    # Step 3 — Predict
    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    confidence  = round(max(probability) * 100, 2)

    label = "positive" if prediction == 1 else "negative"

    return {
        "review"    : review,
        "label"     : label,
        "confidence": f"{confidence}%"
    }