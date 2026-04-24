"""
app.py - FastAPI application for sentiment analysis, Exposes REST API endpoints for predictions.

"""

import logging
from fastapi import FastAPI
from pydantic import BaseModel
from src.models.predict import (
    load_model,
    load_vectorizer,
    predict_sentiment
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analyzer API",
    description="Predicts sentiment of movie reviews",
    version="1.0.0"
)

# Load model and vectorizer on startup
logger.info("Loading model and vectorizer...")
model      = load_model(path="models/sentiment_model.pkl")
vectorizer = load_vectorizer(path="models/vectorizer.pkl")
logger.info("Ready to serve predictions!!")


# Define request format
class ReviewRequest(BaseModel):
    review: str


# Define response format
class PredictionResponse(BaseModel):
    review    : str
    label     : str
    confidence: str


# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "running", "message": "Sentiment Analyzer API is live!"}


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):
    """
    Predicts sentiment of a movie review.

    Args:
        request: ReviewRequest with review text

    Returns:
        PredictionResponse with label and confidence
    """
    logger.info(f"Received review: {request.review[:50]}...")

    result = predict_sentiment(
        review=request.review,
        model=model,
        vectorizer=vectorizer
    )

    return PredictionResponse(
        review    =result['review'],
        label     =result['label'],
        confidence=result['confidence']
    )