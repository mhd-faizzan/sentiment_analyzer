"""
train.py - Trains a Logistic Regression model on TF-IDF features for sentiment analysis.

"""

import logging
import pickle
import os
from sklearn.linear_model import LogisticRegression

# Setup logger
logger = logging.getLogger(__name__)


def train_model(X_train, y_train):
    """
    Trains a Logistic Regression model.

    Args:
        X_train: TF-IDF feature matrix for training
        y_train: Labels for training (0 or 1)

    Returns:
        Trained LogisticRegression model
    """
    logger.info("Training model...")

    model = LogisticRegression(
        max_iter=10000,
        random_state=42
    )

    model.fit(X_train, y_train)

    logger.info("Model training complete!!!!")

    return model


def save_model(model, path: str = "models/sentiment_model.pkl") -> None:
 
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}!!")


