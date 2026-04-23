"""
evaluate.py - Evaluates the trained sentiment analysis model and reports performance metrics.

"""

import logging
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Setup logger
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluates model performance on test data.

    Args:
        model : Trained model
        X_test: TF-IDF feature matrix for testing
        y_test: True labels for testing

    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Evaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy  = accuracy_score(y_test, y_pred)
    report    = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    # Log results
    logger.info(f"Accuracy: {accuracy:.4f}")

    print("\n - Model Evaluation - ")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{confusion}")

    return {
        "accuracy" : accuracy,
        "report"   : report,
        "confusion": confusion
    }

