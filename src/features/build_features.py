"""
build_features.py -  Converts cleaned text into numerical features using TF-IDF vectorization..

"""

import logging
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup logger
logger = logging.getLogger(__name__)


def build_tfidf_features(train_texts, test_texts, max_features: int = 5000):
    """
    Converts text data into TF-IDF numerical features.

    Args:
        train_texts : training reviews (used to fit vectorizer)
        test_texts  : testing reviews (only transformed)
        max_features: maximum number of words to consider

    Returns:
        X_train    : TF-IDF matrix for training
        X_test     : TF-IDF matrix for testing
        vectorizer : fitted TF-IDF vectorizer
    """
    logger.info("Building TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)

    logger.info(f"Feature matrix shape: {X_train.shape}")

    return X_train, X_test, vectorizer


def save_vectorizer(vectorizer, path: str = "models/vectorizer.pkl") -> None:
  
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Vectorizer saved to {path}")