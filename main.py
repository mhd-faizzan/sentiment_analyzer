import os
import yaml
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.load_data import download_imdb_dataset, load_raw_data
from src.data.preprocess import preprocess_dataset
from src.features.build_features import build_tfidf_features, save_vectorizer
from src.models.train import train_model, save_model
from src.models.evaluate import evaluate_model
from src.models.predict import load_model, load_vectorizer, predict_sentiment

# Load config
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# Setup logging
logging.basicConfig(
    level=config['logging']['level'],
    format=config['logging']['format']
)

if __name__ == "__main__":

    # Step 1 — Download data
    download_imdb_dataset(
        save_path=config['data']['raw_path']
    )

    # Step 2 — Load or preprocess data
    processed_path = config['data']['processed_path']

    if os.path.exists(processed_path):
        print("Clean data already exists :-)")
        clean_df = pd.read_csv(processed_path)
    else:
        df = load_raw_data(path=config['data']['raw_path'])
        clean_df = preprocess_dataset(df)
        os.makedirs("data/processed", exist_ok=True)
        clean_df.to_csv(processed_path, index=False)

    # Step 3 — Split data
    X_train, X_test, y_train, y_test = train_test_split(
        clean_df['clean_text'],
        clean_df['label'],
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    print(f"Train size : {len(X_train)}")
    print(f"Test size  : {len(X_test)}")

    # Step 4 — Build or load features
    vectorizer_path = config['features']['vectorizer_path']

    if os.path.exists(vectorizer_path):
        print("Vectorizer already exists!!")
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        X_train_tfidf = vectorizer.transform(X_train)
        X_test_tfidf  = vectorizer.transform(X_test)
    else:
        X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf_features(
            X_train,
            X_test,
            max_features=config['features']['max_features']
        )
        save_vectorizer(vectorizer, path=vectorizer_path)

    # Step 5 — Train model
    model = train_model(
        X_train_tfidf,
        y_train,
        max_iter=config['model']['max_iter'],
        random_state=config['model']['random_state']
    )

    # Step 6 — Save model
    save_model(model, path=config['model']['save_path'])

    # Step 7 — Evaluate model
    evaluate_model(model, X_test_tfidf, y_test)

    # Step 8 — Test predictions
    print("\n - Testing Predictions - ")

    loaded_model      = load_model(path=config['model']['save_path'])
    loaded_vectorizer = load_vectorizer(path=config['features']['vectorizer_path'])

    test_reviews = [
        "This movie was absolutely amazing! Best film ever!",
        "Terrible movie, complete waste of time and money!",
        "The acting was great but the story was boring.",
        "One of the best films I have ever seen in my life!"
    ]

    for review in test_reviews:
        result = predict_sentiment(review, loaded_model, loaded_vectorizer)
        print(f"\nReview     : {result['review']}")
        print(f"Prediction : {result['label']}")
        print(f"Confidence : {result['confidence']}")