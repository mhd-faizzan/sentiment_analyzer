import os
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.load_data import download_imdb_dataset, load_raw_data
from src.data.preprocess import preprocess_dataset
from src.features.build_features import build_tfidf_features, save_vectorizer
from src.models.train import train_model, save_model
from src.models.evaluate import evaluate_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":

    # Step 1 — Download data
    download_imdb_dataset(save_path="data/raw/reviews.csv")

    # Step 2 — Load or preprocess data
    processed_path = "data/processed/clean_reviews.csv"

    if os.path.exists(processed_path):
        print("Clean data already exists :-)")
        clean_df = pd.read_csv(processed_path)
    else:
        df = load_raw_data(path="data/raw/reviews.csv")
        clean_df = preprocess_dataset(df)
        os.makedirs("data/processed", exist_ok=True)
        clean_df.to_csv(processed_path, index=False)

    # Step 3 — Split data
    X_train, X_test, y_train, y_test = train_test_split(
        clean_df['clean_text'],
        clean_df['label'],
        test_size=0.2,
        random_state=42
    )

    print(f"Train size : {len(X_train)}")
    print(f"Test size  : {len(X_test)}")


    vectorizer_path = "models/vectorizer.pkl"

    if os.path.exists(vectorizer_path):
        print("Vectorizer already exists!!")
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        X_train_tfidf = vectorizer.transform(X_train)
        X_test_tfidf  = vectorizer.transform(X_test)
    else:
        X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf_features(
            X_train,
            X_test
        )
        save_vectorizer(vectorizer, path=vectorizer_path)

    # Step 5 — Train model
    model = train_model(X_train_tfidf, y_train)

    # Step 6 — Save model
    save_model(model, path="models/sentiment_model.pkl")

    # Step 7 — Evaluate model
    evaluate_model(model, X_test_tfidf, y_test)
