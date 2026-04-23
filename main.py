import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.load_data import download_imdb_dataset, load_raw_data
from src.data.preprocess import preprocess_dataset
from src.features.build_features import build_tfidf_features, save_vectorizer

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

    # Step 4 — Build features
    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf_features(
        X_train,
        X_test
    )

    # Step 5 — Save vectorizer
    save_vectorizer(vectorizer, path="models/vectorizer.pkl")

    print(f"\nFeature matrix shape : {X_train_tfidf.shape}")
    print("Features built successfully!!!")