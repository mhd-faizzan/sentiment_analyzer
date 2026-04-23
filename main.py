import os
import logging
from src.data.load_data import download_imdb_dataset, load_raw_data
from src.data.preprocess import preprocess_dataset


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":

    # Step 1 — Download data
    download_imdb_dataset(save_path="data/raw/reviews.csv")

    # Step 2 — Load and inspect
    df = load_raw_data(path="data/raw/reviews.csv")

    print(f"\nRaw data shape: {df.shape}")
    print(df.head(2))

    # Step 3 — Preprocess
    clean_df = preprocess_dataset(df)

    print(f"\nClean data shape: {clean_df.shape}")
    print(clean_df[['text', 'clean_text']].head(2))
    
os.makedirs("data/processed", exist_ok=True)

clean_df.to_csv("data/processed/clean_reviews.csv", index=False)

print(" Clean data saved to data/processed/clean_reviews.csv!!!")

