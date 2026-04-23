import logging
from src.data.load_data import download_imdb_dataset, load_raw_data

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

    print("\n--- Dataset Info ---")
    print(f"Shape     : {df.shape}")
    print(f"Columns   : {df.columns.tolist()}")
    print(f"Labels    : {df['label'].value_counts().to_dict()}")
    print(f"\nSample review:\n{df['text'][0][:300]}")