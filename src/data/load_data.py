import os
import logging
import pandas as pd
from datasets import load_dataset

# Setup logger for this module
logger = logging.getLogger(__name__)


def download_imdb_dataset(save_path: str = "data/raw/reviews.csv") -> None:
    """
    Downloads the IMDB dataset and saves it as a CSV file.

    Args:
        save_path: Path where the CSV file will be saved.

    Returns:
        None

    Example:
        download_imdb_dataset(save_path="data/raw/reviews.csv")
    """
    # Don't download again if file already exists
    if os.path.exists(save_path):
        logger.info(f"Dataset already exists at {save_path} — skipping download.")
        return

    logger.info("Downloading IMDB dataset...")

    # Load from HuggingFace datasets
    dataset = load_dataset("imdb")

    # Combine train and test splits into one dataframe
    train_df = pd.DataFrame(dataset["train"])
    test_df  = pd.DataFrame(dataset["test"])
    full_df  = pd.concat([train_df, test_df], ignore_index=True)

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    full_df.to_csv(save_path, index=False)

    logger.info(f"Dataset saved to {save_path} — {len(full_df)} rows total.")


def load_raw_data(path: str = "data/raw/reviews.csv") -> pd.DataFrame:
    """
    Loads the raw IMDB dataset from CSV.

    Args:
        path: Path to the CSV file.

    Returns:
        pd.DataFrame with columns: text, label

    Example:
        df = load_raw_data("data/raw/reviews.csv")
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}. Run download first.")

    logger.info(f"Loading data from {path}...")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows successfully.")

    return df