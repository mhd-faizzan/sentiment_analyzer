"""
preprocess.py -- clean the raw data from given dataset before feeding them to real ML Model 

"""
import re
import logging
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# downlaod require nltk data 
nltk.download('stopwords', quiet=True)

#setup logger
logger = logging.getLogger(__name__)

#initialize stemmer
stemmer = PorterStemmer()

#load stopwords
stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:

    # Step 1 — Lowercase
    text = text.lower()

    # Step 2 — Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Step 3 — Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Step 4 — Remove stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Step 5 — Stemming
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)



# def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    
#     logger.info('Preprocessing dataset!!')

#     df = df.copy()

#     df['clean_text'] = df['text'].apply(clean_text)

#     logger.info(f"Preprocessing complete - {len(df)} reviews cleaned!!!")

#     return df

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies clean_text to entire dataset.

    Args:
        df: Raw dataframe with 'text' and 'label' columns

    Returns:
        Dataframe with cleaned text column
    """
    logger.info("Preprocessing dataset...")

    df = df.copy()
    df['clean_text'] = df['text'].apply(clean_text)

    logger.info(f"Preprocessing complete — {len(df)} reviews cleaned.")

    return df


if __name__ == "__main__":
    
    # Test with some examples
    test_reviews = [
        "I LOVED this movie!!!",
        "<br>The acting was absolutely amazing</br>",
        "Worst film I have ever seen in my life!!!",
        "The best movie ever made 100%"
    ]

    print("Testing clean_text function\n")
    for review in test_reviews:
        cleaned = clean_text(review)
        print(f"Before : {review}")
        print(f"After  : {cleaned}")
        print()


