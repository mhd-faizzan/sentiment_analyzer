"""
push_models.py - Pushes trained models to Hugging Face Hub so they can be loaded during deployment.

"""

from huggingface_hub import HfApi
import os

# Your HF username and repo name
HF_USERNAME  = "faizzan"       
MODEL_REPO   = "sentiment_analyzer"

api = HfApi()

# Create repo if not exists
api.create_repo(
    repo_id=f"{HF_USERNAME}/{MODEL_REPO}",
    repo_type="model",
    exist_ok=True
)

# Upload models
api.upload_file(
    path_or_fileobj="models/sentiment_model.pkl",
    path_in_repo="sentiment_model.pkl",
    repo_id=f"{HF_USERNAME}/{MODEL_REPO}",
    repo_type="model"
)

api.upload_file(
    path_or_fileobj="models/vectorizer.pkl",
    path_in_repo="vectorizer.pkl",
    repo_id=f"{HF_USERNAME}/{MODEL_REPO}",
    repo_type="model"
)

print("Models pushed to Hugging Face Hub!")