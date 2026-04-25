"""
training_results.py - Shows detailed training results and visualizations.

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

# Load model and vectorizer
with open("models/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load data
clean_df = pd.read_csv("data/processed/clean_reviews.csv")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    clean_df['clean_text'],
    clean_df['label'],
    test_size=0.2,
    random_state=42
)

X_test_tfidf = vectorizer.transform(X_test)
y_pred       = model.predict(X_test_tfidf)
y_prob       = model.predict_proba(X_test_tfidf)[:, 1]

# Figure with 4 plots 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Sentiment Analyzer — Detailed Results", fontsize=16)

# Plot 1 — Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive'],
    ax=axes[0, 0]
)
axes[0, 0].set_title("Confusion Matrix")
axes[0, 0].set_xlabel("Predicted")
axes[0, 0].set_ylabel("Actual")

# Plot 2 — ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc     = auc(fpr, tpr)
axes[0, 1].plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.3f}')
axes[0, 1].plot([0, 1], [0, 1], 'k--')
axes[0, 1].set_title("ROC Curve")
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].legend()

# Plot 3 — Precision Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
axes[1, 0].plot(recall, precision, color='green')
axes[1, 0].set_title("Precision-Recall Curve")
axes[1, 0].set_xlabel("Recall")
axes[1, 0].set_ylabel("Precision")

# Plot 4 — Confidence Distribution
axes[1, 1].hist(y_prob[y_test == 1], bins=50, alpha=0.5, label='Positive', color='green')
axes[1, 1].hist(y_prob[y_test == 0], bins=50, alpha=0.5, label='Negative', color='red')
axes[1, 1].set_title("Confidence Distribution")
axes[1, 1].set_xlabel("Predicted Probability")
axes[1, 1].set_ylabel("Count")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("assets/training_results.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n - Detailed Classification Report -")
print(classification_report(y_test, y_pred,
      target_names=['Negative', 'Positive']))
print(f"ROC AUC Score: {roc_auc:.4f}")