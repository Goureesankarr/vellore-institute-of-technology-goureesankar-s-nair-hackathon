"""Train the supervised sentiment + event classifiers (TF-IDF + logistic regression).

Training data: data/eval/labeled_headlines.csv (dev set) + scripted scenario headlines + background
templates. The held-out set data/eval/heldout_headlines.csv is never used for training.
The engine also retrains automatically if models/text_models.joblib is missing or incompatible.

Run:  python scripts/train_text_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.nlp.supervised import MODEL_PATH, train_models  # noqa: E402

if __name__ == "__main__":
    m = train_models(save=True)
    for k, v in m.items():
        print(f"{k}: classes={list(map(str, v.classes_))}")
    print(f"saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")
