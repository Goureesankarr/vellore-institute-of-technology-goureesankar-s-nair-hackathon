"""Supervised text classifiers (TF-IDF word+char n-grams -> logistic regression).

Trained by `python scripts/train_text_models.py` on the labelled dev set plus the scripted
headlines of the synthetic scenario. They complement the lexicon / taxonomy rules:
  * sentiment: blended with the lexicon score, weight 0.5*(1-|lexicon|) (defers to strong lexicon evidence)
  * events:    used when the rules are unsure (no category, or confidence < 0.6)
Stored in models/text_models.joblib (a few hundred KB).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from ... import config

MODEL_PATH = config.ROOT / "models" / "text_models.joblib"


def build_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer

    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", feats), ("clf", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"))])


@lru_cache(maxsize=1)
def load_models() -> Optional[dict]:
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception:  # noqa: BLE001
        return None


def sentiment_proba(text: str) -> Optional[float]:
    m = load_models()
    if not m:
        return None
    clf = m["sentiment"]
    p = dict(zip(clf.classes_, clf.predict_proba([text])[0]))
    return float(p.get("positive", 0.0) - p.get("negative", 0.0))


def event_proba(text: str) -> Optional[tuple[str, float]]:
    m = load_models()
    if not m:
        return None
    clf = m["events"]
    probs = clf.predict_proba([text])[0]
    i = int(probs.argmax())
    return str(clf.classes_[i]), float(probs[i])
