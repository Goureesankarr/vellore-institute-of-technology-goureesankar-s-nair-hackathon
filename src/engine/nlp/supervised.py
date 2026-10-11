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


def training_data() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(sentiment rows, event rows). The held-out set data/eval/heldout_headlines.csv is NEVER used."""
    import csv
    import importlib.util

    sent, ev = [], []
    with open(config.ROOT / "data" / "eval" / "labeled_headlines.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sent.append((r["text"], r["sentiment"]))
            ev.append((r["text"], r["event_type"]))
    spec = importlib.util.spec_from_file_location("gen", config.ROOT / "scripts" / "generate_synthetic_data.py")
    g = importlib.util.module_from_spec(spec)
    import sys
    sys.modules.setdefault("gen", g)
    spec.loader.exec_module(g)  # type: ignore[union-attr]
    for e in g.EVENTS:
        lab = "positive" if e.true_sentiment > 0.2 else "negative" if e.true_sentiment < -0.2 else "neutral"
        for h in e.headlines:
            ev.append((h, e.event_type))
            sent.append((h, lab))
    for name in ["Acme", "the company", "Globex"]:
        for lab, tmpls in (("positive", g.BG_NEWS_POS), ("negative", g.BG_NEWS_NEG), ("neutral", g.BG_NEWS_NEU)):
            for t in tmpls:
                sent.append((t.format(name=name), lab))
                if lab == "neutral":
                    ev.append((t.format(name=name), "Other"))
    return sent, ev


def train_models(save: bool = True) -> dict:
    models = {}
    sent, ev = training_data()
    for key, rows in (("sentiment", sent), ("events", ev)):
        X, y = zip(*rows)
        models[key] = build_pipeline().fit(list(X), list(y))
    if save:
        import joblib
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, MODEL_PATH, compress=3)
    return models


@lru_cache(maxsize=1)
def load_models() -> Optional[dict]:
    """Load the trained models; retrain in ~1s if missing or saved by an incompatible sklearn version."""
    try:
        import joblib
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # treat sklearn version-mismatch warnings as failure
            return joblib.load(MODEL_PATH)
    except Exception:  # noqa: BLE001
        try:
            return train_models(save=True)
        except Exception:  # noqa: BLE001 - scikit-learn not installed: lexicon/rules only
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
