"""Train the supervised sentiment + event classifiers used by the engine.

Training data (all bundled, no downloads):
  * data/eval/labeled_headlines.csv   - 118 labelled dev headlines/posts
  * scripted scenario headlines from scripts/generate_synthetic_data.py (event label known)
  * background news templates (sentiment label known)
The held-out set data/eval/heldout_headlines.csv is NEVER used here.

Run:  python scripts/train_text_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_synthetic_data as g  # noqa: E402
from src.engine.nlp.supervised import MODEL_PATH, build_pipeline  # noqa: E402


def main() -> None:
    import joblib

    dev = pd.read_csv(ROOT / "data" / "eval" / "labeled_headlines.csv")
    ev_rows = [(t, e) for t, e in zip(dev["text"], dev["event_type"])]
    se_rows = [(t, s) for t, s in zip(dev["text"], dev["sentiment"])]
    for e in g.EVENTS:
        for h in e.headlines:
            ev_rows.append((h, e.event_type))
            se_rows.append((h, "positive" if e.true_sentiment > 0.2 else "negative" if e.true_sentiment < -0.2 else "neutral"))
    for name in ["Acme", "the company", "Globex"]:
        for t in g.BG_NEWS_POS:
            se_rows.append((t.format(name=name), "positive"))
        for t in g.BG_NEWS_NEG:
            se_rows.append((t.format(name=name), "negative"))
        for t in g.BG_NEWS_NEU:
            se_rows.append((t.format(name=name), "neutral"))
            ev_rows.append((t.format(name=name), "Other"))
    models = {}
    for key, rows in (("sentiment", se_rows), ("events", ev_rows)):
        X, y = zip(*rows)
        pipe = build_pipeline()
        pipe.fit(list(X), list(y))
        models[key] = pipe
        print(f"{key}: trained on {len(X)} examples, classes={list(pipe.classes_)}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODEL_PATH, compress=3)
    print(f"saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
