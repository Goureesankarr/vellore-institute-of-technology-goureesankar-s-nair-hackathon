"""Model evaluation and ablations.

    python -m src.cli evaluate               # labelled eval set + Module A ablation
    python -m src.cli evaluate --phrasebank  # + Financial PhraseBank (downloads from HuggingFace)

Writes data/output/evaluation.json and prints a summary.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Optional

import pandas as pd

from . import config
from .engine.nlp.events import RuleEventClassifier
from .engine.nlp.sentiment import LexiconSentiment, build_sentiment_model, label_for

log = logging.getLogger(__name__)
LABELS = ["negative", "neutral", "positive"]


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    f1s = {}
    for c in sorted(set(y_true)):
        tp = sum(a == c and b == c for a, b in zip(y_true, y_pred))
        fp = sum(a != c and b == c for a, b in zip(y_true, y_pred))
        fn = sum(a == c and b != c for a, b in zip(y_true, y_pred))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s[c] = round(2 * p * r / (p + r), 3) if p + r else 0.0
    conf = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred")).reindex(
        index=labels, columns=labels, fill_value=0)
    return {"n": len(y_true), "accuracy": round(acc, 4), "macro_f1": round(sum(f1s.values()) / len(f1s), 4),
            "per_class_f1": f1s, "confusion": conf.to_dict()}


def _source_type(text: str) -> str:
    return "social" if ("$" in text or any(ord(ch) > 0x1F000 for ch in text) or text.endswith("?")) else "news"


def eval_sentiment(df: pd.DataFrame, backend: Optional[str]) -> dict:
    out = {}
    lex = LexiconSentiment()
    texts, y = df["text"].tolist(), df["sentiment"].tolist()
    st = [_source_type(t) for t in texts]
    # naive baselines (the "before" in before/after)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        v = SentimentIntensityAnalyzer()
        out["baseline: VADER (general-purpose)"] = metrics(y, [label_for(v.polarity_scores(t)["compound"], 0.05) for t in texts])
    except Exception:  # noqa: BLE001
        pass
    lm_only = []
    for t in texts:
        toks = t.lower().split()
        p = sum(w.strip(".,!?") in lex.pos for w in toks)
        n = sum(w.strip(".,!?") in lex.neg for w in toks)
        lm_only.append(label_for((p - n) / (p + n + 1)))
    out["baseline: Loughran-McDonald words only"] = metrics(y, lm_only)
    lex_only = LexiconSentiment(use_supervised=False)
    out["engine: lexicon ensemble (LM + phrases + VADER)"] = metrics(
        y, [lex_only.score(t, s).label for t, s in zip(texts, st)])
    out["engine: lexicon ensemble + TF-IDF/LR (default offline)"] = metrics(
        y, [lex.score(t, s).label for t, s in zip(texts, st)])
    model = build_sentiment_model(backend)
    if model.name != "lexicon":
        res = [model.score(t, s).label for t, s in zip(texts, st)]
        out["engine: FinBERT + lexicon"] = metrics(y, res)
    return out


def eval_events(df: pd.DataFrame) -> dict:
    y = df["event_type"].tolist()
    major = Counter(y).most_common(1)[0][0]
    rules = RuleEventClassifier(use_supervised=False)
    hybrid = RuleEventClassifier()
    return {"baseline: majority class": metrics(y, [major] * len(y)),
            "engine: taxonomy rules only": metrics(y, [rules.classify(t, "").event_type for t in df["text"]]),
            "engine: rules + TF-IDF/LR fallback (default)": metrics(y, [hybrid.classify(t, "").event_type for t in df["text"]])}


def eval_phrasebank(backend: Optional[str]) -> dict:
    from datasets import load_dataset  # optional dependency
    ds = load_dataset("takala/financial_phrasebank", "sentences_allagree", split="train", trust_remote_code=True)
    names = ds.features["label"].names
    texts = ds["sentence"]
    y = [names[i] for i in ds["label"]]
    out = {"engine: lexicon ensemble": metrics(y, [r.label for r in LexiconSentiment().score_many(texts)])}
    model = build_sentiment_model(backend)
    if model.name != "lexicon":
        out["engine: FinBERT + lexicon"] = metrics(y, [r.label for r in model.score_many(texts)])
    return out


def module_a_ablation() -> list[dict]:
    """Re-run the rebalancer on the stored signals under different settings (no look-ahead)."""
    from .engine.schemas import RiskSignal
    from .engine.store import SignalStore
    from .modules.rebalancer import SentimentRebalancer
    from .pipeline import load_prices

    df = SignalStore(config.path("signals_db")).query().sort_values("timestamp")
    prices = load_prices("replay")
    if df.empty or prices is None:
        return []
    df = df.astype(object).where(df.notna(), None)
    sigs = [RiskSignal(**{**r, "timestamp": r["timestamp"].to_pydatetime()}) for r in df.to_dict("records")]
    days = pd.to_datetime(prices["date"]).tolist()
    variants = {
        "Engine settings (decay 48h, smoothing 60%, credibility-weighted)": {},
        "Naive: no decay, no smoothing": {"sentiment_half_life_hours": 1e6, "smoothing": 1.0},
        "No turnover control (smoothing off)": {"smoothing": 1.0},
        "Fast decay (12h half-life)": {"sentiment_half_life_hours": 12},
    }
    rows = []
    for name, params in variants.items():
        rb = SentimentRebalancer(params)
        j = 0
        for d in days:
            tick = pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(hours=20)
            while j < len(sigs) and sigs[j].timestamp < tick:
                rb.on_signal(sigs[j]); j += 1
            rb.rebalance(tick.to_pydatetime())
        h = rb.history_df()
        s = SentimentRebalancer.summary(SentimentRebalancer.performance(h, prices), h)
        rows.append({"variant": name, "active_return_pct": s["active_return_pct"],
                     "tracking_error_ann_pct": s["tracking_error_ann_pct"], "avg_turnover_pct": s["avg_turnover_pct"],
                     "max_drawdown_pct": s["max_drawdown_pct"]})
    return rows


def run_evaluation(backend: Optional[str] = None, phrasebank: bool = False) -> dict:
    dev = pd.read_csv(config.ROOT / "data" / "eval" / "labeled_headlines.csv")
    test = pd.read_csv(config.ROOT / "data" / "eval" / "heldout_headlines.csv")
    result = {"datasets": {
                  "dev": "data/eval/labeled_headlines.csv - 118 author-labelled items; USED while tuning the lexicon/taxonomy",
                  "heldout": "data/eval/heldout_headlines.csv - 60 author-labelled items written after tuning; never tuned on"},
              "sentiment": eval_sentiment(test, backend), "events": eval_events(test),
              "sentiment_dev": eval_sentiment(dev, backend), "events_dev": eval_events(dev)}
    if phrasebank:
        try:
            result["phrasebank_allagree"] = eval_phrasebank(backend)
        except Exception as exc:  # noqa: BLE001
            result["phrasebank_allagree"] = {"error": str(exc)}
    try:
        result["module_a_ablation"] = module_a_ablation()
    except Exception as exc:  # noqa: BLE001
        log.warning("ablation failed: %s", exc)
    out = config.path("output_dir") / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out, "w", encoding="utf-8"), indent=1)
    for sect in ("sentiment", "events", "sentiment_dev", "events_dev", "phrasebank_allagree"):
        for name, m in result.get(sect, {}).items():
            if isinstance(m, dict) and "accuracy" in m:
                print(f"[{sect}] {name:55s} acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}  n={m['n']}")
    for r in result.get("module_a_ablation", []):
        print(f"[module A] {r['variant']:60s} active={r['active_return_pct']:+.2f}%  TE={r['tracking_error_ann_pct']:.2f}%  turnover={r['avg_turnover_pct']:.2f}%")
    print(f"\nSaved {out}")
    return result
