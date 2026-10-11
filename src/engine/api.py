"""REST API exposing the engine's structured risk signals to downstream consumers.

Run:  python -m src.cli serve      ->  http://127.0.0.1:8000/docs  (interactive OpenAPI docs)
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .. import config
from .engine import RiskEngine
from .store import SignalStore

app = FastAPI(title="AI/NLP Risk Engine API", version="1.0.0",
              description="Structured, machine-readable financial risk signals (sentiment, event type, impact).")


@lru_cache(maxsize=1)
def _store() -> SignalStore:
    return SignalStore(config.path("signals_db"))


@lru_cache(maxsize=1)
def _engine() -> RiskEngine:
    return RiskEngine()


def _records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    df = df.copy()
    df["timestamp"] = df["timestamp"].map(lambda t: t.isoformat())
    return json.loads(df.to_json(orient="records"))


class AnalyzeRequest(BaseModel):
    text: str
    title: str = ""
    source_type: str = "news"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "signals": _store().count(), "time": datetime.now(timezone.utc).isoformat()}


@app.get("/signals")
def signals(entity: Optional[str] = None, event_type: Optional[str] = None,
            min_impact: Optional[float] = Query(None, ge=1, le=10), since: Optional[str] = None,
            until: Optional[str] = None, limit: int = Query(100, le=5000)) -> list[dict]:
    """Query signals, newest first. `entity` is a ticker or MARKET."""
    return _records(_store().query(entity, event_type, min_impact, since, until, limit))


@app.get("/signals/high-impact")
def high_impact(threshold: float = 7.0, limit: int = 50) -> list[dict]:
    return _records(_store().query(min_impact=threshold, limit=limit))


@app.get("/sentiment")
def sentiment(as_of: Optional[str] = None, half_life_hours: float = 48.0) -> list[dict]:
    """Current decayed, credibility-weighted sentiment per entity (what Module A consumes)."""
    df = _store().query(until=as_of)
    if df.empty:
        return []
    now = pd.Timestamp(as_of, tz="UTC") if as_of else df["timestamp"].max()
    age_h = (now - df["timestamp"]).dt.total_seconds() / 3600
    w = df["credibility"] * df["event_confidence"] * (0.5 + df["impact_score"] / 10) * (0.5 ** (age_h / half_life_hours))
    df = df.assign(w=w, ws=w * df["sentiment_score"])
    g = df.groupby("entity").agg(ws=("ws", "sum"), w=("w", "sum"), n=("signal_id", "count"))
    g["sentiment"] = g["ws"] / (g["w"] + 1.0)
    return [{"entity": e, "sentiment": round(float(r.sentiment), 4), "signals": int(r.n)} for e, r in g.iterrows()]


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> list[dict]:
    """Run the engine on arbitrary text (does not persist)."""
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    sigs = _engine().analyze_text(req.text, req.source_type, req.title or (req.text if req.source_type == "news" else ""))
    return [s.model_dump(mode="json") for s in sigs]


@app.get("/modules/index-weights")
def index_weights(latest: bool = True) -> list[dict]:
    p = config.path("output_dir") / "weights_history.csv"
    if not p.exists():
        raise HTTPException(404, "run `python -m src.cli demo` first")
    df = pd.read_csv(p)
    if latest:
        df = df[df["timestamp"] == df["timestamp"].max()]
    return json.loads(df.to_json(orient="records"))


@app.get("/modules/stress-tests")
def stress_tests() -> list[dict]:
    p = config.path("output_dir") / "stress_tests.json"
    if not p.exists():
        raise HTTPException(404, "run `python -m src.cli demo` first")
    res = json.load(open(p, encoding="utf-8"))
    keep = ["triggered_at", "event_type", "impact_score", "headline", "pnl_usd", "value_before_usd",
            "value_after_usd", "cet1_ratio_before_pct", "cet1_ratio_after_pct", "supporting_signals"]
    return [{k: r.get(k) for k in keep} for r in res]
