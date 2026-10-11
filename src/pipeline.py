"""Orchestration: wires sources -> RiskEngine -> SignalStore + downstream modules (via the bus).

* replay(): streams the bundled datasets in timestamp order, as if they were arriving live, and
            rebalances the index on a fixed clock (default daily at the US close, 20:00 UTC).
* live():   polls the live connectors every N seconds and processes only new documents.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config
from .engine.bus import SignalBus
from .engine.engine import RiskEngine
from .engine.ingestion.base import Source
from .engine.ingestion.file_sources import NewsFileSource, SocialFileSource
from .engine.ingestion.live_sources import GDELTSource, NewsAPISource, RedditSource, RSSSource
from .engine.store import SignalStore
from .modules.rebalancer import SentimentRebalancer
from .modules.stress_test import StressTester

log = logging.getLogger(__name__)


def out_dir() -> Path:
    d = config.path("output_dir")
    d.mkdir(parents=True, exist_ok=True)
    return d


def file_sources() -> list[Source]:
    return [NewsFileSource(config.path("sample_news")), SocialFileSource(config.path("sample_social"))]


def live_sources() -> list[Source]:
    s = config.settings()["live"]
    names = " OR ".join(f'"{c["name"].split(" ")[0]}"' for c in config.universe()["constituents"][:10])
    return [NewsAPISource(query=f"({names}) AND (stock OR shares OR market)"),
            GDELTSource(query=s["gdelt_query"]), RSSSource(s["rss_feeds"]),
            RedditSource(s["reddit_subreddits"])]


class Pipeline:
    def __init__(self, sentiment_backend: Optional[str] = None, reset: bool = True) -> None:
        self.bus = SignalBus()
        self.engine = RiskEngine(sentiment_backend=sentiment_backend, bus=self.bus)
        self.store = SignalStore(config.path("signals_db"), config.path("signals_jsonl"))
        if reset:
            self.store.clear()
        self.rebalancer = SentimentRebalancer()
        self.stress = StressTester()
        # downstream modules subscribe to the engine's signal topic
        self.bus.subscribe("signals", self.rebalancer.on_signal)
        self.bus.subscribe("signals", self.stress.on_signal)

    # ------------------------------------------------------------------------------------
    def replay(self, sources: Optional[list[Source]] = None, rebalance_hour_utc: int = 20) -> dict:
        docs = [d for s in (sources or file_sources()) for d in s.safe_fetch()]
        docs.sort(key=lambda d: d.published_at)
        if not docs:
            raise RuntimeError("No documents to process")
        freq = pd.tseries.frequencies.to_offset(config.settings()["module_a"]["rebalance_frequency"])
        first = docs[0].published_at
        if freq.freqstr in ("D", "1D"):
            tick = first.replace(hour=rebalance_hour_utc, minute=0, second=0, microsecond=0)
            if tick < first:
                tick += timedelta(days=1)
        else:
            tick = pd.Timestamp(first).ceil(freq).to_pydatetime()
        t0 = time.perf_counter()
        n_sig = 0
        for d in docs:
            while d.published_at >= tick:
                if tick.weekday() < 5:
                    self.rebalancer.rebalance(tick)
                tick = (pd.Timestamp(tick) + freq).to_pydatetime()
            n_sig += self.store.add(self.engine.process(d))
        if tick.weekday() < 5:
            self.rebalancer.rebalance(tick)
        elapsed = time.perf_counter() - t0
        stats = {"documents": len(docs), "signals": n_sig, "seconds": round(elapsed, 2),
                 "docs_per_second": round(len(docs) / max(elapsed, 1e-9), 1),
                 "sentiment_model": self.engine.model_name, "mode": "replay",
                 "processed_at": datetime.now(timezone.utc).isoformat()}
        self.save_outputs(stats)
        return stats

    def live(self, interval: Optional[int] = None, iterations: Optional[int] = None) -> None:
        interval = interval or int(config.settings()["live"]["poll_interval_seconds"])
        sources = live_sources()
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        seen: set[str] = set()
        i = 0
        while iterations is None or i < iterations:
            i += 1
            docs = [d for s in sources for d in s.safe_fetch(since) if d.doc_id not in seen]
            docs.sort(key=lambda d: d.published_at)
            n = 0
            for d in docs:
                seen.add(d.doc_id)
                n += self.store.add(self.engine.process(d))
            now = datetime.now(timezone.utc)
            self.rebalancer.rebalance(now)
            since = max([since] + [d.published_at for d in docs])
            self.save_outputs({"documents": len(seen), "signals": self.store.count(), "mode": "live",
                               "sentiment_model": self.engine.model_name, "processed_at": now.isoformat()})
            log.info("live tick %d: %d new docs -> %d signals", i, len(docs), n)
            if iterations is None or i < iterations:
                time.sleep(interval)

    # ------------------------------------------------------------------------------------
    def save_outputs(self, stats: dict) -> None:
        od = out_dir()
        hist = self.rebalancer.history_df()
        hist.to_csv(od / "weights_history.csv", index=False)
        prices = load_prices()
        if prices is not None and not hist.empty:
            perf = SentimentRebalancer.performance(hist, prices)
            perf.to_csv(od / "index_performance.csv", index=False)
            stats["module_a"] = SentimentRebalancer.summary(perf, hist)
        stats["module_b"] = {"stress_tests_triggered": len(self.stress.results)}
        with open(od / "stress_tests.json", "w", encoding="utf-8") as f:
            json.dump(self.stress.results, f, indent=1, default=str)
        with open(od / "run_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=1, default=str)


def load_prices() -> Optional[pd.DataFrame]:
    """Real prices (data/output/prices_live.csv from `cli fetch-prices`) if present, else bundled sample."""
    live = config.path("output_dir") / "prices_live.csv"
    for p in (live, config.path("sample_prices")):
        if p.exists():
            return pd.read_csv(p)
    return None
