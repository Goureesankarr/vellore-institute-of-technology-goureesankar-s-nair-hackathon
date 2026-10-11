"""Module A - Tactical, sentiment-driven index rebalancer.

Subscribes to the engine's Sentiment Score stream. For each constituent of the mock index:

    S_i(t)   = sum_k  w_k * s_k * exp(-lambda * age_k)  /  (sum_k w_k * exp(-lambda * age_k) + k0)
               w_k = credibility * event confidence * (0.5 + impact/10)   (shrinks to 0 with few signals)
    target_i = base_i * exp(beta * S_i)                 positive sentiment -> overweight
                                                        negative sentiment -> underweight
    target_i clipped to [min_mult * base_i, max_mult * base_i] and <= max_weight, renormalised
    w_i(t)   = (1 - alpha) * w_i(t-1) + alpha * target_i     (smoothing controls turnover)

Weights are recorded at every rebalance, and performance is measured against the static
benchmark weights using daily closes.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from .. import config
from ..engine.schemas import RiskSignal

PRIOR_STRENGTH = 1.0  # k0: pseudo-weight of a neutral prior


@dataclass
class _Obs:
    ts: datetime
    score: float
    weight: float


class SentimentRebalancer:
    def __init__(self, params: Optional[dict] = None) -> None:
        self.p = {**config.settings()["module_a"], **(params or {})}
        cons = config.universe()["constituents"]
        self.tickers = [c["ticker"] for c in cons]
        self.base = pd.Series({c["ticker"]: float(c["base_weight"]) for c in cons})
        self.base /= self.base.sum()
        self.weights = self.base.copy()
        self.obs: dict[str, list[_Obs]] = defaultdict(list)
        self.history: list[dict] = []
        self.lam = math.log(2) / (float(self.p["sentiment_half_life_hours"]) * 3600.0)

    # ---- subscription -------------------------------------------------------------------
    def on_signal(self, sig: RiskSignal) -> None:
        if sig.entity not in self.base.index:
            return
        w = sig.credibility * sig.event_confidence * (0.5 + sig.impact_score / 10.0)
        self.obs[sig.entity].append(_Obs(sig.timestamp, sig.sentiment_score, w))

    # ---- signal aggregation -------------------------------------------------------------
    def aggregate(self, as_of: datetime) -> pd.DataFrame:
        rows = []
        for t in self.tickers:
            num = den = 0.0
            n = 0
            for o in self.obs.get(t, []):
                age = (as_of - o.ts).total_seconds()
                if age < 0:
                    continue
                d = math.exp(-self.lam * age)
                num += o.weight * o.score * d
                den += o.weight * d
                n += d > 0.05
            s = num / (den + PRIOR_STRENGTH) if n >= int(self.p.get("min_signals", 1)) else 0.0
            rows.append(dict(ticker=t, sentiment=round(s, 4), fresh_signals=int(n), evidence=round(den, 3)))
        return pd.DataFrame(rows).set_index("ticker")

    # ---- rebalancing ------------------------------------------------------------------------
    def target_weights(self, sentiment: pd.Series) -> pd.Series:
        beta = float(self.p["tilt_strength"])
        tgt = self.base * np.exp(beta * sentiment.reindex(self.base.index).fillna(0.0))
        lo = self.base * float(self.p["min_multiple_of_base"])
        hi = np.minimum(self.base * float(self.p["max_multiple_of_base"]), float(self.p["max_weight"]))
        for _ in range(50):  # iterative clip + renormalise until bounds hold
            tgt = tgt.clip(lower=lo, upper=hi)
            tgt = tgt / tgt.sum()
            if ((tgt >= lo - 1e-9) & (tgt <= hi + 1e-9)).all():
                break
        return tgt

    def rebalance(self, as_of: datetime) -> pd.DataFrame:
        agg = self.aggregate(as_of)
        tgt = self.target_weights(agg["sentiment"])
        alpha = float(self.p["smoothing"])
        prev = self.weights
        new = (1 - alpha) * prev + alpha * tgt
        new = new / new.sum()
        turnover = 0.5 * float((new - prev).abs().sum())
        self.weights = new
        ts = pd.Timestamp(as_of)
        for t in self.tickers:
            self.history.append(dict(timestamp=ts, ticker=t, weight=round(float(new[t]), 6),
                                     base_weight=round(float(self.base[t]), 6), target_weight=round(float(tgt[t]), 6),
                                     sentiment=float(agg.loc[t, "sentiment"]),
                                     fresh_signals=int(agg.loc[t, "fresh_signals"]),
                                     turnover=round(turnover, 6)))
        return self.history_df().query("timestamp == @ts")

    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    # ---- evaluation ---------------------------------------------------------------------------
    @staticmethod
    def performance(history: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Daily index level of the sentiment-tilted index vs the static benchmark.

        `prices`: columns date + one column per ticker (daily closes). Weights set at the last
        rebalance of day d earn the close-to-close return from d to d+1 (no look-ahead).
        """
        if history.empty:
            return pd.DataFrame()
        px = prices.copy()
        px["date"] = pd.to_datetime(px["date"]).dt.date
        px = px.set_index("date").sort_index()
        rets = px.pct_change()
        h = history.copy()
        h["date"] = pd.to_datetime(h["timestamp"]).dt.date
        last = h.sort_values("timestamp").groupby(["date", "ticker"]).last().reset_index()
        w = last.pivot(index="date", columns="ticker", values="weight")
        b = last.pivot(index="date", columns="ticker", values="base_weight")
        rows, idx, bench = [], 100.0, 100.0
        dates = list(rets.index)
        for i in range(1, len(dates)):
            d_prev, d = dates[i - 1], dates[i]
            wp = w.loc[:d_prev].iloc[-1] if not w.loc[:d_prev].empty else b.iloc[0]
            bp = b.iloc[0]
            r = rets.loc[d].reindex(wp.index).fillna(0.0)
            idx *= 1 + float((wp * r).sum())
            bench *= 1 + float((bp * r).sum())
            rows.append(dict(date=d, index_value=round(idx, 4), benchmark_value=round(bench, 4)))
        out = pd.DataFrame(rows)
        if not out.empty:
            out = pd.concat([pd.DataFrame([dict(date=dates[0], index_value=100.0, benchmark_value=100.0)]), out])
        return out.reset_index(drop=True)

    @staticmethod
    def summary(perf: pd.DataFrame, history: pd.DataFrame) -> dict:
        if perf.empty:
            return {}
        r_i = perf["index_value"].pct_change().dropna()
        r_b = perf["benchmark_value"].pct_change().dropna()
        active = r_i - r_b
        dd = (perf["index_value"] / perf["index_value"].cummax() - 1).min()
        dd_b = (perf["benchmark_value"] / perf["benchmark_value"].cummax() - 1).min()
        turns = history.groupby("timestamp")["turnover"].first()
        return {
            "index_return_pct": round(perf["index_value"].iloc[-1] - 100, 2),
            "benchmark_return_pct": round(perf["benchmark_value"].iloc[-1] - 100, 2),
            "active_return_pct": round(perf["index_value"].iloc[-1] - perf["benchmark_value"].iloc[-1], 2),
            "tracking_error_ann_pct": round(float(active.std() * math.sqrt(252) * 100), 2),
            "max_drawdown_pct": round(float(dd) * 100, 2),
            "benchmark_max_drawdown_pct": round(float(dd_b) * 100, 2),
            "avg_turnover_pct": round(float(turns.mean() * 100), 2),
            "rebalances": int(turns.size),
        }
