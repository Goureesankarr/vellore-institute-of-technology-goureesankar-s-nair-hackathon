"""Impact Score (1-10): predicted market-impact severity of a signal.

The score is a transparent, additive model so every number can be explained to a risk user:

    raw = base_severity(event_type)            # e.g. Credit Event 7.0, Product Launch 3.5
        + sentiment_term                       # 1.2*|s| if negative, 0.8*|s| if positive
        + intensity_term                       # "war", "bankruptcy", "record", "plunge" ... (capped)
        + confidence_term                      # classifier certainty
        + reach_term                           # market-wide items and large index weights matter more
        + burst_term                           # 0.3*log2(1+n) for n prior reports of this event type in 6h (cap 1)
    impact = 1 + (raw - 1) * (0.7 + 0.3 * source_credibility),  clipped to [1, 10]

The decomposition is returned as `drivers` and stored with each signal.
"""
from __future__ import annotations

import math
import re
from collections import deque
from datetime import datetime, timedelta

from ... import config


class ImpactScorer:
    def __init__(self) -> None:
        tax = config.taxonomy()
        self.base = {c: float(s["base_severity"]) for c, s in tax["categories"].items()}
        self.base[tax.get("default_category", "Other")] = float(tax.get("other_base_severity", 2.5))
        self.intensity = [(re.compile(rf"(?<![a-z])(?:{p})(?![a-z])", re.I), float(w))
                          for p, w in tax.get("intensity_terms", {}).items()]
        self.window = timedelta(hours=float(config.settings()["engine"].get("burst_window_hours", 6)))
        self._recent: deque[tuple[datetime, str]] = deque()
        weights = [c["base_weight"] for c in config.universe()["constituents"]]
        self.max_weight = max(weights)

    def reset(self) -> None:
        self._recent.clear()

    def _burst(self, ts: datetime, event_type: str) -> int:
        while self._recent and ts - self._recent[0][0] > self.window:
            self._recent.popleft()
        n = sum(1 for _, e in self._recent if e == event_type)
        self._recent.append((ts, event_type))
        return n

    def score(self, *, text: str, event_type: str, event_confidence: float, sentiment: float,
              entity_weight: float | None, credibility: float, timestamp: datetime,
              count_burst: bool = True) -> tuple[float, dict[str, float]]:
        base = self.base.get(event_type, 2.5)
        sent = (1.2 if sentiment < 0 else 0.8) * abs(sentiment)
        inten = min(2.0, sum(w for rx, w in self.intensity if rx.search(text)))
        conf = 0.8 * (event_confidence - 0.5)
        reach = 0.5 if entity_weight is None else 0.8 * entity_weight / self.max_weight
        n = self._burst(timestamp, event_type) if count_burst and event_type != "Other" else 0
        burst = min(1.0, 0.3 * math.log2(1 + n))
        raw = base + sent + inten + conf + reach + burst
        impact = 1 + (raw - 1) * (0.7 + 0.3 * credibility)
        impact = round(max(1.0, min(10.0, impact)), 2)
        drivers = {"base_severity": round(base, 2), "sentiment": round(sent, 2), "intensity": round(inten, 2),
                   "confidence": round(conf, 2), "reach": round(reach, 2), "burst": round(burst, 2),
                   "credibility_factor": round(0.7 + 0.3 * credibility, 2)}
        return impact, drivers
