"""Event classification.

``RuleEventClassifier`` (default) scores each category in config/taxonomy.yaml with weighted regex
patterns (title matches count double). It is fast, deterministic and explainable: the matched
keywords are returned as evidence. ``ZeroShotEventClassifier`` is an optional transformer backend
(facebook/bart-large-mnli) for broader recall when GPU/CPU budget allows.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ... import config
from .supervised import event_proba

log = logging.getLogger(__name__)


@dataclass
class EventResult:
    event_type: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)


class RuleEventClassifier:
    name = "rules"

    def __init__(self, min_score: float = 2.0, use_supervised: bool = True) -> None:
        self.use_supervised = use_supervised
        tax = config.taxonomy()
        self.default = tax.get("default_category", "Other")
        self.min_score = min_score
        self.cats = {
            cat: [(re.compile(rf"(?<![a-z])(?:{pat})(?![a-z])", re.I), float(w)) for pat, w in spec["patterns"].items()]
            for cat, spec in tax["categories"].items()
        }

    def classify(self, title: str, body: str = "") -> EventResult:
        scores: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}
        for cat, pats in self.cats.items():
            s, ev = 0.0, []
            for rx, w in pats:
                n_title = len(rx.findall(title)) if title else 0
                n_body = len(rx.findall(body)) if body else 0
                hits = min(2, n_title) * 2 + min(3, n_body)
                if hits:
                    s += w * hits
                    m = rx.search(title) or rx.search(body)
                    ev.append(m.group(0).lower())
            if s:
                scores[cat], evidence[cat] = round(s, 2), ev
        if not scores or max(scores.values()) < self.min_score:
            res = EventResult(self.default, 0.5, scores, [])
        else:
            best = max(scores, key=scores.get)
            total = sum(scores.values())
            # confidence: share of evidence for the winner, boosted by absolute evidence strength
            conf = 0.5 * scores[best] / total + 0.5 * min(1.0, scores[best] / 12.0)
            res = EventResult(best, round(min(0.99, 0.35 + 0.65 * conf), 3), scores, evidence[best][:6])
        if self.use_supervised and res.confidence < 0.6:
            sup = event_proba(f"{title}. {body}" if body else title)
            if sup and sup[1] > 0.35 and sup[0] != res.event_type:
                res = EventResult(sup[0], round(min(0.6, sup[1]), 3), scores, res.keywords + ["(tfidf-lr)"])
        return res


class ZeroShotEventClassifier:
    name = "zero_shot"

    def __init__(self, model_name: str = "facebook/bart-large-mnli") -> None:
        from transformers import pipeline

        self.pipe = pipeline("zero-shot-classification", model=model_name)
        self.labels = list(config.taxonomy()["categories"].keys())
        self.rules = RuleEventClassifier()

    def classify(self, title: str, body: str = "") -> EventResult:
        text = f"{title}. {body}"[:1000]
        out = self.pipe(text, candidate_labels=self.labels + ["Other"], multi_label=False)
        best, conf = out["labels"][0], float(out["scores"][0])
        rule = self.rules.classify(title, body)
        return EventResult(best, round(conf, 3), dict(zip(out["labels"], out["scores"])), rule.keywords)


def build_event_classifier(backend: str | None = None):
    cfg = config.settings()["engine"]
    backend = backend or cfg.get("event_backend", "rules")
    if backend == "zero_shot":
        try:
            return ZeroShotEventClassifier(cfg.get("zero_shot_model", "facebook/bart-large-mnli"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Zero-shot classifier unavailable (%s); using rules", exc)
    return RuleEventClassifier()
