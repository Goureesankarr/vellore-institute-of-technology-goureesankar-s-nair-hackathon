"""The unified AI/NLP Risk Engine: Document -> [RiskSignal].

Pipeline per document
    1. clean + entity linking (tickers via aliases/cashtags, sectors, regions)
    2. event classification (taxonomy rules or zero-shot)
    3. sentiment: document level + entity-targeted (sentences that mention the entity)
    4. impact score (explainable additive model, see nlp/impact.py)
    5. emit one RiskSignal per entity (or one MARKET signal for macro / geopolitical items)
"""
from __future__ import annotations

import logging
import math
from typing import Iterable, Optional

from .. import config
from .bus import SignalBus
from .nlp.entities import EntityLinker
from .nlp.events import build_event_classifier
from .nlp.impact import ImpactScorer
from .nlp.sentiment import build_sentiment_model, label_for
from .nlp.text import clean, sentences
from .schemas import Document, RiskSignal

log = logging.getLogger(__name__)


class RiskEngine:
    def __init__(self, sentiment_backend: Optional[str] = None, event_backend: Optional[str] = None,
                 bus: Optional[SignalBus] = None) -> None:
        self.linker = EntityLinker()
        self.sentiment = build_sentiment_model(sentiment_backend)
        self.events = build_event_classifier(event_backend)
        self.impact = ImpactScorer()
        self.bus = bus
        self.credibility = config.settings()["engine"].get("source_credibility", {})
        self.n_docs = 0

    @property
    def model_name(self) -> str:
        from .nlp.supervised import load_models
        sup = "+tfidf-lr" if load_models() else ""
        return f"sentiment:{getattr(self.sentiment, 'name', '?')}{sup} / events:{getattr(self.events, 'name', '?')}{sup}"

    def _credibility(self, doc: Document) -> float:
        base = float(self.credibility.get(doc.source, 0.9 if doc.source_type == "news" else 0.6))
        if doc.source_type == "social":
            # engagement lends a little credibility to social posts (capped)
            base *= 0.85 + 0.15 * min(1.0, math.log10(1 + doc.engagement) / 3.0)
        return round(base, 3)

    def process(self, doc: Document) -> list[RiskSignal]:
        self.n_docs += 1
        text = clean(doc.full_text)
        if not text:
            return []
        title = doc.title or (text if doc.source_type == "social" else "")
        body = doc.text if doc.source_type == "news" else ""
        tickers = self.linker.companies_in(text)
        ev = self.events.classify(title, body)
        sectors = self.linker.sectors_in(text, tickers)
        regions = self.linker.regions_in(text)
        if not tickers and ev.event_type == "Other" and not (doc.source_type == "news" and (sectors or regions)):
            return []  # nothing a downstream consumer could act on

        # sentiment: whole document + entity-targeted sentences, scored in one batch
        sents = sentences(text) if doc.source_type == "news" else [text]
        targeted: dict[str, list[str]] = {}
        for t in tickers:
            ss = [s for s in sents if self.linker.mentions(t, s)]
            targeted[t] = ss
        doc_text = f"{doc.title}. {' '.join(sents[:4])}" if doc.title else text
        batch = [doc_text] + [s for t in tickers for s in targeted[t]]
        results = self.sentiment.score_many(batch, doc.source_type)
        doc_sent = results[0]
        cursor, ent_sent = 1, {}
        for t in tickers:
            n = len(targeted[t])
            vals = [r.score for r in results[cursor:cursor + n]]
            cursor += n
            # headline tone anchors the entity score; the entity's own sentences refine it
            ent_sent[t] = round(0.4 * doc_sent.score + 0.6 * (sum(vals) / n), 4) if vals else doc_sent.score

        cred = self._credibility(doc)
        entities = tickers or ["MARKET"]
        out: list[RiskSignal] = []
        for i, ent in enumerate(entities):
            s = ent_sent.get(ent, doc_sent.score)
            weight = self.linker.by_ticker[ent].base_weight if ent != "MARKET" else None
            impact, drivers = self.impact.score(
                text=text, event_type=ev.event_type, event_confidence=ev.confidence, sentiment=s,
                entity_weight=weight, credibility=cred, timestamp=doc.published_at, count_burst=(i == 0))
            sig = RiskSignal(
                signal_id=f"{doc.doc_id}:{ent}", doc_id=doc.doc_id, timestamp=doc.published_at,
                source=doc.source, source_type=doc.source_type, entity=ent,
                entity_name=self.linker.by_ticker[ent].name if ent != "MARKET" else "Market-wide",
                headline=(doc.title or text)[:240], sentiment_score=s, sentiment_label=label_for(s),
                event_type=ev.event_type, event_confidence=ev.confidence, impact_score=impact,
                sectors=sectors, regions=regions, credibility=cred, model=doc_sent.model,
                drivers=drivers, keywords=ev.keywords, url=doc.url)
            out.append(sig)
            if self.bus:
                self.bus.publish("signals", sig)
        return out

    def process_many(self, docs: Iterable[Document]) -> list[RiskSignal]:
        out: list[RiskSignal] = []
        for d in sorted(docs, key=lambda d: d.published_at):
            out.extend(self.process(d))
        return out

    def analyze_text(self, text: str, source_type: str = "news", title: str = "") -> list[RiskSignal]:
        """Ad-hoc analysis (API / dashboard playground). Does not affect burst state."""
        from datetime import datetime, timezone
        doc = Document(doc_id="adhoc", source="adhoc", source_type=source_type,
                       published_at=datetime.now(timezone.utc), title=title, text=text)
        saved = list(self.impact._recent)
        try:
            bus, self.bus = self.bus, None
            return self.process(doc)
        finally:
            self.bus = bus
            self.impact._recent.clear()
            self.impact._recent.extend(saved)
