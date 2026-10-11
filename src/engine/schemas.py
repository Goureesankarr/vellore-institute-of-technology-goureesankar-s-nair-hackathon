"""Data contracts shared by the engine, the API and downstream modules."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A normalised unit of unstructured text from any source."""

    doc_id: str
    source: str                     # connector id, e.g. newsapi, gdelt, reddit, news_file, social_file
    source_type: str                # "news" or "social"
    published_at: datetime
    title: str = ""
    text: str = ""
    url: Optional[str] = None
    author: Optional[str] = None
    engagement: float = 0.0         # likes + reposts etc. (social only)
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def full_text(self) -> str:
        if self.title and self.text and not self.text.startswith(self.title):
            return f"{self.title}. {self.text}"
        return self.text or self.title


class RiskSignal(BaseModel):
    """Structured, machine-readable risk signal emitted for one entity mentioned in one document."""

    signal_id: str
    doc_id: str
    timestamp: datetime
    source: str
    source_type: str
    entity: str                     # ticker, or "MARKET" for market-wide / macro items
    entity_name: str
    headline: str
    sentiment_score: float          # -1.0 (very negative) .. +1.0 (very positive)
    sentiment_label: str            # negative | neutral | positive
    event_type: str                 # Geopolitical, Macroeconomic, Credit Event, ...
    event_confidence: float         # 0..1
    impact_score: float             # 1..10 predicted market-impact severity
    sectors: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    credibility: float = 1.0
    model: str = ""                 # which sentiment backend produced the score
    drivers: dict[str, float] = Field(default_factory=dict)   # impact score decomposition
    keywords: list[str] = Field(default_factory=list)         # event evidence
    url: Optional[str] = None
