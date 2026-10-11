"""Source connector interface. Every connector returns normalised `Document`s."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from ..schemas import Document

log = logging.getLogger(__name__)


class Source(ABC):
    name: str = "source"
    source_type: str = "news"

    @abstractmethod
    def fetch(self, since: Optional[datetime] = None) -> list[Document]:
        """Return documents published after `since` (all available if None)."""

    def safe_fetch(self, since: Optional[datetime] = None) -> list[Document]:
        """Fetch without ever raising: live APIs fail, the pipeline must not."""
        try:
            docs = self.fetch(since)
            log.info("%s: fetched %d documents", self.name, len(docs))
            return docs
        except Exception as exc:  # noqa: BLE001 - connectors must be fault tolerant
            log.warning("%s: fetch failed (%s); continuing without it", self.name, exc)
            return []
