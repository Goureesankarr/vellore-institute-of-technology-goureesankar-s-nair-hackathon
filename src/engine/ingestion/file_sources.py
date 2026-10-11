"""File-based connectors: replay bundled CSV datasets (news + social) as if they were live feeds."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ..schemas import Document
from .base import Source


def _parse_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, format="ISO8601")


class NewsFileSource(Source):
    """CSV columns: doc_id, outlet, published_at, title, body, url."""

    name = "news_file"
    source_type = "news"

    def __init__(self, path: Path):
        self.path = Path(path)

    def fetch(self, since: Optional[datetime] = None) -> list[Document]:
        df = pd.read_csv(self.path)
        df["published_at"] = _parse_ts(df["published_at"])
        if since is not None:
            df = df[df["published_at"] > pd.Timestamp(since)]
        return [
            Document(doc_id=str(r.doc_id), source=self.name, source_type=self.source_type,
                     published_at=r.published_at.to_pydatetime(), title=str(r.title),
                     text=str(r.body), url=getattr(r, "url", None), author=getattr(r, "outlet", None))
            for r in df.itertuples(index=False)
        ]


class SocialFileSource(Source):
    """CSV columns: post_id, author, published_at, text, likes, reposts."""

    name = "social_file"
    source_type = "social"

    def __init__(self, path: Path):
        self.path = Path(path)

    def fetch(self, since: Optional[datetime] = None) -> list[Document]:
        df = pd.read_csv(self.path)
        df["published_at"] = _parse_ts(df["published_at"])
        if since is not None:
            df = df[df["published_at"] > pd.Timestamp(since)]
        return [
            Document(doc_id=str(r.post_id), source=self.name, source_type=self.source_type,
                     published_at=r.published_at.to_pydatetime(), title="", text=str(r.text),
                     author=str(r.author), engagement=float(r.likes + r.reposts))
            for r in df.itertuples(index=False)
        ]
