"""Live connectors for real-time ingestion.

* NewsAPISource  - newsapi.org /v2/everything (free developer key in env NEWSAPI_KEY)
* GDELTSource    - GDELT DOC 2.0 API (no key; global news, 15-minute updates)
* RSSSource      - any RSS/Atom feed (e.g. Google News search feeds)
* RedditSource   - public subreddit JSON (social media; no key)

All connectors are fault tolerant via Source.safe_fetch: if the network or a key is missing,
the pipeline keeps running on the remaining sources.
"""
from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import requests

from ..schemas import Document
from .base import Source

UA = {"User-Agent": "Mozilla/5.0 (risk-engine hackathon prototype)"}
TIMEOUT = 15


def _id(prefix: str, key: str) -> str:
    return f"{prefix}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


class NewsAPISource(Source):
    name = "newsapi"
    source_type = "news"

    def __init__(self, query: str, api_key: Optional[str] = None, page_size: int = 50):
        self.query = query
        self.api_key = api_key or os.environ.get("NEWSAPI_KEY")
        self.page_size = page_size

    def fetch(self, since: Optional[datetime] = None) -> list[Document]:
        if not self.api_key:
            raise RuntimeError("NEWSAPI_KEY not set")
        since = since or datetime.now(timezone.utc) - timedelta(days=1)
        r = requests.get("https://newsapi.org/v2/everything", headers=UA, timeout=TIMEOUT, params={
            "q": self.query, "language": "en", "sortBy": "publishedAt", "pageSize": self.page_size,
            "from": since.strftime("%Y-%m-%dT%H:%M:%S"), "apiKey": self.api_key})
        r.raise_for_status()
        out = []
        for a in r.json().get("articles", []):
            ts = datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00"))
            out.append(Document(doc_id=_id("newsapi", a.get("url", a.get("title", ""))), source=self.name,
                                source_type=self.source_type, published_at=ts, title=a.get("title") or "",
                                text=" ".join(filter(None, [a.get("description"), a.get("content")])),
                                url=a.get("url"), author=(a.get("source") or {}).get("name")))
        return out


class GDELTSource(Source):
    name = "gdelt"
    source_type = "news"

    def __init__(self, query: str, max_records: int = 75, timespan: str = "1d"):
        self.query, self.max_records, self.timespan = query, max_records, timespan

    def fetch(self, since: Optional[datetime] = None) -> list[Document]:
        r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", headers=UA, timeout=TIMEOUT, params={
            "query": self.query, "mode": "ArtList", "format": "json", "maxrecords": self.max_records,
            "timespan": self.timespan, "sort": "DateDesc"})
        r.raise_for_status()
        out = []
        for a in r.json().get("articles", []):
            ts = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            if since and ts <= since:
                continue
            out.append(Document(doc_id=_id("gdelt", a["url"]), source=self.name, source_type=self.source_type,
                                published_at=ts, title=a.get("title", ""), text=a.get("title", ""),
                                url=a.get("url"), author=a.get("domain"),
                                meta={"country": a.get("sourcecountry"), "language": a.get("language")}))
        return out


class RSSSource(Source):
    name = "rss"
    source_type = "news"

    def __init__(self, feeds: list[str]):
        self.feeds = feeds

    def fetch(self, since: Optional[datetime] = None) -> list[Document]:
        out = []
        for feed in self.feeds:
            try:
                r = requests.get(feed, headers=UA, timeout=TIMEOUT)
                r.raise_for_status()
                root = ET.fromstring(r.content)
            except Exception:  # noqa: BLE001
                continue
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                link = item.findtext("link")
                desc = (item.findtext("description") or "").strip()
                try:
                    ts = parsedate_to_datetime(item.findtext("pubDate"))
                except Exception:  # noqa: BLE001
                    ts = datetime.now(timezone.utc)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if since and ts <= since:
                    continue
                out.append(Document(doc_id=_id("rss", link or title), source=self.name,
                                    source_type=self.source_type, published_at=ts, title=title,
                                    text=_strip_html(desc) or title, url=link))
        return out


class RedditSource(Source):
    name = "reddit"
    source_type = "social"

    def __init__(self, subreddits: list[str], limit: int = 50):
        self.subreddits, self.limit = subreddits, limit

    def fetch(self, since: Optional[datetime] = None) -> list[Document]:
        out = []
        for sub in self.subreddits:
            r = requests.get(f"https://www.reddit.com/r/{sub}/new.json", headers=UA, timeout=TIMEOUT,
                             params={"limit": self.limit})
            r.raise_for_status()
            for child in r.json().get("data", {}).get("children", []):
                p = child["data"]
                ts = datetime.fromtimestamp(p["created_utc"], tz=timezone.utc)
                if since and ts <= since:
                    continue
                out.append(Document(doc_id=_id("reddit", p["id"]), source=self.name,
                                    source_type=self.source_type, published_at=ts, title=p.get("title", ""),
                                    text=(p.get("selftext") or "")[:2000], author=p.get("author"),
                                    url="https://www.reddit.com" + p.get("permalink", ""),
                                    engagement=float(p.get("score", 0) + p.get("num_comments", 0))))
        return out


def _strip_html(s: str) -> str:
    import re
    return re.sub(r"<[^>]+>", " ", s).replace("&nbsp;", " ").strip()
