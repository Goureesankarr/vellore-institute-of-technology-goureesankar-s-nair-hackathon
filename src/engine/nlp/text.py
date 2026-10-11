"""Lightweight text utilities (no heavy NLP dependencies)."""
from __future__ import annotations

import re

_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z$\"'])")
_TOKEN = re.compile(r"[a-z][a-z'\-]*|\$[a-z]{1,5}|[0-9]+(?:\.[0-9]+)?%?|[\U0001F300-\U0001FAFF]")


def clean(text: str) -> str:
    text = _URL.sub(" ", text or "")
    text = text.replace("&amp;", "&").replace("\u2019", "'").replace("\u2018", "'")
    return _WS.sub(" ", text).strip()


def sentences(text: str) -> list[str]:
    text = clean(text)
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()] if text else []


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())
