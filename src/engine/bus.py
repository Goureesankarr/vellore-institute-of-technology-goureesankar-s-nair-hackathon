"""Minimal in-process publish/subscribe bus.

Downstream modules *subscribe* to the engine's output topic ("signals"). In a production setting
this interface maps 1:1 onto Kafka / Redis Streams topics; the API exposes the same signals over HTTP.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


class SignalBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        self._subs[topic].append(callback)

    def publish(self, topic: str, message: Any) -> None:
        for cb in self._subs.get(topic, []):
            cb(message)
