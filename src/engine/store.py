"""Signal persistence: SQLite (queryable by the API/dashboard) + JSON Lines (file hand-off)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .schemas import RiskSignal

COLUMNS = ["signal_id", "doc_id", "timestamp", "source", "source_type", "entity", "entity_name", "headline",
           "sentiment_score", "sentiment_label", "event_type", "event_confidence", "impact_score", "sectors",
           "regions", "credibility", "model", "drivers", "keywords", "url"]
JSON_COLS = {"sectors", "regions", "drivers", "keywords"}


class SignalStore:
    def __init__(self, db_path: Path, jsonl_path: Optional[Path] = None) -> None:
        self.db_path, self.jsonl_path = Path(db_path), Path(jsonl_path) if jsonl_path else None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(f"CREATE TABLE IF NOT EXISTS signals ({', '.join(f'{k} TEXT' if k not in ('sentiment_score','event_confidence','impact_score','credibility') else f'{k} REAL' for k in COLUMNS)}, PRIMARY KEY(signal_id))")
            c.execute("CREATE INDEX IF NOT EXISTS ix_ts ON signals(timestamp)")
            c.execute("CREATE INDEX IF NOT EXISTS ix_entity ON signals(entity)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def clear(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM signals")
        if self.jsonl_path and self.jsonl_path.exists():
            self.jsonl_path.unlink()

    def add(self, signals: Iterable[RiskSignal]) -> int:
        rows, lines = [], []
        for s in signals:
            d = s.model_dump(mode="json")
            rows.append(tuple(json.dumps(d[k]) if k in JSON_COLS else d[k] for k in COLUMNS))
            lines.append(json.dumps(d, ensure_ascii=False))
        if not rows:
            return 0
        with self._conn() as c:
            c.executemany(f"INSERT OR REPLACE INTO signals VALUES ({','.join('?' * len(COLUMNS))})", rows)
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        return len(rows)

    def query(self, entity: Optional[str] = None, event_type: Optional[str] = None,
              min_impact: Optional[float] = None, since: Optional[str] = None, until: Optional[str] = None,
              limit: Optional[int] = None) -> pd.DataFrame:
        sql, args = "SELECT * FROM signals WHERE 1=1", []
        if entity:
            sql += " AND entity = ?"; args.append(entity.upper())
        if event_type:
            sql += " AND event_type = ?"; args.append(event_type)
        if min_impact is not None:
            sql += " AND impact_score >= ?"; args.append(float(min_impact))
        if since:
            sql += " AND timestamp >= ?"; args.append(since)
        if until:
            sql += " AND timestamp <= ?"; args.append(until)
        sql += " ORDER BY timestamp DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._conn() as c:
            df = pd.read_sql_query(sql, c, params=args)
        for k in JSON_COLS:
            if k in df:
                df[k] = df[k].map(lambda v: json.loads(v) if isinstance(v, str) else v)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        return df

    def count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
