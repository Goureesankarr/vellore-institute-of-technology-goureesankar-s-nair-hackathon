"""Download real daily closes for the index constituents with yfinance (needs internet)."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from . import config


def fetch_prices(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    import yfinance as yf

    tickers = [c["ticker"] for c in config.universe()["constituents"]]
    start = start or (date.today() - timedelta(days=120)).isoformat()
    end = end or date.today().isoformat()
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    df = raw.reset_index().rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    out = config.path("output_dir") / "prices_live.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} days x {len(tickers)} tickers -> {out}")
    return df
