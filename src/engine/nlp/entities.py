"""Entity linking: companies (tickers), sectors and regions mentioned in a text."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ... import config

REGION_PATTERNS = {
    "Middle East": r"middle east|strait of hormuz|hormuz|persian gulf|\bgulf\b|iran|israel|saudi|qatar|red sea|yemen",
    "Europe": r"europe(an)?|\beu\b|eurozone|\becb\b|germany|german|france|french|italy|spain|\buk\b|britain|bank of england|euro\b",
    "Asia-Pacific": r"china|chinese|beijing|taiwan|japan|japanese|\bboj\b|korea|asia|hong kong|australia|south china sea",
    "Latin America": r"latin america|brazil|argentina|mexico|chile|colombia|peru",
    "India": r"\bindia\b|indian|\brbi\b|mumbai|rupee",
    "North America": r"\bu\.?s\.?\b|united states|america\b|american|federal reserve|\bfed\b|washington|wall street|canada|treasury",
}
SECTOR_PATTERNS = {
    "Energy": r"\boil\b|crude|brent|\bwti\b|energy|natural gas|\blng\b|opec|refin|shale|petrochem",
    "Financials": r"\bbanks?\b|lender|banking|financial institution|deposit|credit market|insurer",
    "Shipping & Logistics": r"shipping|tanker|freight|vessel|container|logistics|maritime",
    "Industrials": r"airline|aircraft|aerospace|planemaker|manufactur|industrial|airlines",
    "Technology": r"\bchips?\b|semiconductor|software|\btech\b|cloud|\bai\b|artificial intelligence|data.center",
    "Real Estate": r"real estate|commercial property|\bcre\b|reit|property market",
    "Consumer": r"retail|consumer|shoppers|e-commerce|grocery",
    "Health Care": r"pharma|drugmaker|\bfda\b|biotech|health care|healthcare|vaccine",
    "Utilities": r"utilit(y|ies)|power grid|electricity",
    "Materials": r"\bsteel\b|mining|metals|copper|aluminium|aluminum",
    "Sovereign": r"sovereign|government bond|treasur(y|ies)|gilts?|bunds?",
}
TICKER_SECTOR = {
    "Information Technology": "Technology", "Communication Services": "Technology",
    "Consumer Discretionary": "Consumer", "Consumer Staples": "Consumer", "Financials": "Financials",
    "Energy": "Energy", "Health Care": "Health Care", "Industrials": "Industrials",
}


@dataclass
class Company:
    ticker: str
    name: str
    sector: str
    base_weight: float
    pattern: re.Pattern


class EntityLinker:
    def __init__(self) -> None:
        self.companies: list[Company] = []
        for c in config.universe()["constituents"]:
            aliases = sorted(set(c["aliases"]) | {c["name"]}, key=len, reverse=True)
            alias_re = "|".join(re.escape(a) for a in aliases)
            # aliases are case-sensitive whole words; cashtags are case-insensitive
            pat = re.compile(rf"(?<![\w$])(?:{alias_re})(?!\w)|(?i:\${re.escape(c['ticker'])}\b)")
            self.companies.append(Company(c["ticker"], c["name"], c["sector"], float(c["base_weight"]), pat))
        self.by_ticker = {c.ticker: c for c in self.companies}
        self._regions = {k: re.compile(v, re.I) for k, v in REGION_PATTERNS.items()}
        self._sectors = {k: re.compile(v, re.I) for k, v in SECTOR_PATTERNS.items()}

    def companies_in(self, text: str) -> list[str]:
        return [c.ticker for c in self.companies if c.pattern.search(text)]

    def mentions(self, ticker: str, text: str) -> bool:
        return bool(self.by_ticker[ticker].pattern.search(text))

    def regions_in(self, text: str) -> list[str]:
        return [k for k, p in self._regions.items() if p.search(text)]

    def sectors_in(self, text: str, tickers: list[str] | None = None) -> list[str]:
        found = [k for k, p in self._sectors.items() if p.search(text)]
        for t in tickers or []:
            s = TICKER_SECTOR.get(self.by_ticker[t].sector)
            if s and s not in found:
                found.append(s)
        return found
