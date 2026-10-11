"""Module B - Strategic, event-driven portfolio stress testing for a wholesale banking book.

Subscribes to Event Classification + Impact Score. When a signal's event type is in the trigger
list and its Impact Score exceeds the threshold (default > 7), the matching scenario from
config/shocks.yaml is scaled by the impact score and applied to every position.

Valuation (simplified, first-order where possible):
    Bonds        dP/P = -ModDur * dy + 0.5 * Convexity * dy^2,   dy = d_rates + d_spread(rating)
    Loans        credit:  -EAD * (PD' * LGD' - PD * LGD) * min(1, maturity)
                 rate:    -EAD * ModDur * d_rates          (fixed-rate loans only)
                 jump-to-default if the counterparty is named in the triggering news
    IRS          direction * notional * ModDur * d_rates   (+1 = pay fixed)
    FX forward   direction * notional * d_fx(currency)     (+1 = long foreign currency)
    CDS          direction * notional * ModDur * d_spread  (+1 = protection bought)
    Equity       MV * beta * d_equity
    FX translation applies to non-USD loans/bonds/equity.
Sector / region amplifiers apply when the news names the position's sector or region.
CET1 ratio = CET1 capital / RWA (standardised risk weights; stressed RWA uses a rating-migration proxy).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import config
from ..engine.schemas import RiskSignal

DERIV_CCF = 0.05  # exposure add-on for derivatives when computing RWA


def load_portfolio(path: Optional[Path] = None) -> pd.DataFrame:
    """Aggregate trade-level booking records into the position book used for stress testing."""
    df = pd.read_csv(path or config.path("portfolio_trades"))
    df["exposure_usd"] = np.where(df["asset_class"].isin(["Loan", "Bond", "Equity"]),
                                  df["market_value_usd"].abs(), df["notional_usd"].abs() * DERIV_CCF)
    return df


class StressTester:
    def __init__(self, portfolio: Optional[pd.DataFrame] = None, params: Optional[dict] = None) -> None:
        self.p = {**config.settings()["module_b"], **(params or {})}
        self.book = portfolio if portfolio is not None else load_portfolio()
        sh = config.shocks()
        self.scenarios = sh["scenarios"]
        self.rw = sh["risk_weights"]
        self.results: list[dict] = []
        self._triggers: list[RiskSignal] = []
        self._last_trigger: dict[str, tuple[datetime, int]] = {}
        self.cooldown = timedelta(hours=12)
        self.counterparties = sorted(self.book["counterparty"].unique(), key=len, reverse=True)

    # ---- subscription -------------------------------------------------------------------
    def should_trigger(self, sig: RiskSignal) -> bool:
        return (sig.event_type in self.p["trigger_event_types"]
                and sig.impact_score > float(self.p["impact_threshold"])
                and (sig.sentiment_score < 0 or not self.p.get("require_adverse", True)))

    def on_signal(self, sig: RiskSignal) -> Optional[dict]:
        """Trigger (or refresh) a stress test. Reports of the same event type within the cooldown
        window form one *event cluster*: the cluster keeps the highest impact seen and the union of
        sectors, regions and named counterparties, and is re-run only when that state worsens."""
        if not self.should_trigger(sig):
            return None
        named = self.named_counterparties(sig.headline)
        prev = self._last_trigger.get(sig.event_type)
        if prev and sig.timestamp - prev[0] < self.cooldown:
            idx = prev[1]
            cur = self.results[idx]
            cur["supporting_signals"] += 1
            sectors = sorted(set(cur["sectors_hit"]) | set(sig.sectors))
            regions = sorted(set(cur["regions_hit"]) | set(sig.regions))
            defaulted = sorted(set(cur["defaulted_counterparties"]) | set(named))
            worse = (sig.impact_score > cur["impact_score"] or sectors != sorted(cur["sectors_hit"])
                     or regions != sorted(cur["regions_hit"]) or defaulted != sorted(cur["defaulted_counterparties"]))
            if not worse:
                return None
            best = sig if sig.impact_score > cur["impact_score"] else self._triggers[idx]
            res = self.run(sig.event_type, max(sig.impact_score, cur["impact_score"]), sectors, regions,
                           defaulted, trigger=best, cluster_start=cur["triggered_at"])
            res["supporting_signals"] = cur["supporting_signals"]
            self.results[idx], self._triggers[idx] = res, best
            return res
        res = self.run(sig.event_type, sig.impact_score, sig.sectors, sig.regions, named, trigger=sig)
        self.results.append(res)
        self._triggers.append(sig)
        self._last_trigger[sig.event_type] = (sig.timestamp, len(self.results) - 1)
        return res

    def named_counterparties(self, text: str) -> list[str]:
        out = []
        for cp in self.counterparties:
            key = " ".join(cp.split()[:2])
            if len(key) > 6 and re.search(rf"\b{re.escape(key)}\b", text):
                out.append(cp)
        return out

    # ---- core stress engine -------------------------------------------------------------
    def run(self, event_type: str, impact: float, sectors: list[str] | None = None,
            regions: list[str] | None = None, defaulted: list[str] | None = None,
            trigger: Optional[RiskSignal] = None, cluster_start: Optional[str] = None) -> dict:
        sc = self.scenarios[event_type]
        sectors, regions, defaulted = sectors or [], regions or [], defaulted or []
        scale = min(float(self.p["max_scale"]), impact / float(self.p["reference_impact"]))
        b = self.book.copy()
        amp = np.ones(len(b))
        amp *= np.where(b["sector"].isin(sectors), float(sc.get("sector_amplifier", 1.0)), 1.0)
        amp *= np.where(b["region"].isin(regions), float(sc.get("region_amplifier", 1.0)), 1.0)
        d_r = float(sc.get("rates_bp", 0)) * scale / 1e4
        d_s = b["rating"].map(lambda r: sc["spread_bp"].get(r, 100)).astype(float) * scale * amp / 1e4
        d_eq = float(sc.get("equity_pct", 0)) / 100 * scale * amp
        fx_map = sc.get("fx_pct", {}) or {}
        d_fx = b["currency"].map(lambda c: fx_map.get(c, 0.0)).astype(float) / 100 * scale
        d_fx = d_fx * np.where(b["region"].isin(regions), 1.25, 1.0)
        pd_mult = 1 + (float(sc.get("pd_multiplier", 1.0)) - 1) * scale * amp
        lgd_add = float(sc.get("lgd_add", 0.0)) * scale * amp

        mv = b["market_value_usd"].astype(float)
        dv = np.zeros(len(b))
        cls, prod = b["asset_class"], b["product"]
        # bonds
        m = (cls == "Bond").to_numpy()
        dy = d_r + d_s
        dv[m] = (mv * (-b["mod_duration"] * dy + 0.5 * b["convexity"] * dy ** 2))[m]
        # loans
        m = (cls == "Loan").to_numpy()
        tenor = b["maturity_years"].clip(upper=1.0)
        pd0, lgd0 = b["pd"], b["lgd"]
        pd1 = np.minimum(1.0, pd0 * pd_mult)
        lgd1 = np.minimum(1.0, lgd0 + lgd_add)
        jtd = b["counterparty"].isin(defaulted).to_numpy()
        pd1 = np.where(jtd, 1.0, pd1)
        lgd1 = np.where(jtd, np.maximum(lgd1, 0.6), lgd1)
        tenor_eff = np.where(jtd, 1.0, tenor)
        credit_loss = -mv * (pd1 * lgd1 - pd0 * lgd0) * tenor_eff
        rate_loss = np.where(b["rate_type"] == "Fixed", -mv * b["mod_duration"] * d_r, 0.0)
        dv[m] = (credit_loss + rate_loss)[m]
        # bonds of a defaulted issuer: mark to recovery
        mb = (cls == "Bond").to_numpy() & jtd
        dv[mb] = (-mv * np.maximum(lgd1, 0.6))[mb]
        # derivatives
        notional, direction = b["notional_usd"].astype(float), b["direction"].astype(float)
        m = (prod == "Interest Rate Swap").to_numpy()
        dv[m] = (direction * notional * b["mod_duration"] * d_r)[m]
        m = (prod == "FX Forward").to_numpy()
        dv[m] = (direction * notional * d_fx)[m]
        m = (prod == "Credit Default Swap").to_numpy()
        dv[m] = (direction * notional * b["mod_duration"] * d_s)[m]
        # equities
        m = (cls == "Equity").to_numpy()
        dv[m] = (mv * b["beta"] * d_eq)[m]
        # FX translation of non-USD cash positions
        m = cls.isin(["Loan", "Bond", "Equity"]).to_numpy() & (b["currency"] != "USD").to_numpy()
        dv[m] += ((mv + dv) * d_fx)[m]

        b["stress_pnl_usd"] = dv
        b["value_after_usd"] = mv + dv
        b["jump_to_default"] = jtd

        # capital: CET1 ratio before / after (RWA with rating-migration proxy under stress)
        rw = b["rating"].map(lambda r: self.rw.get(r, 1.0)).astype(float)
        rwa0 = float((b["exposure_usd"] * rw).sum())
        rwa1 = float((b["exposure_usd"] * rw * (1 + 0.3 * (pd_mult - 1))).sum())
        cap0 = float(self.p["cet1_capital_usd"])
        cap1 = cap0 + float(dv.sum())

        def grp(col: str) -> list[dict]:
            g = b.groupby(col).agg(value_before=("market_value_usd", "sum"), pnl=("stress_pnl_usd", "sum"),
                                   positions=("trade_id", "count")).reset_index()
            g["value_after"] = g["value_before"] + g["pnl"]
            g = g.sort_values("pnl")
            return g.round(0).to_dict("records")

        top = b.sort_values("stress_pnl_usd").head(10)[
            ["trade_id", "counterparty", "asset_class", "product", "sector", "region", "rating", "currency",
             "market_value_usd", "stress_pnl_usd", "jump_to_default"]]
        total0 = float(mv.sum())
        res = {
            "triggered_at": cluster_start or (trigger.timestamp.isoformat() if trigger else datetime.utcnow().isoformat()),
            "updated_at": trigger.timestamp.isoformat() if trigger else None,
            "event_type": event_type, "impact_score": round(float(impact), 2), "scale": round(scale, 3),
            "headline": trigger.headline if trigger else "What-if scenario", "entity": trigger.entity if trigger else "",
            "signal_id": trigger.signal_id if trigger else None, "supporting_signals": 1,
            "sectors_hit": sectors, "regions_hit": regions, "defaulted_counterparties": defaulted,
            "scenario": {"description": sc.get("description", ""),
                         "equity_pct": round(float(sc.get("equity_pct", 0)) * scale, 2),
                         "rates_bp": round(float(sc.get("rates_bp", 0)) * scale, 1),
                         "spread_bp": {k: round(v * scale, 1) for k, v in sc["spread_bp"].items()},
                         "pd_multiplier": round(1 + (float(sc.get("pd_multiplier", 1)) - 1) * scale, 3)},
            "value_before_usd": round(total0, 0), "value_after_usd": round(total0 + float(dv.sum()), 0),
            "pnl_usd": round(float(dv.sum()), 0),
            "pnl_pct_of_book": round(float(dv.sum()) / sum(abs(mv)) * 100, 3),
            "cet1_capital_before": cap0, "cet1_capital_after": round(cap1, 0),
            "rwa_before": round(rwa0, 0), "rwa_after": round(rwa1, 0),
            "cet1_ratio_before_pct": round(cap0 / rwa0 * 100, 2), "cet1_ratio_after_pct": round(cap1 / rwa1 * 100, 2),
            "by_asset_class": grp("asset_class"), "by_sector": grp("sector"), "by_region": grp("region"),
            "by_product": grp("product"), "top_losses": top.round(0).to_dict("records"),
        }
        return res
