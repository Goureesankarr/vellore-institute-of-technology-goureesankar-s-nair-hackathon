"""Unit tests: run with `pytest -q`."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.engine.engine import RiskEngine
from src.engine.nlp.entities import EntityLinker
from src.engine.nlp.events import RuleEventClassifier
from src.engine.nlp.sentiment import LexiconSentiment
from src.engine.schemas import RiskSignal
from src.modules.rebalancer import SentimentRebalancer
from src.modules.stress_test import StressTester

T0 = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def engine():
    return RiskEngine(sentiment_backend="lexicon")


@pytest.mark.parametrize("text,sign", [
    ("Walmart beats estimates and raises full-year guidance", 1),
    ("Company misses expectations and cuts its outlook; shares tumble", -1),
    ("Lender files for Chapter 11 bankruptcy after missing a coupon payment", -1),
    ("Ceasefire agreed; stocks rally as tensions ease", 1),
    ("Contagion fears are overdone, the bank is a fortress", 1),
])
def test_sentiment_polarity(text, sign):
    s = LexiconSentiment().score(text).score
    assert s * sign > 0.1


def test_sentiment_bounds():
    s = LexiconSentiment().score("record record record surge soar beat estimates " * 20).score
    assert -1.0 <= s <= 1.0


@pytest.mark.parametrize("text,label", [
    ("Fed raises interest rates by 50 basis points", "Macroeconomic"),
    ("Missile strikes escalate conflict near the border", "Geopolitical"),
    ("Moody's downgrades the utility to junk status", "Credit Event"),
    ("Microsoft agrees to acquire CyberNova in all-cash deal", "Merger/Acquisition"),
    ("Apple unveils new iPhone at annual event", "Product Launch"),
    ("Regulators open probe and fine the bank for compliance failures", "Regulatory/Legal"),
])
def test_event_classifier(text, label):
    assert RuleEventClassifier().classify(text).event_type == label


def test_no_false_war_match():
    # "warns" must not trigger the Geopolitical pattern "war"
    assert RuleEventClassifier(use_supervised=False).classify("Chevron warns of higher costs").event_type != "Geopolitical"


def test_entity_linking():
    el = EntityLinker()
    assert set(el.companies_in("JPMorgan and $BAC fell while Exxon Mobil rose")) == {"JPM", "BAC", "XOM"}
    assert "Middle East" in el.regions_in("Tankers attacked near the Strait of Hormuz")


def test_engine_signal_contract(engine):
    sigs = engine.analyze_text("Harbor Point Financial defaults on bonds; JPMorgan shares slide")
    assert sigs and sigs[0].entity == "JPM"
    s = sigs[0]
    assert -1 <= s.sentiment_score <= 1 and 1 <= s.impact_score <= 10
    assert s.event_type == "Credit Event" and s.sentiment_score < 0
    assert set(s.drivers) >= {"base_severity", "sentiment", "intensity", "burst"}


def test_market_signal_for_macro(engine):
    sigs = engine.analyze_text("US inflation unexpectedly accelerates as consumer prices surge")
    assert sigs[0].entity == "MARKET" and sigs[0].event_type == "Macroeconomic"


def _sig(entity, score, impact=6.0, event="Earnings", hours=0):
    return RiskSignal(signal_id=f"x{entity}{hours}", doc_id="d", timestamp=T0 + timedelta(hours=hours),
                      source="news_file", source_type="news", entity=entity, entity_name=entity, headline="h",
                      sentiment_score=score, sentiment_label="x", event_type=event, event_confidence=0.9,
                      impact_score=impact, credibility=1.0)


def test_rebalancer_direction_and_constraints():
    rb = SentimentRebalancer()
    rb.on_signal(_sig("NVDA", 0.9))
    rb.on_signal(_sig("BA", -0.9))
    rb.rebalance(T0 + timedelta(hours=1))
    w = rb.weights
    assert abs(w.sum() - 1) < 1e-9
    assert w["NVDA"] > rb.base["NVDA"] and w["BA"] < rb.base["BA"]
    assert (w <= float(rb.p["max_weight"]) + 1e-9).all()


def test_stress_trigger_and_loss():
    st = StressTester()
    assert st.on_signal(_sig("MARKET", -0.8, impact=6.5, event="Geopolitical")) is None  # below threshold
    assert st.on_signal(_sig("MARKET", 0.8, impact=9.0, event="Geopolitical")) is None   # not adverse
    res = st.on_signal(_sig("MARKET", -0.8, impact=9.0, event="Geopolitical"))
    assert res and res["pnl_usd"] < 0 and res["value_after_usd"] < res["value_before_usd"]
    assert res["cet1_ratio_after_pct"] < res["cet1_ratio_before_pct"]


def test_stress_scales_with_impact():
    st = StressTester()
    lo = st.run("Macroeconomic", 7.5)["pnl_usd"]
    hi = st.run("Macroeconomic", 9.5)["pnl_usd"]
    assert hi < lo < 0


def test_jump_to_default():
    st = StressTester()
    base = st.run("Credit Event", 8.0)["pnl_usd"]
    jtd = st.run("Credit Event", 8.0, defaulted=["Harbor Point Financial"])["pnl_usd"]
    assert jtd < base
