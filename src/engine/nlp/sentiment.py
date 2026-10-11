"""Sentiment models.

Backends (selected by settings.engine.sentiment_backend):

* ``finbert``  - ProsusAI/finbert transformer (score = P(positive) - P(negative)), blended 75/25 with
                 the lexicon model so social-media slang and emojis still count.
* ``lexicon``  - fully offline, explainable ensemble:
                 (1) Loughran-McDonald financial word lists,
                 (2) a curated finance phrase lexicon (e.g. "beats estimates", "cuts guidance",
                     "inflation surges"), with negation handling,
                 (3) VADER for social posts (emojis, slang, intensity), when installed.
* ``auto``     - FinBERT when transformers + weights are available, otherwise lexicon.

Every backend returns a score in [-1, 1].
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

from ... import config
from .text import clean, tokens

log = logging.getLogger(__name__)

NEGATORS = {"not", "no", "never", "n't", "without", "hardly", "neither", "nor", "fails", "failed",
            "unlikely", "isn't", "wasn't", "aren't", "don't", "doesn't", "didn't", "won't", "cannot"}

# Curated phrase lexicon: regex -> weight. Checked before single words; matched spans are not
# double-counted by the word lists. Order matters only for readability.
PHRASES: dict[str, float] = {
    # ---- dismissals of risk (must precede the risk words they contain)
    r"(contagion |recession |crash |sell-?off )?(fears|worries|concerns) (are|look|seem) (overdone|overblown|exaggerated)": 2.0,
    r"(is|remains) a fortress|fortress balance sheet": 1.5,
    # ---- earnings / guidance / analysts
    r"beat(s|ing)? (analyst )?(estimates|expectations|forecasts|consensus)": 2.0,
    r"(top|tops|topped|above|ahead of) (analyst )?(estimates|expectations|consensus|forecasts)": 2.0,
    r"miss(es|ed)? (analyst )?(estimates|expectations|forecasts|consensus)": -2.0,
    r"(below|short of) (analyst )?(estimates|expectations|consensus)": -2.0,
    r"(raise[sd]?|lift(s|ed)?|boost(s|ed)?|hike[sd]?) (its |their |full-year |annual )*(guidance|outlook|forecast|price targets?|estimates|dividend)": 2.0,
    r"(cut(s|ting)?|lower(s|ed)?|slash(es|ed)?|trim(s|med)?) (its |their |full-year |annual )*(guidance|outlook|forecast|price targets?|estimates|dividend)": -2.0,
    r"upgrade[sd]?( to buy)?": 1.5, r"downgrade[sd]?": -1.5, r"price target raised": 1.5,
    r"record (high|revenue|profit|sales|earnings)": 1.5, r"all-time high": 1.5,
    r"strong (demand|results|quarter|growth|reception|execution)": 1.5,
    r"weak(er)? (demand|results|quarter|guidance|growth)": -1.5, r"slowing demand": -1.5,
    r"(share )?buyback|share repurchase": 1.0, r"cost synergies": 0.8,
    r"bolster(s|ed)?|strengthen(s|ed)?|accretive|add to earnings|fills a gap": 0.8,
    r"margin (pressure|squeeze)": -1.2, r"higher costs": -1.0,
    # ---- macro: rising prices/yields/rates are bad news for markets
    r"(inflation|prices|costs|yields?|rates) (surge[sd]?|soar(s|ed|ing)?|jump(s|ed)?|climb(s|ed)?|spike[sd]?|accelerat(es|ed|ing))": -1.8,
    r"(inflation|cpi) (cools?|eases?|slows?|falls?)": 1.5,
    r"(rate|interest rate)s? hikes?|hikes? (interest )?rates|raises? (interest )?rates|jumbo rate hike": -1.5,
    r"rate cuts?|cuts? (interest )?rates": 1.2,
    r"higher for longer|hawkish": -1.2, r"dovish": 1.0,
    r"recession|stagflation|slowdown": -1.8, r"soft landing": 1.2,
    r"risk-off|sell-?off|flight to (quality|safety)|safe-haven": -1.5, r"risk-on|relief rally": 1.5,
    # ---- credit
    r"default(s|ed)?( on)?|missed (a )?(coupon|payment)": -2.5,
    r"bankrupt(cy)?|chapter 11|insolven(t|cy)": -2.5,
    r"contagion|bank run|deposit (outflows|flight)|liquidity crisis": -2.0,
    r"spreads? (widen(s|ed|ing)?|blow(ing)? out)|blowing out": -1.5,
    r"credit quality (remains|remained) strong": 1.5,
    r"loan losses|write-?downs?|impairments?": -1.5,
    # ---- geopolitics / operations / legal
    r"\bwar\b|invasion|airstrikes?|drone strikes?|attack(s|ed)?|military escalation|escalat(es|ed|ion)": -2.0,
    r"ceasefire|peace deal|truce|de-?escalat(es|ed|ion)": 1.8,
    r"sanctions?|embargo|blockade|tariffs?": -1.2,
    r"recall(s|ed|ing)?": -1.2, r"(probe|investigation)s?": -1.0, r"lawsuits?|class action": -1.0,
    r"fine[ds]\b|record (fine|penalty)|penalt(y|ies)": -1.2,
    r"outage|disrupt(s|ed|ion|ing)": -1.2, r"halt(s|ed)? (deliveries|production|transits|sales)|shipping halts": -1.5,
    r"layoffs?|job cuts": -1.0, r"quality (lapses|problems)": -1.5,
    r"(fda )?approv(al|es|ed)": 1.2, r"settle(s|d|ment)? .{0,40}(lawsuits?|litigation)|overhang removed|removes? .{0,25}overhang": 1.2,
    # ---- market moves
    r"shares? (rise|rose|jump(s|ed)?|gain(s|ed)?|climb(s|ed)?|rall(y|ies|ied)|surge[sd]?|soar(s|ed)?)": 1.5,
    r"shares? (fall|fell|slip(s|ped)?|slide|slid|drop(s|ped)?|sink(s)?|sank|tumble[sd]?|plunge[sd]?)": -1.5,
    r"stocks? (rall(y|ies|ied)|rebound(s|ed)?|rise|rose|climb(s|ed)?)": 1.2,
    r"stocks? (fall|fell|slip(s|ped)?|slide|tumble[sd]?|plunge[sd]?|sink|sank)": -1.2,
    r"rall(y|ies|ied)|rebound(s|ed)?|soar(s|ed)?|surge[sd]?|jump(s|ed)?": 1.0,
    r"tumble[sd]?|plunge[sd]?|plummet(s|ed)?|slump(s|ed)?|tank(s|ed|ing)|crash(es|ed)?|slide|slid|bleeding|getting (smoked|crushed|hammered|dragged)|hammered": -1.4,
    # ---- social / retail slang
    r"to the moon|\U0001F680": 1.5, r"\U0001F4C8": 1.0, r"\U0001F4C9": -1.0,
    r"bullish|long and strong|breaking out|crushed it|buying the dip|buy the dip|adding (more|back)|added more": 1.5,
    r"bearish|shorting|cutting my position|trimming|avoid until|sold my|dead money|ouch|disaster": -1.4,
    r"nothingburger|overdone": 0.6, r"game changer|monster": 1.2, r"only green": 1.2,
    r"can't catch a break|never end|not great|not what bulls wanted|ugly": -1.2,
}
_PHRASE_RES = [(re.compile(rf"(?<![a-z])(?:{p})(?![a-z])", re.I), w) for p, w in PHRASES.items()]


@dataclass
class SentimentResult:
    score: float
    label: str
    model: str
    evidence: list[str] = field(default_factory=list)


def label_for(score: float, threshold: float = 0.15) -> str:
    return "positive" if score > threshold else "negative" if score < -threshold else "neutral"


class SentimentModel(Protocol):
    name: str

    def score(self, text: str, source_type: str = "news") -> SentimentResult: ...

    def score_many(self, texts: list[str], source_type: str = "news") -> list[SentimentResult]: ...


def _load_words(folder: Path, name: str) -> set[str]:
    p = folder / f"lm_{name}.txt"
    if not p.exists():
        return set()
    return {w.strip() for w in p.read_text(encoding="utf-8").splitlines() if w.strip() and not w.startswith("#")}


class LexiconSentiment:
    """Offline, explainable finance-aware lexicon ensemble."""

    name = "lexicon"

    def __init__(self, lexicon_dir: Optional[Path] = None) -> None:
        folder = lexicon_dir or config.path("lexicons")
        self.pos = _load_words(folder, "positive")
        self.neg = _load_words(folder, "negative")
        # LM lists flag some words that are neutral in news flow; drop the noisiest.
        self.neg -= {"question", "questions", "against", "closed", "closing", "critical", "force"}
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
        except Exception:  # noqa: BLE001
            self.vader = None

    def _finance_score(self, text: str) -> tuple[float, list[str]]:
        low = clean(text).lower()
        toks = tokens(low)
        covered: list[tuple[int, int]] = []
        P = N = 0.0
        evidence: list[str] = []
        for rx, w in _PHRASE_RES:
            for m in rx.finditer(low):
                if any(a <= m.start() < b or a < m.end() <= b for a, b in covered):
                    continue
                covered.append((m.start(), m.end()))
                prefix = tokens(low[max(0, m.start() - 30):m.start()])[-3:]
                ww = -0.7 * w if any(t in NEGATORS for t in prefix) else w
                P, N = (P + ww, N) if ww > 0 else (P, N - ww)
                evidence.append(f"{m.group(0)}({ww:+.1f})")
        # single words from Loughran-McDonald, skipping spans already explained by phrases
        pos_iter = 0
        for i, t in enumerate(toks):
            idx = low.find(t, pos_iter)
            pos_iter = idx + len(t) if idx >= 0 else pos_iter
            if idx >= 0 and any(a <= idx < b for a, b in covered):
                continue
            w = 0.8 if t in self.pos else -0.8 if t in self.neg else 0.0
            if not w:
                continue
            if any(x in NEGATORS for x in toks[max(0, i - 3):i]):
                w = -0.7 * w
            P, N = (P + w, N) if w > 0 else (P, N - w)
            evidence.append(f"{t}({w:+.1f})")
        if P == 0 and N == 0:
            return 0.0, evidence
        # bounded net-tone: (P-N)/(P+N+1) rewards agreement, tanh term rewards intensity
        s = 0.6 * (P - N) / (P + N + 1.0) + 0.4 * math.tanh((P - N) / 3.0)
        return max(-1.0, min(1.0, s)), evidence

    def score(self, text: str, source_type: str = "news") -> SentimentResult:
        s, ev = self._finance_score(text)
        model = "lexicon:LM+phrases"
        if source_type == "social" and self.vader is not None:
            v = self.vader.polarity_scores(text)["compound"]
            s = 0.65 * s + 0.35 * v if ev else 0.5 * v
            model += "+vader"
        s = round(max(-1.0, min(1.0, s)), 4)
        return SentimentResult(s, label_for(s), model, ev[:8])

    def score_many(self, texts: list[str], source_type: str = "news") -> list[SentimentResult]:
        return [self.score(t, source_type) for t in texts]


class FinBERTSentiment:
    """ProsusAI/finbert via HuggingFace transformers, blended with the lexicon for slang/emojis."""

    name = "finbert"

    def __init__(self, model_name: str = "ProsusAI/finbert", lexicon: Optional[LexiconSentiment] = None):
        from transformers import pipeline  # imported lazily: optional dependency

        self.pipe = pipeline("text-classification", model=model_name, top_k=None, truncation=True,
                             max_length=256)
        self.lexicon = lexicon or LexiconSentiment()
        self._cache: dict[str, float] = {}

    def _finbert(self, texts: list[str]) -> list[float]:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            for t, out in zip(todo, self.pipe(todo, batch_size=16)):
                probs = {d["label"].lower(): d["score"] for d in out}
                self._cache[t] = probs.get("positive", 0.0) - probs.get("negative", 0.0)
        return [self._cache[t] for t in texts]

    def score_many(self, texts: list[str], source_type: str = "news") -> list[SentimentResult]:
        fb = self._finbert([clean(t)[:1000] for t in texts])
        out = []
        for t, f in zip(texts, fb):
            lex = self.lexicon.score(t, source_type)
            s = round(0.75 * f + 0.25 * lex.score, 4)
            out.append(SentimentResult(s, label_for(s), "finbert+lexicon", lex.evidence))
        return out

    def score(self, text: str, source_type: str = "news") -> SentimentResult:
        return self.score_many([text], source_type)[0]


def build_sentiment_model(backend: Optional[str] = None) -> SentimentModel:
    cfg = config.settings()["engine"]
    backend = backend or cfg.get("sentiment_backend", "auto")
    if backend in ("auto", "finbert"):
        try:
            model = FinBERTSentiment(cfg.get("finbert_model", "ProsusAI/finbert"))
            log.info("Sentiment backend: FinBERT (%s)", cfg.get("finbert_model"))
            return model
        except Exception as exc:  # noqa: BLE001
            if backend == "finbert":
                raise
            log.warning("FinBERT unavailable (%s); falling back to offline lexicon model", exc)
    return LexiconSentiment()
