"""Generate the synthetic demo datasets bundled in /data/sample.

Everything produced here is SYNTHETIC. Company tickers are real S&P 100 names so the index
looks familiar, but every headline, post, price path, counterparty and trade is invented.
Fictional counterparties (e.g. "Harbor Point Financial") do not refer to real entities.

Outputs
-------
data/sample/news_articles.csv     financial-news style articles (source 1)
data/sample/social_posts.csv      X/Twitter-style posts (source 2)
data/sample/prices_daily.csv      daily closes for the 15 index constituents
data/sample/wholesale_trades.csv  trade-level booking records for the banking book (Module B)
data/sample/scenario_ground_truth.csv  the scripted events behind the stream (for reference)

Run:  python scripts/generate_synthetic_data.py
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample"
SEED = 42
START = datetime(2026, 8, 17, tzinfo=timezone.utc)
N_DAYS = 30  # trading days

OUTLETS = ["Synthetic Markets Wire", "Demo Financial Times-Desk", "Mock Business Daily",
           "Sample Street Journal", "Simulated Reuters-style Feed", "Example Capital News"]
HANDLES = ["@quant_quinn", "@valuehunter", "@macro_maya", "@chartist_raj", "@deeptech_dev",
           "@bondvigilante", "@retail_ravi", "@optionsoracle", "@divi_dan", "@fintwit_fiona",
           "@energy_ed", "@bank_analyst", "@tech_tara", "@longterm_lee", "@bearish_bob",
           "@momentum_mo", "@risk_ronan", "@em_emma", "@yield_yuki", "@gamma_gita"]

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "BAC", "XOM",
           "CVX", "JNJ", "PFE", "BA", "WMT"]
NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon",
         "GOOGL": "Alphabet", "META": "Meta", "TSLA": "Tesla", "JPM": "JPMorgan",
         "BAC": "Bank of America", "XOM": "Exxon Mobil", "CVX": "Chevron",
         "JNJ": "Johnson & Johnson", "PFE": "Pfizer", "BA": "Boeing", "WMT": "Walmart"}
SECTOR = {"AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "AMZN": "consumer", "GOOGL": "tech",
          "META": "tech", "TSLA": "consumer", "JPM": "bank", "BAC": "bank", "XOM": "energy",
          "CVX": "energy", "JNJ": "health", "PFE": "health", "BA": "industrial", "WMT": "consumer"}
START_PRICE = {"AAPL": 231.0, "MSFT": 505.0, "NVDA": 178.0, "AMZN": 224.0, "GOOGL": 201.0,
               "META": 748.0, "TSLA": 338.0, "JPM": 298.0, "BAC": 49.5, "XOM": 114.0,
               "CVX": 158.0, "JNJ": 176.0, "PFE": 25.2, "BA": 221.0, "WMT": 99.0}


@dataclass
class ScenarioEvent:
    day: int                      # trading-day index
    hour: int                     # UTC hour of first report
    event_type: str
    tickers: list[str]            # affected tickers ([] = market-wide)
    true_sentiment: float         # -1..1 (for the price simulation)
    true_impact: float            # 1..10 (reference only)
    headlines: list[str]
    bodies: list[str]
    posts: list[str]
    market_move: float = 0.0      # market-wide return shock on the day
    ticker_moves: dict = field(default_factory=dict)  # extra idiosyncratic shocks


# --------------------------------------------------------------------------------------
# Scripted storyline (ground truth). Each event yields several articles and many posts.
# --------------------------------------------------------------------------------------
EVENTS: list[ScenarioEvent] = [
    ScenarioEvent(
        1, 13, "Product Launch", ["NVDA"], 0.7, 5.5,
        ["Nvidia unveils next-generation Rubin Ultra AI accelerator, claims 3x inference gains",
         "Nvidia launches new data-center chip as hyperscalers line up early orders",
         "Analysts cheer Nvidia's new AI platform; several raise price targets"],
        ["Nvidia introduced its next-generation accelerator at a developer event on Tuesday, saying the chip "
         "delivers roughly three times the inference throughput of the prior generation. Several cloud providers "
         "confirmed early orders, and analysts said the launch strengthens the company's lead in AI hardware.",
         "The new product will be available to customers in the fourth quarter. Shares rose in early trading as "
         "brokers lifted estimates, citing strong demand visibility and improving margins."],
        ["$NVDA new chip looks like a monster. 3x inference?? Demand is not slowing down 🚀",
         "Nvidia keeps executing. Raised my PT after the launch event. $NVDA",
         "Every hyperscaler pre-ordering the new Nvidia platform. Bullish $NVDA 📈",
         "Not sure the stock can keep ripping but the Rubin launch was impressive $NVDA",
         "$NVDA launch day pump again lol. Strong product though."],
        ticker_moves={"NVDA": 0.035}),
    ScenarioEvent(
        2, 18, "Macroeconomic", [], -0.5, 6.0,
        ["Fed minutes show officials open to further rate hikes as inflation stays sticky",
         "Treasury yields climb after hawkish Fed minutes; stocks slip",
         "FOMC minutes: several policymakers warn inflation risks remain tilted to the upside"],
        ["Minutes of the Federal Reserve's last policy meeting showed a number of officials saw a case for "
         "additional interest rate increases if inflation fails to cool. Treasury yields rose and equities "
         "slipped as investors priced a higher path for rates.",
         "The two-year yield climbed 9 basis points. Rate-sensitive sectors underperformed, while bank shares "
         "were mixed."],
        ["Fed minutes way more hawkish than expected. Yields ripping, tech bleeding",
         "Higher for longer is back on the menu. Not great for multiples.",
         "2y up 9bps on the minutes. Market not ready for more hikes",
         "Fed sounds worried about inflation again. Trimming risk here."],
        market_move=-0.009),
    ScenarioEvent(
        3, 11, "Earnings", ["WMT"], 0.6, 5.0,
        ["Walmart beats estimates and raises full-year outlook as shoppers trade down",
         "Walmart quarterly profit tops expectations; e-commerce sales jump 22%",
         "Walmart lifts guidance, shares hit record high"],
        ["Walmart reported quarterly earnings per share above consensus and raised its full-year forecast, "
         "citing strong grocery sales and continued growth in e-commerce. Management said higher-income "
         "households continued to shift spending to the retailer.",
         "Comparable sales rose 4.6%, ahead of estimates, and the company said its advertising business grew "
         "more than 30%."],
        ["$WMT crushed it. Raised guidance in this environment is huge",
         "Walmart is the defensive growth stock nobody talks about. Record high $WMT",
         "E-comm +22% at Walmart. Amazon should be paying attention $WMT $AMZN",
         "Solid beat from Walmart, guidance raise is the cherry on top"],
        ticker_moves={"WMT": 0.045}),
    ScenarioEvent(
        5, 6, "Geopolitical", ["XOM", "CVX"], -0.8, 8.6,
        ["Tankers attacked near the Strait of Hormuz; oil prices surge as shipping halts",
         "Middle East tensions escalate after drone strikes on vessels in the Gulf; Brent jumps 9%",
         "Global stocks tumble as Strait of Hormuz conflict threatens energy supplies",
         "Shipping insurers suspend cover in the Gulf amid military escalation"],
        ["Several commercial tankers were struck by drones near the Strait of Hormuz early on Monday, prompting "
         "major shipping companies to halt transits through the waterway. Brent crude surged 9% and global "
         "equities tumbled as investors fled to safe-haven assets amid fears of a wider regional conflict.",
         "Analysts warned that a prolonged disruption could push oil above $100 a barrel and reignite inflation. "
         "Energy producers such as Exxon Mobil and Chevron rallied, while airlines, industrials and emerging-market "
         "currencies sold off sharply. Credit spreads on high-yield bonds widened by the most in months."],
        ["Strait of Hormuz tankers hit. This is a serious escalation. Oil +9%",
         "Risk-off everywhere. Futures tanking on the Middle East news",
         "Brent going vertical. $XOM $CVX the only green on my screen",
         "If Hormuz stays closed we are looking at $100+ oil and an inflation scare",
         "HY spreads blowing out on the Gulf conflict. Credit desks are busy today",
         "Sold my airline names. War risk is not something I want to hold through",
         "This conflict could get much worse. Hedging with puts",
         "Emerging market FX getting hammered on the Gulf escalation"],
        market_move=-0.028, ticker_moves={"XOM": 0.06, "CVX": 0.055, "BA": -0.035}),
    ScenarioEvent(
        7, 15, "Regulatory/Legal", ["BA"], -0.7, 6.8,
        ["FAA opens investigation into Boeing 737 MAX production after new quality lapses",
         "Boeing shares slide as regulators halt deliveries pending probe",
         "Boeing faces fresh regulatory scrutiny; delivery halt could hit cash flow"],
        ["The Federal Aviation Administration said it had opened an investigation into Boeing's 737 MAX "
         "production line after inspectors found new quality lapses, and ordered a temporary halt to deliveries. "
         "The probe could delay the planemaker's cash-flow recovery.",
         "Analysts said the halt could cost about $1 billion a month in delayed payments and cut estimates."],
        ["Boeing again?? Another 737 MAX probe. This company can't catch a break",
         "Delivery halt is a big hit to Boeing's free cash flow story",
         "Regulators halting Boeing deliveries. Avoid until clarity",
         "Boeing quality problems never end. Shorting the bounce"],
        ticker_moves={"BA": -0.065}),
    ScenarioEvent(
        8, 12, "Merger/Acquisition", ["MSFT"], 0.35, 6.2,
        ["Microsoft agrees to acquire cybersecurity firm CyberNova in $28 billion all-cash deal",
         "Microsoft's $28 billion CyberNova takeover would bolster its security cloud",
         "Microsoft bets big on security with largest acquisition in three years"],
        ["Microsoft said it agreed to buy cybersecurity company CyberNova for $28 billion in cash, its biggest "
         "deal since its gaming acquisition. The company expects the deal to add to earnings within two years "
         "and to close next year, subject to antitrust review.",
         "Analysts said the acquisition fills a gap in Microsoft's security portfolio, although the price "
         "implies a rich multiple."],
        ["$MSFT buying CyberNova for $28B. Pricey but strategic",
         "Microsoft security cloud just got a lot stronger. Like the deal",
         "Big acquisition from Microsoft. Antitrust review could drag",
         "$MSFT keeps adding moats. Long and strong"],
        ticker_moves={"MSFT": 0.012}),
    ScenarioEvent(
        10, 9, "Credit Event", ["JPM", "BAC"], -0.85, 9.0,
        ["Regional lender Harbor Point Financial defaults on bonds, files for Chapter 11 bankruptcy",
         "Harbor Point Financial collapse sparks fears of bank contagion; credit spreads widen sharply",
         "Moody's downgrades several mid-size banks after Harbor Point default; bank stocks plunge",
         "Deposit outflows and liquidity crisis fears hit bank shares after Harbor Point bankruptcy"],
        ["Harbor Point Financial, a mid-size commercial real-estate lender, missed a coupon payment and filed for "
         "Chapter 11 bankruptcy protection on Thursday after deposit outflows accelerated. The default sent "
         "credit default swap spreads on regional banks sharply wider and triggered a broad sell-off in bank stocks.",
         "Large lenders including JPMorgan and Bank of America fell as investors weighed counterparty exposure and "
         "the risk of contagion. Ratings agencies placed several mid-size banks on review for downgrade, and "
         "analysts warned of rising loan losses in commercial real estate."],
        ["Harbor Point just defaulted. Chapter 11. This could spread to other regionals",
         "Bank stocks plunging. $JPM $BAC getting dragged down by the Harbor Point collapse",
         "CDS on regionals blowing out. Feels like 2023 all over again",
         "Deposit flight is the real risk here. Watching $BAC closely",
         "Contagion fears are overdone for the big banks imo, $JPM is a fortress",
         "CRE loan losses were always going to hit someone. Harbor Point is the first domino",
         "Downgrades rolling in for mid-size banks. Credit is cracking"],
        market_move=-0.018, ticker_moves={"JPM": -0.045, "BAC": -0.07}),
    ScenarioEvent(
        12, 17, "Product Launch", ["AAPL"], 0.55, 5.0,
        ["Apple unveils iPhone 18 lineup with on-device AI features at annual event",
         "Apple launches new iPhones and Watch; pre-orders open Friday",
         "Apple's new iPhone gets warm reception from analysts on AI upgrades"],
        ["Apple introduced its new iPhone lineup on Tuesday, highlighting on-device artificial intelligence "
         "features and a faster in-house chip. The devices will be available for pre-order on Friday.",
         "Analysts said the upgrade cycle could be stronger than last year, citing an aging installed base."],
        ["iPhone 18 looks solid. AI features might finally drive an upgrade cycle $AAPL",
         "Apple event was fine. Nothing revolutionary but steady $AAPL",
         "Pre-ordering the new iPhone day one. Apple ecosystem is too good",
         "$AAPL launch event, sell the news as usual?"],
        ticker_moves={"AAPL": 0.012}),
    ScenarioEvent(
        13, 14, "Product Launch", ["PFE"], 0.65, 5.2,
        ["Pfizer wins FDA approval for new obesity drug, shares jump",
         "FDA approves Pfizer's once-daily weight-loss pill in boost to drugmaker",
         "Pfizer's approval opens door to multibillion-dollar obesity market"],
        ["The Food and Drug Administration approved Pfizer's oral weight-loss treatment, giving the drugmaker "
         "a long-awaited entry into the fast-growing obesity market. Analysts estimate peak sales could exceed "
         "$5 billion.",
         "Pfizer said the drug would launch in the United States within weeks."],
        ["$PFE finally gets its obesity pill approved. Huge for the turnaround",
         "FDA approval for Pfizer! This stock has been dead money for years",
         "Pfizer obesity drug approval is a game changer for sentiment $PFE"],
        ticker_moves={"PFE": 0.06}),
    ScenarioEvent(
        15, 13, "Regulatory/Legal", ["TSLA"], -0.55, 5.8,
        ["Tesla recalls 400,000 vehicles over steering software defect",
         "Regulators push Tesla recall as safety investigation widens",
         "Tesla shares fall after large recall, safety probe"],
        ["Tesla is recalling about 400,000 vehicles in the United States to fix a steering software defect, "
         "according to a filing with the National Highway Traffic Safety Administration, which also widened "
         "its safety investigation.",
         "The fix will be delivered over the air, but analysts said the probe adds regulatory risk."],
        ["Another Tesla recall. OTA fix but the headlines are ugly $TSLA",
         "Tesla safety probe widening is not what bulls wanted",
         "$TSLA recall is a nothingburger, it's a software update",
         "Regulatory risk piling up for Tesla. Trimming"],
        ticker_moves={"TSLA": -0.04}),
    ScenarioEvent(
        17, 12, "Macroeconomic", [], -0.7, 7.8,
        ["US inflation unexpectedly accelerates; CPI rises 0.6% as energy prices surge",
         "Hot CPI report sends Treasury yields soaring, stocks slide",
         "Inflation shock revives fears of more Fed rate hikes and recession"],
        ["US consumer prices rose 0.6% last month, well above economists' expectations, as energy costs surged "
         "following the Gulf supply disruption. Core inflation also accelerated, sending the 10-year Treasury "
         "yield up 18 basis points.",
         "Traders now price a significant chance of a half-point rate hike at the next Federal Reserve meeting, "
         "and several economists raised their recession probabilities."],
        ["CPI 0.6% m/m. That is a disaster. Fed has to hike",
         "10y yield +18bps on the inflation print. Duration getting crushed",
         "Stagflation risk is real now. Energy shock feeding straight into CPI",
         "Hot inflation = more hikes = recession. Going defensive",
         "Market pricing a 50bp hike now after that CPI"],
        market_move=-0.022, ticker_moves={"JPM": 0.01}),
    ScenarioEvent(
        19, 10, "Regulatory/Legal", ["META", "GOOGL"], -0.6, 7.3,
        ["European Commission fines Meta EUR 4 billion under Digital Markets Act",
         "EU regulators fine Meta and open new antitrust probe into Google ad tech",
         "Meta hit with record EU penalty; Alphabet faces fresh investigation"],
        ["The European Commission fined Meta Platforms 4 billion euros for breaching the Digital Markets Act and "
         "opened a new antitrust investigation into Google's advertising technology business.",
         "Both companies said they would appeal. Analysts said the penalties are manageable but regulatory risk "
         "in Europe is rising."],
        ["Record EU fine for Meta. Europe really hates US big tech $META",
         "$GOOGL another EU antitrust probe. These never end",
         "EU fines are a cost of doing business for Meta. Buying the dip",
         "Regulatory overhang on big tech growing again $META $GOOGL"],
        ticker_moves={"META": -0.03, "GOOGL": -0.018}),
    ScenarioEvent(
        21, 18, "Macroeconomic", [], -0.6, 8.2,
        ["Federal Reserve raises interest rates by 50 basis points to fight inflation",
         "Fed delivers jumbo rate hike, signals more tightening ahead",
         "Stocks fall, dollar rallies after Fed's surprise 50bp hike"],
        ["The Federal Reserve raised its benchmark interest rate by half a percentage point and signalled that "
         "further increases may be needed, citing the renewed surge in inflation. Treasury yields jumped and "
         "equities fell.",
         "Fed Chair said the central bank was prepared to accept slower economic growth to restore price stability. "
         "The dollar rallied against emerging-market currencies."],
        ["Fed hikes 50. Powell sounds very hawkish. Ouch",
         "Rate hike + more to come = recession 2027?",
         "Banks might actually like higher rates. $JPM net interest income up",
         "The Fed just broke something. Long-duration tech getting smoked",
         "Dollar ripping after the Fed. EM pain trade continues"],
        market_move=-0.016, ticker_moves={"JPM": 0.012, "BAC": 0.01}),
    ScenarioEvent(
        22, 15, "Other", ["AMZN"], -0.45, 5.0,
        ["Amazon Web Services suffers major outage, disrupting thousands of websites",
         "AWS outage knocks banking apps and streaming services offline for hours"],
        ["A major outage at Amazon Web Services disrupted thousands of websites and apps for several hours on "
         "Wednesday. Amazon said the issue was caused by a networking configuration error and has been resolved.",
         "Analysts said the incident was unlikely to have a lasting financial impact but could raise questions "
         "about cloud concentration risk."],
        ["AWS down. Half the internet is broken right now",
         "This AWS outage is a reminder of how concentrated cloud risk is $AMZN",
         "Amazon outage resolved. Buying the dip, nothing changes long term"],
        ticker_moves={"AMZN": -0.02}),
    ScenarioEvent(
        24, 11, "Earnings", ["JPM"], 0.55, 5.2,
        ["JPMorgan raises net interest income outlook at investor day; shares rise",
         "JPMorgan says credit quality remains strong despite regional bank stress"],
        ["JPMorgan Chase raised its full-year net interest income forecast at its investor day and said credit "
         "quality across its loan book remained strong, easing concerns about contagion from the Harbor Point "
         "default.",
         "The bank also announced a larger share buyback."],
        ["$JPM raising NII guidance and a bigger buyback. Fortress balance sheet",
         "Dimon says credit is fine. Good enough for me $JPM",
         "JPMorgan investor day was a strong reassurance for the sector"],
        ticker_moves={"JPM": 0.03, "BAC": 0.015}),
    ScenarioEvent(
        26, 7, "Geopolitical", ["XOM", "CVX"], 0.6, 6.4,
        ["Ceasefire agreed in Gulf; shipping through the Strait of Hormuz to resume",
         "Oil prices tumble as Middle East ceasefire eases supply fears; stocks rally",
         "Global markets rebound after Gulf ceasefire deal"],
        ["Regional powers agreed to a ceasefire on Monday, and major shipping companies said they would resume "
         "transits through the Strait of Hormuz. Brent crude fell 7% and global equities rallied as supply "
         "fears eased.",
         "Energy stocks gave back some of their recent gains, while airlines and emerging-market assets rebounded."],
        ["Ceasefire!! Futures ripping. Risk-on is back",
         "Oil dumping on the ceasefire news. $XOM $CVX giving back gains",
         "Strait of Hormuz reopening is great news for inflation",
         "Relief rally on the Gulf ceasefire. Adding back equity exposure"],
        market_move=0.019, ticker_moves={"XOM": -0.045, "CVX": -0.04, "BA": 0.02}),
    ScenarioEvent(
        27, 13, "Regulatory/Legal", ["JNJ"], 0.3, 5.0,
        ["Johnson & Johnson reaches $6.5 billion settlement to resolve talc lawsuits",
         "J&J settlement removes major litigation overhang, analysts say"],
        ["Johnson & Johnson agreed to pay $6.5 billion to resolve the bulk of its talc litigation, a settlement "
         "analysts said removes a significant overhang on the stock.",
         "The company said the agreement does not include an admission of wrongdoing."],
        ["$JNJ finally settling the talc mess. Overhang removed",
         "J&J settlement is expensive but certainty is worth it",
         "Litigation clarity for Johnson & Johnson, bullish"],
        ticker_moves={"JNJ": 0.025}),
    ScenarioEvent(
        28, 12, "Merger/Acquisition", ["XOM"], 0.25, 6.0,
        ["Exxon Mobil to acquire shale producer Ridgeline Energy in $19 billion all-stock deal",
         "Exxon's Ridgeline takeover extends shale consolidation wave"],
        ["Exxon Mobil agreed to acquire Ridgeline Energy in an all-stock transaction valued at $19 billion, "
         "expanding its Permian Basin footprint. Exxon expects $1.5 billion in annual cost synergies.",
         "The deal is subject to regulatory approval and Ridgeline shareholder vote."],
        ["$XOM buying Ridgeline. Shale consolidation continues",
         "Exxon acquisition makes sense, synergies look real",
         "All-stock deal from Exxon, slight dilution but strategic"],
        ticker_moves={"XOM": -0.008}),
]

# --------------------------------------------------------------------------------------
# Background chatter templates (routine news/posts with mild sentiment)
# --------------------------------------------------------------------------------------
BG_NEWS_POS = [
    "{name} upgraded to buy at a major broker on improving margins",
    "{name} shares rise after analyst raises price target",
    "{name} sees strong demand trends, channel checks suggest",
    "{name} expands buyback program, signals confidence",
    "{name} wins large enterprise contract, shares gain",
]
BG_NEWS_NEG = [
    "{name} downgraded to hold as valuation looks stretched",
    "{name} shares slip after analyst cuts estimates",
    "{name} faces slowing demand in key markets, survey shows",
    "{name} executive departure raises questions about strategy",
    "{name} warns of higher costs weighing on margins",
]
BG_NEWS_NEU = [
    "{name} to present at industry conference next week",
    "{name} schedules quarterly earnings call",
    "{name} shares little changed in quiet trading",
    "{name} files routine annual proxy statement",
]
BG_POST_POS = ["${t} looking strong today 📈", "Added more ${t}, long-term winner",
               "${t} breaking out. Momentum is real", "Love the setup on ${t} here",
               "{name} is executing really well this year ${t}"]
BG_POST_NEG = ["${t} looks weak, cutting my position", "Not liking the price action in ${t} 📉",
               "${t} overvalued imo, waiting for a pullback", "{name} disappointed me again ${t}",
               "Bearish on ${t} into the next print"]
BG_POST_NEU = ["What's everyone's take on ${t}?", "Watching ${t} this week",
               "${t} flat today, nothing to see", "Anyone holding ${t} through earnings?"]


def _ts(day_dt: datetime, hour: float) -> datetime:
    return day_dt + timedelta(hours=hour)


def trading_days() -> list[datetime]:
    days, d = [], START
    while len(days) < N_DAYS:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


BG_MOVES: list[tuple[int, str, float]] = []  # (day, ticker, return shock) from routine news


def build_stream(rng: random.Random, days: list[datetime]):
    news, posts = [], []
    nid = pid = 0
    for ev in EVENTS:
        d = days[ev.day]
        for i, h in enumerate(ev.headlines):
            nid += 1
            body = " ".join(ev.bodies) if i == 0 else rng.choice(ev.bodies)
            t = _ts(d, ev.hour + i * rng.uniform(0.4, 2.5) + rng.uniform(0, 0.3))
            news.append(dict(doc_id=f"N{nid:05d}", source="news_file", outlet=rng.choice(OUTLETS),
                             published_at=t.isoformat(), title=h, body=body,
                             url=f"https://example.com/synthetic/news/N{nid:05d}"))
        for p in ev.posts:
            for rep in range(rng.choice([1, 1, 2])):
                pid += 1
                t = _ts(d, ev.hour + rng.uniform(0.1, 7.0))
                posts.append(dict(post_id=f"S{pid:05d}", source="social_file", platform="X (synthetic)",
                                  author=rng.choice(HANDLES), published_at=t.isoformat(), text=p,
                                  likes=int(rng.lognormvariate(3, 1.2)),
                                  reposts=int(rng.lognormvariate(1.5, 1.1))))
    # background noise: every day, a few routine items for random tickers
    for d in days:
        for _ in range(rng.randint(3, 6)):
            t = rng.choice(TICKERS)
            mood = rng.choices(["pos", "neg", "neu"], weights=[0.4, 0.3, 0.3])[0]
            tmpl = rng.choice({"pos": BG_NEWS_POS, "neg": BG_NEWS_NEG, "neu": BG_NEWS_NEU}[mood])
            nid += 1
            title = tmpl.format(name=NAMES[t])
            BG_MOVES.append((days.index(d), t, {"pos": 0.008, "neg": -0.008, "neu": 0.0}[mood]))
            news.append(dict(doc_id=f"N{nid:05d}", source="news_file", outlet=rng.choice(OUTLETS),
                             published_at=_ts(d, rng.uniform(8, 21)).isoformat(), title=title,
                             body=f"{title}. The move comes amid routine trading; no other material news was reported.",
                             url=f"https://example.com/synthetic/news/N{nid:05d}"))
        for _ in range(rng.randint(8, 14)):
            t = rng.choice(TICKERS)
            mood = rng.choices(["pos", "neg", "neu"], weights=[0.4, 0.3, 0.3])[0]
            tmpl = rng.choice({"pos": BG_POST_POS, "neg": BG_POST_NEG, "neu": BG_POST_NEU}[mood])
            pid += 1
            posts.append(dict(post_id=f"S{pid:05d}", source="social_file", platform="X (synthetic)",
                              author=rng.choice(HANDLES),
                              published_at=_ts(d, rng.uniform(0, 23.9)).isoformat(),
                              text=tmpl.replace("${t}", f"${t}").format(name=NAMES[t]),
                              likes=int(rng.lognormvariate(2, 1.1)), reposts=int(rng.lognormvariate(0.8, 1.0))))
    news.sort(key=lambda r: r["published_at"])
    posts.sort(key=lambda r: r["published_at"])
    return news, posts


def build_prices(days: list[datetime]) -> pd.DataFrame:
    """Daily closes: market factor + sector factor + idiosyncratic noise + scripted event shocks.

    News shocks are partly priced on the event day (70%) and partly over the next two days (30%),
    a stylised under-reaction. Routine analyst headlines move prices by +/-0.8%; social posts do not
    move prices (pure noise). Because the sentiment stream is generated from the same scripted
    events, a sentiment-tilted portfolio has an edge here BY CONSTRUCTION; use real prices
    (python -m src.cli fetch-prices, via yfinance) for an honest backtest.
    """
    np_rng = np.random.default_rng(SEED)
    n = len(days)
    mkt = np_rng.normal(0.0004, 0.008, n)
    sectors = sorted(set(SECTOR.values()))
    sec = {s: np_rng.normal(0, 0.005, n) for s in sectors}
    rets = {t: mkt * np_rng.uniform(0.8, 1.4) + sec[SECTOR[t]] + np_rng.normal(0, 0.011, n) for t in TICKERS}
    for ev in EVENTS:
        for t in TICKERS:
            shock = ev.market_move * (1.3 if SECTOR[t] == "tech" and ev.event_type == "Macroeconomic" else 1.0)
            shock += ev.ticker_moves.get(t, 0.0)
            if shock == 0:
                continue
            day = ev.day if ev.hour < 20 else min(ev.day + 1, n - 1)
            rets[t][day] += 0.7 * shock
            for k in (1, 2):
                if day + k < n:
                    rets[t][day + k] += 0.15 * shock
    for day, t, shock in BG_MOVES:  # routine analyst news moves prices a little (same 70/30 split)
        rets[t][day] += 0.7 * shock
        if day + 1 < n:
            rets[t][day + 1] += 0.3 * shock
    rows = []
    for t in TICKERS:
        p = START_PRICE[t]
        for i, d in enumerate(days):
            p *= math.exp(rets[t][i])
            rows.append(dict(date=d.date().isoformat(), ticker=t, close=round(p, 2)))
    return pd.DataFrame(rows).pivot(index="date", columns="ticker", values="close").reset_index()


# --------------------------------------------------------------------------------------
# Wholesale banking book (Module B)
# --------------------------------------------------------------------------------------
COUNTERPARTIES = [
    # name, sector, region, currency, rating
    ("Harbor Point Financial", "Financials", "North America", "USD", "BB"),
    ("Northwind Energy Corp", "Energy", "North America", "USD", "BBB"),
    ("Gulfstar Petrochemicals", "Energy", "Middle East", "USD", "BBB"),
    ("Aurora Shipping Ltd", "Shipping & Logistics", "Middle East", "USD", "BB"),
    ("Meridian Tankers SA", "Shipping & Logistics", "Europe", "EUR", "BB"),
    ("Kestrel Airlines", "Industrials", "Europe", "EUR", "BB"),
    ("Helios Utilities plc", "Utilities", "Europe", "GBP", "A"),
    ("Banco Austral", "Financials", "Latin America", "BRL", "BB"),
    ("Pinecrest Real Estate Trust", "Real Estate", "North America", "USD", "BBB"),
    ("Summit Commercial Properties", "Real Estate", "North America", "USD", "B"),
    ("Orion Semiconductors", "Technology", "Asia-Pacific", "USD", "A"),
    ("Lotus Electronics Co", "Technology", "Asia-Pacific", "CNY", "BBB"),
    ("Deccan Infrastructure Ltd", "Industrials", "India", "INR", "BBB"),
    ("Ganga Steel & Power", "Materials", "India", "INR", "BB"),
    ("Anatolia Telecom", "Communication Services", "Europe", "TRY", "B"),
    ("Cascade Health Systems", "Health Care", "North America", "USD", "A"),
    ("Evergreen Retail Group", "Consumer", "North America", "USD", "BBB"),
    ("Sakura Auto Parts", "Industrials", "Asia-Pacific", "JPY", "A"),
    ("Republic of Synthetica (sovereign)", "Sovereign", "Latin America", "USD", "BB"),
    ("US Treasury", "Sovereign", "North America", "USD", "AAA"),
    ("Federal Republic of Germany", "Sovereign", "Europe", "EUR", "AAA"),
    ("Granite Bank AG", "Financials", "Europe", "EUR", "A"),
    ("Atlas Mining Corp", "Materials", "Latin America", "USD", "BB"),
    ("Blue Harbor Logistics", "Shipping & Logistics", "Asia-Pacific", "USD", "BBB"),
]
PD_BY_RATING = {"AAA": 0.0002, "AA": 0.0004, "A": 0.0008, "BBB": 0.002, "BB": 0.009, "B": 0.03, "CCC": 0.12}
SPREAD_BY_RATING = {"AAA": 0.0, "AA": 0.004, "A": 0.007, "BBB": 0.012, "BB": 0.025, "B": 0.045, "CCC": 0.09}


def build_trades(rng: random.Random) -> pd.DataFrame:
    rows, k = [], 0
    trade_start = datetime(2025, 1, 6)
    products = [
        ("Loan", "Term Loan", 0.30), ("Loan", "Revolving Credit Facility", 0.12), ("Loan", "Trade Finance", 0.08),
        ("Bond", "Corporate Bond", 0.20), ("Bond", "Sovereign Bond", 0.06),
        ("Derivative", "Interest Rate Swap", 0.09), ("Derivative", "FX Forward", 0.07),
        ("Derivative", "Credit Default Swap", 0.04), ("Equity", "Listed Equity", 0.04),
    ]
    weights = [p[2] for p in products]
    for _ in range(140):
        asset_class, product, _w = rng.choices(products, weights=weights)[0]
        cp = rng.choice(COUNTERPARTIES)
        if product == "Sovereign Bond":
            cp = rng.choice([c for c in COUNTERPARTIES if c[1] == "Sovereign"])
        elif cp[1] == "Sovereign" and asset_class != "Derivative":
            cp = rng.choice([c for c in COUNTERPARTIES if c[1] != "Sovereign"])
        name, sector, region, ccy, rating = cp
        k += 1
        notional = round(rng.choice([5, 10, 15, 20, 25, 40, 50, 75, 100, 150]) * 1e6 * rng.uniform(0.8, 1.2), -4)
        maturity = round(rng.uniform(0.5, 10.0), 2)
        trade_date = trade_start + timedelta(days=rng.randint(0, 560))
        r = dict(trade_id=f"T{k:05d}", trade_date=trade_date.date().isoformat(), counterparty=name,
                 sector=sector, region=region, currency=ccy, rating=rating, asset_class=asset_class,
                 product=product, notional_usd=notional, maturity_years=maturity, direction=1,
                 rate_type="", coupon=0.0, mod_duration=0.0, convexity=0.0, pd=0.0, lgd=0.0, beta=0.0,
                 market_value_usd=0.0)
        base_rate = 0.042
        if asset_class == "Loan":
            r["rate_type"] = rng.choice(["Floating", "Floating", "Fixed"])
            r["coupon"] = round(base_rate + SPREAD_BY_RATING[rating] + rng.uniform(0.003, 0.01), 4)
            r["pd"] = round(PD_BY_RATING[rating] * rng.uniform(0.7, 1.4), 5)
            r["lgd"] = round(rng.uniform(0.30, 0.55) if product != "Trade Finance" else rng.uniform(0.15, 0.3), 3)
            r["mod_duration"] = round(min(maturity, 7) * 0.85, 2) if r["rate_type"] == "Fixed" else 0.25
            if product == "Revolving Credit Facility":
                r["notional_usd"] = round(notional * rng.uniform(0.4, 0.8), -4)  # drawn amount
            r["market_value_usd"] = r["notional_usd"]
        elif asset_class == "Bond":
            y = base_rate + SPREAD_BY_RATING[rating]
            r["rate_type"] = "Fixed"
            r["coupon"] = round(y + rng.uniform(-0.005, 0.005), 4)
            dur = (1 - (1 + y) ** (-maturity)) / y if y > 0 else maturity
            r["mod_duration"] = round(dur / (1 + y), 2)
            r["convexity"] = round(r["mod_duration"] ** 2 * 1.1, 2)
            r["pd"] = PD_BY_RATING[rating]
            r["lgd"] = 0.6 if sector != "Sovereign" else 0.45
            r["market_value_usd"] = round(notional * rng.uniform(0.96, 1.03), -3)
        elif product == "Interest Rate Swap":
            r["direction"] = rng.choice([1, 1, 1, -1])  # +1 = pay fixed (hedges fixed-rate assets; gains when rates rise)
            r["notional_usd"] = round(notional * 2.5, -4)
            r["mod_duration"] = round(min(maturity, 10) * 0.9, 2)
            r["market_value_usd"] = round(notional * rng.uniform(-0.01, 0.01), -3)
            r["rating"] = rating if sector != "Sovereign" else "A"
        elif product == "FX Forward":
            r["currency"] = rng.choice(["EUR", "GBP", "JPY", "INR", "BRL", "TRY", "CNY"])
            r["direction"] = rng.choice([1, -1])  # +1 = long foreign currency vs USD
            r["maturity_years"] = round(rng.uniform(0.1, 1.5), 2)
            r["market_value_usd"] = round(notional * rng.uniform(-0.01, 0.01), -3)
        elif product == "Credit Default Swap":
            r["direction"] = rng.choice([1, 1, -1])  # +1 = protection bought (gains when spreads widen)
            r["mod_duration"] = round(min(maturity, 5) * 0.9, 2)
            r["market_value_usd"] = round(notional * rng.uniform(-0.005, 0.005), -3)
        else:  # Listed equity
            r["beta"] = round(rng.uniform(0.8, 1.5), 2)
            r["notional_usd"] = round(notional * 0.3, -4)
            r["market_value_usd"] = r["notional_usd"]
            r["maturity_years"] = 0.0
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    days = trading_days()
    news, posts = build_stream(rng, days)
    with open(OUT / "news_articles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(news[0].keys()))
        w.writeheader()
        w.writerows(news)
    with open(OUT / "social_posts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(posts[0].keys()))
        w.writeheader()
        w.writerows(posts)
    build_prices(days).to_csv(OUT / "prices_daily.csv", index=False)
    build_trades(rng).to_csv(OUT / "wholesale_trades.csv", index=False)
    gt = [dict(date=days[e.day].date().isoformat(), hour_utc=e.hour, event_type=e.event_type,
               tickers=" ".join(e.tickers) or "MARKET", true_sentiment=e.true_sentiment,
               true_impact=e.true_impact, headline=e.headlines[0]) for e in EVENTS]
    pd.DataFrame(gt).to_csv(OUT / "scenario_ground_truth.csv", index=False)
    print(f"news={len(news)} posts={len(posts)} days={len(days)} events={len(EVENTS)} -> {OUT}")


if __name__ == "__main__":
    main()
