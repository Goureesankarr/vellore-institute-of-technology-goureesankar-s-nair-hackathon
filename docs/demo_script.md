# Demo walkthrough script (about 5 minutes, fits both the live demo and the video)

**Setup before recording:** run `python -m src.cli demo` once, then start `python -m src.cli dashboard`.
In a second terminal, start `python -m src.cli serve`. Use a browser zoom of about 90%.

| Time | Show | Say |
|---|---|---|
| 0:00–0:30 | Title slide, then the dashboard header | "SENTINEL turns news and social media into structured risk signals (sentiment, event type, impact) and uses them to rebalance an index and stress-test a banking book." |
| 0:30–1:00 | Terminal: `pip install -r requirements.txt`, then `python -m src.cli demo` | "One command replays 625 news articles and posts through the engine: 641 signals in a few seconds, written to SQLite, JSONL and the API, with five stress tests triggered." |
| 1:00–1:45 | **Risk signals** tab. Drag the replay clock back to 21 Aug, then press ▶ Play | "Each dot is a signal: height is impact, colour is sentiment. Watch the 24 August Hormuz attacks jump above the stress threshold, then the 31 August credit event." Open the table and filter to `Credit Event`. |
| 1:45–2:15 | Sidebar **Try the engine**: paste "Fed unexpectedly raises rates by 75 basis points as inflation surges; bank stocks tumble" | "Any text, scored live: market-wide Macroeconomic, sentiment −0.95, impact 7.8, above the stress threshold. The driver breakdown explains why." |
| 2:15–3:00 | **Module A** tab | "The index subscribes to sentiment. Positive names are tilted up and negative names down, within bounds, with smoothing to cap turnover. JPM drops from 7% to 4.4% at the close on the day of the default headline. The heatmap shows every rebalance." |
| 3:00–4:00 | **Module B** tab. Select the Harbor Point stress test | "An adverse credit event with impact 10 fires the stress test automatically. Harbor Point is named in the news, so it is marked jump-to-default; financials and real estate are amplified. The book loses $503m and CET1 falls from 13.5% to 2.5%, a breach." Switch to **What-if**: Geopolitical, impact 9, Middle East + Energy. |
| 4:00–4:30 | Browser: `http://127.0.0.1:8000/docs`, run `GET /signals?min_impact=7` | "Any downstream app can consume the same signals over REST." |
| 4:30–5:00 | Results slide | "On a held-out set, the offline engine beats naive baselines (58% vs 50% on sentiment, 60% vs 17% on events). FinBERT plugs in through configuration. Everything is in the public repo." |

**Likely jury questions**
- *Why rules plus a lexicon instead of only FinBERT?* Explainability and zero-dependency operation; FinBERT is the default whenever it is installed (`--backend finbert`).
- *How is look-ahead avoided?* The replay processes documents in timestamp order. Weights set at day *d*'s close earn the *d→d+1* return.
- *Why don't positive high-impact events trigger stress tests?* `require_adverse: true`. A ceasefire is high-impact but not a downside scenario.
- *How are shocks scaled?* Shocks are linear in impact around the reference level of 8, capped at 1.25×. Sector and region amplifiers apply when the news names them.
- *What would go to production?* Kafka for the bus, Postgres for the store, a fine-tuned FinBERT, embedding-based story deduplication, and full revaluation with correlated shocks.
