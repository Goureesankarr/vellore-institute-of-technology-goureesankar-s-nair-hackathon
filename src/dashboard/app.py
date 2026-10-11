"""Streamlit dashboard: live risk-signal feed, Module A (index rebalancer), Module B (stress test).

Run:  python -m src.cli dashboard     (or: streamlit run src/dashboard/app.py)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.engine.store import SignalStore  # noqa: E402

st.set_page_config(page_title="AI/NLP Risk Engine", page_icon="📡", layout="wide")

# ---- palette (validated reference palette; see docs) ------------------------------------
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEG, NEU, POS = "#d03b3b", "#b9b8b0", "#2a78d6"
DIVERGING = [[0.0, "#b8302f"], [0.25, "#e98b89"], [0.5, "#ecebe7"], [0.75, "#86b6ef"], [1.0, "#1c5cab"]]
GRID, AXIS, MUTED = "rgba(137,135,129,0.25)", "rgba(137,135,129,0.6)", "#898781"
CRITICAL, GOOD = "#d03b3b", "#0ca30c"
OUT = config.path("output_dir")

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
  div[data-testid="stMetricValue"] {font-size: 1.55rem;}
  .small-note {color: #898781; font-size: 0.82rem;}
  .tag {display:inline-block;padding:1px 8px;border-radius:10px;font-size:0.78rem;border:1px solid rgba(137,135,129,.45);margin-right:4px}
</style>""", unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 340, legend: bool = True, ytitle: str = "", xtitle: str = "") -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=legend, hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12))
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont_color=MUTED, title_text=xtitle, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, linecolor=AXIS, tickfont_color=MUTED, title_text=ytitle,
                     zeroline=False)
    return fig


# ---- data loading -----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_signals(mtime_key: float) -> pd.DataFrame:
    df = SignalStore(config.path("signals_db")).query()
    return df.sort_values("timestamp") if not df.empty else df


@st.cache_data(show_spinner=False)
def load_csv(name: str, mtime_key: float) -> pd.DataFrame:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json(name: str, mtime_key: float):
    p = OUT / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def mtime(name: str) -> float:
    p = OUT / name
    return p.stat().st_mtime if p.exists() else 0.0


def run_demo() -> dict:
    from src.pipeline import Pipeline
    return Pipeline().replay()


if not config.path("signals_db").exists() or not (OUT / "weights_history.csv").exists():
    with st.spinner("First run: processing the bundled news + social stream through the engine…"):
        run_demo()

signals = load_signals(mtime("signals.jsonl"))
weights = load_csv("weights_history.csv", mtime("weights_history.csv"))
perf = load_csv("index_performance.csv", mtime("index_performance.csv"))
stress = load_json("stress_tests.json", mtime("stress_tests.json")) or []
stats = load_json("run_stats.json", mtime("run_stats.json")) or {}
evaluation = load_json("evaluation.json", mtime("evaluation.json"))
if not weights.empty:
    weights["timestamp"] = pd.to_datetime(weights["timestamp"], utc=True)

# ---- sidebar: replay clock ----------------------------------------------------------------
ticks = sorted(weights["timestamp"].unique()) if not weights.empty else []
st.sidebar.title("📡 Risk Engine")
st.sidebar.caption(f"Mode: **{stats.get('mode', '?')}** · model: `{stats.get('sentiment_model', '?')}`")
if "cursor" not in st.session_state:
    st.session_state.cursor = len(ticks) - 1
if "playing" not in st.session_state:
    st.session_state.playing = False

st.sidebar.subheader("Replay clock")
st.sidebar.caption("Move through the stream as if it were arriving live. Everything on the page is "
                   "computed only from signals published before this time.")
if ticks:
    st.session_state.cursor = st.sidebar.select_slider(
        "As of", options=list(range(len(ticks))), value=min(st.session_state.cursor, len(ticks) - 1),
        format_func=lambda i: pd.Timestamp(ticks[i]).strftime("%a %d %b %H:%M UTC"))
c1, c2 = st.sidebar.columns(2)
if c1.button("▶ Play" if not st.session_state.playing else "⏸ Pause", width="stretch"):
    st.session_state.playing = not st.session_state.playing
    if st.session_state.playing and st.session_state.cursor >= len(ticks) - 1:
        st.session_state.cursor = 0
    st.rerun()
if c2.button("⟲ Re-run", width="stretch", help="Re-process all data through the engine"):
    with st.spinner("Re-running pipeline…"):
        run_demo()
    st.cache_data.clear()
    st.rerun()
speed = st.sidebar.slider("Playback speed (s / step)", 0.3, 3.0, 1.0, 0.1)

as_of = pd.Timestamp(ticks[st.session_state.cursor]) if ticks else pd.Timestamp.now(tz="UTC")
sig_now = signals[signals["timestamp"] <= as_of] if not signals.empty else signals
w_now = weights[weights["timestamp"] <= as_of]
stress_now = [s for s in stress if pd.Timestamp(s["triggered_at"]) <= as_of]

st.sidebar.divider()
st.sidebar.subheader("Try the engine")
txt = st.sidebar.text_area("Paste a headline or post", "Fed unexpectedly raises rates by 75 basis points as "
                           "inflation surges; bank stocks tumble", height=90)
is_social = st.sidebar.toggle("Social post", value=False)
if st.sidebar.button("Analyze", width="stretch"):
    from src.engine.engine import RiskEngine

    @st.cache_resource(show_spinner="Loading models…")
    def _engine():
        return RiskEngine()

    res = _engine().analyze_text(txt, "social" if is_social else "news", "" if is_social else txt)
    if not res:
        st.sidebar.info("No actionable entity or event detected.")
    for s in res:
        st.sidebar.markdown(f"**{s.entity}** · {s.event_type} ({s.event_confidence:.0%})  \n"
                            f"Sentiment **{s.sentiment_score:+.2f}** ({s.sentiment_label}) · Impact **{s.impact_score:.1f}**/10")
        st.sidebar.caption("Impact drivers: " + ", ".join(f"{k} {v:+.2f}" for k, v in s.drivers.items()))

# ---- header -------------------------------------------------------------------------------
st.title("AI/NLP Risk Engine")
st.caption(f"Unstructured news + social text → structured risk signals → tactical index rebalancing and "
           f"event-driven stress testing.  As of **{as_of.strftime('%d %b %Y %H:%M UTC')}**")

tab1, tab2, tab3, tab4 = st.tabs(["Risk signals", "Module A · Index rebalancer",
                                  "Module B · Stress test", "Engine & API"])

# =============================================================================================
# TAB 1: signal feed
# =============================================================================================
with tab1:
    day_ago = as_of - pd.Timedelta(hours=24)
    last24 = sig_now[sig_now["timestamp"] > day_ago]
    k = st.columns(5)
    k[0].metric("Documents processed", f"{sig_now['doc_id'].nunique():,}")
    k[1].metric("Signals emitted", f"{len(sig_now):,}", f"{len(last24)} in last 24h")
    k[2].metric("Sources", f"{sig_now['source'].nunique()}", help=", ".join(sorted(sig_now['source'].unique())) if len(sig_now) else "")
    k[3].metric("Avg sentiment (24h)", f"{last24['sentiment_score'].mean():+.2f}" if len(last24) else "–")
    hi = sig_now[sig_now["impact_score"] > 7]
    k[4].metric("High-impact signals (>7)", f"{len(hi)}", f"{(last24['impact_score'] > 7).sum()} in last 24h")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Impact timeline")
        st.caption("Each dot is a signal. Height = predicted impact; colour = sentiment (red negative, blue positive).")
        plot = sig_now[sig_now["event_type"] != "Other"]
        fig = go.Figure(go.Scatter(
            x=plot["timestamp"], y=plot["impact_score"], mode="markers",
            marker=dict(size=9, color=plot["sentiment_score"], colorscale=DIVERGING, cmin=-1, cmax=1,
                        line=dict(width=1, color="rgba(252,252,251,0.9)"),
                        colorbar=dict(title="Sentiment", thickness=10, len=0.8)),
            customdata=plot[["entity", "event_type", "headline", "sentiment_score", "source_type"]],
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}<br>"
                          "Impact %{y:.1f} · sentiment %{customdata[3]:+.2f} · %{customdata[4]}<extra></extra>"))
        fig.add_hline(y=config.settings()["module_b"]["impact_threshold"], line_dash="dot", line_color=CRITICAL,
                      annotation_text="stress-test threshold", annotation_font_color=MUTED)
        st.plotly_chart(style(fig, 360, legend=False, ytitle="Impact score (1–10)"), width="stretch")
    with right:
        st.subheader("Event mix")
        st.caption("Signals by event classification (excl. Other)")
        ev = sig_now[sig_now["event_type"] != "Other"].groupby("event_type").agg(
            n=("signal_id", "count"), impact=("impact_score", "mean")).sort_values("n")
        fig = go.Figure(go.Bar(x=ev["n"], y=ev.index, orientation="h", marker_color=CAT[0],
                               customdata=ev["impact"], text=ev["n"], textposition="outside",
                               hovertemplate="%{y}: %{x} signals · avg impact %{customdata:.1f}<extra></extra>"))
        fig.update_traces(marker_line_width=0, cliponaxis=False)
        fig = style(fig, 360, legend=False)
        fig.update_layout(margin=dict(r=40))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Sentiment heatmap · per stock per day")
    st.caption("Daily mean entity sentiment from news + social (credibility-weighted). Blank = no signals that day.")
    stocks = sig_now[sig_now["entity"] != "MARKET"].copy()
    if not stocks.empty:
        stocks["day"] = stocks["timestamp"].dt.strftime("%b %d")
        stocks["w"] = stocks["credibility"]
        stocks["ws"] = stocks["w"] * stocks["sentiment_score"]
        hm = stocks.groupby(["entity", "day"]).agg(ws=("ws", "sum"), w=("w", "sum"), n=("signal_id", "count"))
        hm["s"] = hm["ws"] / hm["w"]
        z = hm["s"].unstack("day")
        cnt = hm["n"].unstack("day")
        fig = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index, colorscale=DIVERGING, zmin=-1, zmax=1,
                                   xgap=2, ygap=2, customdata=cnt.values,
                                   hovertemplate="%{y} · %{x}<br>sentiment %{z:+.2f} (%{customdata} signals)<extra></extra>",
                                   colorbar=dict(thickness=10, title="Sentiment")))
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=sorted(z.columns, key=lambda d: pd.to_datetime(d + " 2026")))
        st.plotly_chart(style(fig, 420, legend=False), width="stretch")

    st.subheader("Latest signals")
    f1, f2, f3 = st.columns([2, 2, 1])
    ents = f1.multiselect("Entity", sorted(sig_now["entity"].unique()))
    evs = f2.multiselect("Event type", sorted(sig_now["event_type"].unique()))
    minimp = f3.number_input("Min impact", 1.0, 10.0, 1.0, 0.5)
    t = sig_now[sig_now["impact_score"] >= minimp]
    if ents:
        t = t[t["entity"].isin(ents)]
    if evs:
        t = t[t["event_type"].isin(evs)]
    show = t.sort_values("timestamp", ascending=False).head(300)[
        ["timestamp", "source_type", "entity", "headline", "sentiment_score", "event_type", "event_confidence",
         "impact_score", "regions", "sectors"]]
    st.dataframe(show, width="stretch", hide_index=True, height=360, column_config={
        "timestamp": st.column_config.DatetimeColumn("Time (UTC)", format="DD MMM HH:mm"),
        "sentiment_score": st.column_config.NumberColumn("Sentiment", format="%+.2f"),
        "impact_score": st.column_config.ProgressColumn("Impact", min_value=1, max_value=10, format="%.1f"),
        "event_confidence": st.column_config.NumberColumn("Conf.", format="%.2f"),
        "headline": st.column_config.TextColumn("Headline / post", width="large")})

# =============================================================================================
# TAB 2: Module A
# =============================================================================================
with tab2:
    ma = config.settings()["module_a"]
    st.caption(f"{config.universe()['index_name']} · 15 S&P 100 constituents · rebalanced every "
               f"{ma['rebalance_frequency']} on decayed sentiment (half-life {ma['sentiment_half_life_hours']}h). "
               f"Target = base × exp({ma['tilt_strength']} × sentiment), bounded to "
               f"[{ma['min_multiple_of_base']}×, {ma['max_multiple_of_base']}×] base, ≤ {ma['max_weight']:.0%}, "
               f"smoothed {ma['smoothing']:.0%} toward target.")
    if w_now.empty:
        st.info("No rebalance yet at this point of the replay.")
    else:
        latest = w_now[w_now["timestamp"] == w_now["timestamp"].max()].copy()
        latest["active"] = latest["weight"] - latest["base_weight"]
        p_now = perf.copy()
        if not p_now.empty:
            p_now["date"] = pd.to_datetime(p_now["date"])
            p_now = p_now[p_now["date"] <= as_of.tz_localize(None).normalize()]
        k = st.columns(4)
        if not p_now.empty:
            iv, bv = p_now["index_value"].iloc[-1], p_now["benchmark_value"].iloc[-1]
            k[0].metric("Sentiment index", f"{iv:.2f}", f"{iv - 100:+.2f}% since start")
            k[1].metric("Benchmark (static weights)", f"{bv:.2f}", f"{bv - 100:+.2f}%")
            k[2].metric("Active return", f"{iv - bv:+.2f} pts")
        k[3].metric("Turnover at last rebalance", f"{latest['turnover'].iloc[0]:.1%}",
                    f"{w_now['timestamp'].nunique()} rebalances")

        a, b = st.columns([3, 2])
        with a:
            st.subheader("Weights over time")
            top = (w_now.assign(dev=(w_now["weight"] - w_now["base_weight"]).abs())
                   .groupby("ticker")["dev"].max().sort_values(ascending=False).index[:4].tolist())
            pick = st.multiselect("Stocks (up to 4)", sorted(w_now["ticker"].unique()), default=top, max_selections=4)
            fig = go.Figure()
            for i, tk in enumerate(pick):
                d = w_now[w_now["ticker"] == tk]
                fig.add_trace(go.Scatter(x=d["timestamp"], y=d["weight"] * 100, name=tk, mode="lines+markers",
                                         line=dict(color=CAT[i], width=2), marker=dict(size=6),
                                         customdata=d[["sentiment", "base_weight"]].values,
                                         hovertemplate=f"<b>{tk}</b> %{{y:.2f}}% (base %{{customdata[1]:.1%}})"
                                                       "<br>sentiment %{customdata[0]:+.2f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=d["timestamp"], y=d["base_weight"] * 100, mode="lines", showlegend=False,
                                         line=dict(color=CAT[i], width=1, dash="dot"), hoverinfo="skip"))
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(style(fig, 380, ytitle="Index weight (%) · dotted = benchmark"), width="stretch")
        with b:
            st.subheader("Current tilt vs benchmark")
            l2 = latest.sort_values("active")
            fig = go.Figure(go.Bar(x=l2["active"] * 100, y=l2["ticker"], orientation="h",
                                   marker_color=[NEG if v < 0 else POS for v in l2["active"]],
                                   customdata=l2[["weight", "base_weight", "sentiment", "fresh_signals"]].values,
                                   hovertemplate="<b>%{y}</b> %{customdata[0]:.2%} vs base %{customdata[1]:.2%}"
                                                 "<br>sentiment %{customdata[2]:+.2f} · %{customdata[3]} fresh signals"
                                                 "<extra></extra>"))
            fig.add_vline(x=0, line_color=AXIS)
            st.plotly_chart(style(fig, 430, legend=False, xtitle="Active weight (pp): over / under-weight"),
                            width="stretch")

        st.subheader("Active weight map")
        st.caption("Over-weight (blue) / under-weight (red) vs benchmark at every rebalance, all 15 stocks.")
        piv = w_now.assign(active=(w_now["weight"] - w_now["base_weight"]) * 100,
                           day=w_now["timestamp"].dt.strftime("%m-%d %a")).pivot(index="ticker", columns="day",
                                                                              values="active")
        lim = max(0.5, float(piv.abs().max().max()))
        fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale=DIVERGING, zmin=-lim, zmax=lim,
                                   xgap=2, ygap=2, colorbar=dict(thickness=10, title="pp"),
                                   hovertemplate="%{y} · %{x}<br>active weight %{z:+.2f} pp<extra></extra>"))
        fig.update_xaxes(type="category")
        st.plotly_chart(style(fig, 420, legend=False), width="stretch")

        if not p_now.empty:
            st.subheader("Index level vs benchmark")
            src = "real closes (yfinance)" if stats.get("mode") == "live" else \
                "synthetic demo prices generated with the stream — illustrative only (see README)"
            st.caption(f"Rebased to 100. Weights set at each day's close earn the next day's return. Prices: {src}.")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=p_now["date"], y=p_now["index_value"], name="Sentiment-tilted index",
                                     line=dict(color=CAT[0], width=2)))
            fig.add_trace(go.Scatter(x=p_now["date"], y=p_now["benchmark_value"], name="Static benchmark",
                                     line=dict(color=CAT[1], width=2)))
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(style(fig, 320, ytitle="Index level"), width="stretch")

# =============================================================================================
# TAB 3: Module B
# =============================================================================================
with tab3:
    mb = config.settings()["module_b"]
    st.caption(f"Synthetic wholesale-banking book (loans, bonds, derivatives, equity). A stress test fires when "
               f"an adverse signal of type {', '.join(mb['trigger_event_types'])} has Impact Score > "
               f"{mb['impact_threshold']}. Shocks scale with impact; sectors/regions named in the news are amplified.")
    from src.modules.stress_test import StressTester, load_portfolio

    @st.cache_resource
    def _tester():
        return StressTester(load_portfolio())

    tester = _tester()
    mode = st.radio("Scenario source", ["Triggered by the engine", "What-if (manual)"], horizontal=True)
    res = None
    if mode == "Triggered by the engine":
        if not stress_now:
            st.info("No stress test has been triggered yet at this point of the replay. Move the clock forward "
                    "or press ▶ Play.")
        else:
            labels = [f"{pd.Timestamp(s['triggered_at']).strftime('%d %b %H:%M')} · {s['event_type']} · impact "
                      f"{s['impact_score']:.1f} · {s['headline'][:70]}" for s in stress_now]
            i = st.selectbox("Triggered stress tests", range(len(stress_now)), index=len(stress_now) - 1,
                             format_func=lambda j: labels[j])
            res = stress_now[i]
    else:
        w1, w2, w3 = st.columns(3)
        et = w1.selectbox("Event type", list(tester.scenarios.keys()))
        imp = w1.slider("Impact score", 1.0, 10.0, 8.5, 0.1)
        secs = w2.multiselect("Sectors named in news", sorted(tester.book["sector"].unique()), ["Energy"])
        regs = w2.multiselect("Regions named in news", sorted(tester.book["region"].unique()), ["Middle East"])
        dft = w3.multiselect("Counterparty defaults", sorted(tester.book["counterparty"].unique()))
        res = tester.run(et, imp, secs, regs, dft)

    if res:
        st.markdown(f"**{res['event_type']}** · impact **{res['impact_score']:.1f}** (shock scale "
                    f"{res['scale']:.2f}×) — _{res['headline']}_")
        st.markdown(" ".join(f"<span class='tag'>{x}</span>" for x in
                             res["regions_hit"] + res["sectors_hit"] +
                             [f"default: {c}" for c in res["defaulted_counterparties"]]), unsafe_allow_html=True)
        sc = res["scenario"]
        st.caption(f"Scenario: {sc['description']}. Applied: equities {sc['equity_pct']:+.1f}%, rates "
                   f"{sc['rates_bp']:+.0f}bp, BBB spreads +{sc['spread_bp']['BBB']:.0f}bp, HY (B) spreads "
                   f"+{sc['spread_bp']['B']:.0f}bp, PD × {sc['pd_multiplier']:.2f}.")
        k = st.columns(4)
        k[0].metric("Portfolio value before", f"${res['value_before_usd'] / 1e9:,.3f}bn")
        k[1].metric("After stress", f"${res['value_after_usd'] / 1e9:,.3f}bn",
                    f"{res['pnl_usd'] / 1e6:,.0f}m ({res['pnl_pct_of_book']:+.2f}%)", delta_color="normal")
        k[2].metric("CET1 ratio", f"{res['cet1_ratio_after_pct']:.2f}%",
                    f"{res['cet1_ratio_after_pct'] - res['cet1_ratio_before_pct']:+.2f} pp from "
                    f"{res['cet1_ratio_before_pct']:.2f}%")
        breach = res["cet1_ratio_after_pct"] < 7.0
        k[3].metric("Capital vs 7.0% threshold", "⚠ Breach" if breach else "✓ Adequate",
                    help="7.0% = 4.5% CET1 minimum + 2.5% capital conservation buffer (Basel III)")

        a, b = st.columns(2)
        with a:
            st.subheader("Value before vs after, by asset class")
            g = pd.DataFrame(res["by_asset_class"])
            g = g[g["asset_class"] != "Derivative"].sort_values("value_before", ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=g["asset_class"], y=g["value_before"] / 1e6, name="Before", marker_color=CAT[0]))
            fig.add_trace(go.Bar(x=g["asset_class"], y=g["value_after"] / 1e6, name="After stress", marker_color=CAT[1]))
            fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
            fig.update_traces(hovertemplate="%{x}: $%{y:,.0f}m<extra>%{fullData.name}</extra>")
            st.plotly_chart(style(fig, 340, ytitle="USD m"), width="stretch")
        with b:
            st.subheader("P&L attribution by product")
            g = pd.DataFrame(res["by_product"]).sort_values("pnl")
            fig = go.Figure(go.Bar(x=g["pnl"] / 1e6, y=g["product"], orientation="h",
                                   marker_color=[NEG if v < 0 else POS for v in g["pnl"]],
                                   customdata=g["positions"],
                                   hovertemplate="%{y}: %{x:,.1f}m (%{customdata} positions)<extra></extra>"))
            fig.add_vline(x=0, line_color=AXIS)
            st.plotly_chart(style(fig, 340, legend=False, xtitle="Stress P&L (USD m)"), width="stretch")

        a, b = st.columns(2)
        with a:
            st.subheader("Loss by sector")
            g = pd.DataFrame(res["by_sector"]).sort_values("pnl", ascending=False)
            fig = go.Figure(go.Bar(x=g["pnl"] / 1e6, y=g["sector"], orientation="h",
                                   marker_color=[NEG if (v < 0 and s in res["sectors_hit"]) else
                                                 ("#e98b89" if v < 0 else POS) for v, s in zip(g["pnl"], g["sector"])],
                                   hovertemplate="%{y}: %{x:,.1f}m<extra></extra>"))
            st.plotly_chart(style(fig, 380, legend=False, xtitle="USD m · darker red = sector named in the news"),
                            width="stretch")
        with b:
            st.subheader("Loss by region")
            g = pd.DataFrame(res["by_region"]).sort_values("pnl", ascending=False)
            fig = go.Figure(go.Bar(x=g["pnl"] / 1e6, y=g["region"], orientation="h",
                                   marker_color=[NEG if (v < 0 and r in res["regions_hit"]) else
                                                 ("#e98b89" if v < 0 else POS) for v, r in zip(g["pnl"], g["region"])],
                                   hovertemplate="%{y}: %{x:,.1f}m<extra></extra>"))
            st.plotly_chart(style(fig, 380, legend=False, xtitle="USD m · darker red = region named in the news"),
                            width="stretch")

        st.subheader("Largest position losses")
        tl = pd.DataFrame(res["top_losses"])
        st.dataframe(tl, width="stretch", hide_index=True, column_config={
            "market_value_usd": st.column_config.NumberColumn("Value (USD)", format="%,.0f"),
            "stress_pnl_usd": st.column_config.NumberColumn("Stress P&L (USD)", format="%,.0f"),
            "jump_to_default": st.column_config.CheckboxColumn("Default")})

    if stress_now:
        st.subheader("Stress-test timeline")
        tl = pd.DataFrame([{"time": pd.Timestamp(s["triggered_at"]), "event": s["event_type"],
                            "impact": s["impact_score"], "pnl": s["pnl_usd"] / 1e6,
                            "cet1": s["cet1_ratio_after_pct"], "headline": s["headline"]} for s in stress_now])
        fig = go.Figure(go.Bar(x=tl["time"], y=tl["pnl"], marker_color=NEG, width=1000 * 3600 * 10,
                               customdata=tl[["event", "impact", "cet1", "headline"]].values,
                               hovertemplate="%{customdata[0]} · impact %{customdata[1]:.1f}<br>%{customdata[3]}"
                                             "<br>P&L %{y:,.0f}m · CET1 after %{customdata[2]:.2f}%<extra></extra>"))
        st.plotly_chart(style(fig, 260, legend=False, ytitle="Stress P&L (USD m)"), width="stretch")

# =============================================================================================
# TAB 4: Engine
# =============================================================================================
with tab4:
    a, b = st.columns(2)
    with a:
        st.subheader("Run statistics")
        st.json({k: v for k, v in stats.items() if k not in ("module_a",)}, expanded=False)
        st.subheader("Model evaluation")
        if evaluation:
            for name, m in evaluation.get("sentiment", {}).items():
                st.markdown(f"**Sentiment · {name}** — accuracy **{m['accuracy']:.1%}**, macro-F1 "
                            f"**{m['macro_f1']:.3f}** on {m['n']} held-out items")
            for name, m in evaluation.get("events", {}).items():
                st.markdown(f"**Event classifier · {name}** — accuracy **{m['accuracy']:.1%}**, macro-F1 "
                            f"**{m['macro_f1']:.3f}** on {m['n']} items")
            if "module_a_ablation" in evaluation:
                st.markdown("**Module A ablation** (same data)")
                st.dataframe(pd.DataFrame(evaluation["module_a_ablation"]), hide_index=True, width="stretch")
        else:
            st.caption("Run `python -m src.cli evaluate` to populate.")
    with b:
        st.subheader("Signal schema (example)")
        if len(sig_now):
            ex = sig_now.sort_values("impact_score").iloc[-1].to_dict()
            ex["timestamp"] = str(ex["timestamp"])
            st.json(ex, expanded=True)
        st.subheader("REST API")
        st.code("python -m src.cli serve   # http://127.0.0.1:8000/docs\n\n"
                "GET  /signals?entity=JPM&min_impact=7\nGET  /signals/high-impact?threshold=7\n"
                "GET  /sentiment                 # decayed sentiment per entity\nPOST /analyze  {\"text\": \"...\"}\n"
                "GET  /modules/index-weights\nGET  /modules/stress-tests", language="bash")

# ---- playback loop ---------------------------------------------------------------------------
if st.session_state.playing and ticks:
    time.sleep(speed)
    if st.session_state.cursor < len(ticks) - 1:
        st.session_state.cursor += 1
    else:
        st.session_state.playing = False
    st.rerun()
