"""Command-line entry point.

    python -m src.cli demo            # replay bundled data through engine + Module A + Module B
    python -m src.cli live            # poll live sources (NewsAPI / GDELT / RSS / Reddit)
    python -m src.cli serve           # REST API on http://127.0.0.1:8000 (docs at /docs)
    python -m src.cli dashboard       # Streamlit dashboard on http://localhost:8501
    python -m src.cli analyze "text"  # score one piece of text
    python -m src.cli evaluate        # evaluate sentiment + event models on labelled data
    python -m src.cli fetch-prices    # download real daily closes with yfinance
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys

from . import config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="risk-engine", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["auto", "finbert", "lexicon"], default=None,
                    help="sentiment backend (default from config/settings.yaml)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="replay bundled data end-to-end")
    lv = sub.add_parser("live", help="poll live sources")
    lv.add_argument("--interval", type=int, default=None)
    lv.add_argument("--iterations", type=int, default=None)
    sv = sub.add_parser("serve", help="start REST API")
    sv.add_argument("--port", type=int, default=8000)
    db = sub.add_parser("dashboard", help="start Streamlit dashboard")
    db.add_argument("--port", type=int, default=8501)
    an = sub.add_parser("analyze", help="analyze a text")
    an.add_argument("text")
    an.add_argument("--social", action="store_true")
    ev = sub.add_parser("evaluate", help="evaluate models")
    ev.add_argument("--phrasebank", action="store_true", help="also run Financial PhraseBank (needs internet)")
    fp = sub.add_parser("fetch-prices", help="download real prices with yfinance")
    fp.add_argument("--start", default=None)
    fp.add_argument("--end", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "demo":
        from .pipeline import Pipeline
        stats = Pipeline(args.backend).replay()
        print(json.dumps(stats, indent=2, default=str))
        print(f"\nOutputs written to {config.path('output_dir')}")
    elif args.cmd == "live":
        from .pipeline import Pipeline
        Pipeline(args.backend, reset=False).live(args.interval, args.iterations)
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("src.engine.api:app", host="127.0.0.1", port=args.port)
    elif args.cmd == "dashboard":
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(config.ROOT / "src" / "dashboard" / "app.py"),
                        "--server.port", str(args.port)], check=False)
    elif args.cmd == "analyze":
        from .engine.engine import RiskEngine
        sigs = RiskEngine(args.backend).analyze_text(args.text, "social" if args.social else "news",
                                                     title="" if args.social else args.text)
        print(json.dumps([s.model_dump(mode="json") for s in sigs], indent=2))
    elif args.cmd == "evaluate":
        from .evaluation import run_evaluation
        run_evaluation(args.backend, args.phrasebank)
    elif args.cmd == "fetch-prices":
        from .market_data import fetch_prices
        fetch_prices(args.start, args.end)


if __name__ == "__main__":
    main()
