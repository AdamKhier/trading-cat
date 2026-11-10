import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

# interval -> (period, max_age_minutes) ; after this age we refetch
INTERVALS = {
    "1m":  ("5d",   30),
    "5m":  ("60d",  180),
    "15m": ("60d",  720),
    "30m": ("60d",  1440),
    "1h":  ("60d",  1440),
    "1d":  ("20y",  None),  # never refetch
}

DROP = ["Dividends", "Stock Splits", "Capital Gains"]


def fetch(ticker: str, interval: str, force: bool = False) -> pd.DataFrame:
    out = DATA / f"{ticker.replace('=', '_')}_{interval}.csv"
    period, max_age = INTERVALS[interval]
    if not force and out.exists() and max_age is not None:
        age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(out.stat().st_mtime, unit="s", tz="UTC")
        if age.total_seconds() / 60 < max_age:
            return pd.read_csv(out, index_col=0, parse_dates=True)
    if not force and out.exists() and max_age is None:
        return pd.read_csv(out, index_col=0, parse_dates=True)

    flt = yf.Ticker(ticker)
    hist = flt.history(period=period, interval=interval, auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"no data returned for {ticker} @ {interval}")
    hist = hist.drop(columns=[c for c in DROP if c in hist.columns])
    hist.index = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index

    # yfinance 1m comes out tz-naive already; keep naive UTC
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    hist = hist.dropna(subset=["Open", "High", "Low", "Close"])
    hist = hist[~hist.index.duplicated(keep="last")].sort_index()
    hist.to_csv(out)
    print(f"  cached {out.name}: {len(hist)} rows  {hist.index[0]} -> {hist.index[-1]}")
    return hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="ES=F")
    ap.add_argument("--interval", default="all",
                    help="1m|5m|15m|30m|1h|1d|all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    intervals = list(INTERVALS) if args.interval == "all" else [args.interval]
    for iv in intervals:
        if iv not in INTERVALS:
            sys.exit(f"unknown interval {iv}")
        fetch(args.ticker, iv, force=args.force)


if __name__ == "__main__":
    main()