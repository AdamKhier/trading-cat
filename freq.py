"""Frequency-boosted variants of the validated strategy, judged fully OOS.

Fixes the converged trading rule (rr=4, bracket=0.3) and asks whether a
shorter structure window produces more signals per year WITHOUT losing the
out-of-sample edge (rolling walk-forward, same protocol as walkforward.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as bt
import categorical as ct
from run import load

STABLE = 1e9
RR, BRACKET = 4.0, 0.3
EFF = 0.25


def long_only(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.loc[d["cat"].isin(["TREND_DOWN", "RANGE_S"]), "cat"] = "NEUTRAL"
    return d


def oos_walk(det: pd.DataFrame, direction: str) -> pd.DataFrame:
    trades = []
    t_start = pd.Timestamp("2014-01-01")
    step = pd.Timedelta(days=365 * 2)
    last = det.index[-1]
    while t_start < last:
        t_run_end = min(t_start + step, last + pd.Timedelta(days=1))
        d = det if direction == "both" else long_only(det)
        res = bt.run(d, params={"rr": RR, "bracket": BRACKET}, active=None)
        tr = res.trade_frame[(res.trade_frame.entry_ts >= t_start) &
                             (res.trade_frame.entry_ts < t_run_end)]
        trades.append(tr)
        t_start = t_run_end
    return pd.concat(trades, ignore_index=True)


def main():
    rows = []
    for nw in (8, 10, 12, 15, 20):
        det = ct.detect(load("1d"),
                        {"n_win": nw, "eff_range": EFF, "stable_vol_max": STABLE})
        for ddir in ("long", "both"):
            t = oos_walk(det, ddir)
            if t.empty:
                continue
            days = t["entry_ts"].nunique()
            pf = (t["r"].clip(lower=0).sum() / -t["r"].clip(upper=0).sum()) \
                if (t["r"] < 0).any() else float("inf")
            rows.append({
                "n_win": nw, "dir": ddir, "n": len(t),
                "trades/yr": len(t) / t["entry_ts"].dt.year.nunique(),
                "pf": pf, "sumR": t["r"].sum(),
            })
    res = pd.DataFrame(rows).sort_values("n_win")
    print(res.round(2).to_string(index=False))
    print("\n  want: high trades/yr AND pf >= ~1.3")


if __name__ == "__main__":
    main()