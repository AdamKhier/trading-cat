"""Portfolio adaptation for a 30-day pass: same validated strategy, run on
multiple markets, equalised per-trade risk, one 30-day prop simulation.

Each market gets its own rolling walk-forward OOS pass with the converged
rule (rr=4, bracket=0.3, both sides, n_win=20). Risk is equalised so that
every market's median per-trade risk maps to a common 'unit'. Daily PnL
streams are summed into a portfolio, then a 30-day account simulation runs
with the portfolio scaled by k.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as bt
import categorical as ct
import config as C
from fetch import fetch
from walkforward import long_only

MARKETS = {
    "ES=F": {"pv": 50},
    "NQ=F": {"pv": 20},
    "YM=F": {"pv": 5},
    "CL=F": {"pv": 1000},
    "GC=F": {"pv": 100},
}
RR, BRACKET = 4.0, 0.3
TARGET_RISK = 150.0          # each market's median risk scaled to this $/unit
COST = {"ES=F": 30.0, "NQ=F": 25.0, "YM=F": 20.0, "CL=F": 22.0, "GC=F": 22.0}
DEF_DD = 5_000
DEF_TARGET = 3_000
CUTOFFS = [10, 20, 30, 60]


def oos_trades(symbol: str) -> pd.DataFrame:
    df = ct.detect(fetch(symbol, "1d"),
                   {"n_win": 20, "eff_range": 0.25, "stable_vol_max": 1e9})
    trades = []
    t_start = pd.Timestamp("2014-01-01")
    step = pd.Timedelta(days=365 * 2)
    last = df.index[-1]
    while t_start < last:
        t_run_end = min(t_start + step, last + pd.Timedelta(days=1))
        res = bt.run(long_only(df), params={"rr": RR, "bracket": BRACKET},
                     active=None)
        tr = res.trade_frame[(res.trade_frame.entry_ts >= t_start) &
                             (res.trade_frame.entry_ts < t_run_end)]
        trades.append(tr)
        t_start = t_run_end
    return pd.concat(trades, ignore_index=True)


def main():
    streams, report = {}, []
    for sym, meta in MARKETS.items():
        t = oos_trades(sym)
        risk = (t["entry"] - t["sl"]).abs() * meta["pv"]
        t["pnl"] = t["r"] * risk - COST[sym]
        med = risk.median()
        s = TARGET_RISK / med if t.empty is False else 0.0
        t["pnl_eq"] = t["pnl"] * s          # normalised to common unit
        pf = (t["r"].clip(lower=0).sum() / -t["r"].clip(upper=0).sum()) \
            if (t["r"] < 0).any() else float("inf")
        report.append((sym, len(t), t["r"].sum(), pf, t["entry_ts"].dt.year.nunique(),
                       len(t) / max(t["entry_ts"].dt.year.nunique(), 1)))
        t["day"] = t["exit_ts"].dt.date
        streams[sym] = t

    rr = pd.DataFrame(report, columns=["sym", "n", "sumR", "pf", "yrs", "trades/yr"])
    print("=== per-market OOS validation (rr=4, brk=0.3, long, n_win=20) ===")
    print(rr.round(2).to_string(index=False))

    port = pd.concat([s.set_index("day")["pnl_eq"] for s in streams.values()],
                     axis=1).fillna(0.0).sum(axis=1).sort_index()
    print(f"\nportfolio: {port.size} trading days, mean ${port.mean():+,.0f}/day, "
          f"std ${port.std():+,.0f}, maxL ${port.min():,.0f}")

    print(f"\n=== 30-day prop sim on the portfolio (dd=${DEF_DD:,}, "
          f"target ${DEF_TARGET:,}, 90% payout, $110 eval) ===")
    n = len(port)
    for k in (1, 2, 3, 4):
        pnl = port.to_numpy() * k
        hits = {c: 0 for c in CUTOFFS}
        blows = 0
        for s in range(n):
            bal = peak = 0.0
            blown = False
            for off in range(CUTOFFS[-1]):
                i = s + off
                if i >= n:
                    break
                bal += pnl[i]
                peak = max(peak, bal)
                if bal <= -DEF_DD or peak - bal > DEF_DD:
                    blows += 1
                    blown = True
                    break
                if bal >= DEF_TARGET:
                    for c in CUTOFFS:
                        if off + 1 <= c:
                            hits[c] += 1
                    break
        ev30 = (hits[30] / n) * DEF_TARGET * 0.9 - 110
        print(f"  portfolio k={k}: "
              + "  ".join(f"<{c}d={hits[c] / n:.0%}" for c in CUTOFFS)
              + f" | blow={blows / n:.0%} EV30=${ev30:+,.0f}")


if __name__ == "__main__":
    main()