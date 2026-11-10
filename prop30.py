"""30-day prop-eval adaptation: sizing + signal-frequency sweep.

Metric: P(pass within CUTOFF days) at 90% payout vs $110 eval cost.
Running the same rolling-account simulation as propsize.py but scoring
against hard day-counters instead of median days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POINT_V = 5.0
COST_M = 2.5
SLIP_M = 0.25
CUTOFFS = [10, 20, 30, 60]
DEF_TARGET = 3_000
PAYOUT = 0.90
EVAL_COST = 110

DD_TIERS = [2_500, 3_000, 4_000, 5_000, 7_500]
K_RANGE = [2, 4, 6, 8, 12, 16, 20]


def load_daily() -> pd.Series:
    t = pd.read_csv("data/oos_trades.csv", parse_dates=["exit_ts"])
    risk = (t["entry"] - t["sl"]).abs() * POINT_V
    t["pnl_m"] = t["r"] * risk - (2 * SLIP_M + COST_M)
    t["day"] = t["exit_ts"].dt.date
    return t.groupby("day")["pnl_m"].sum().sort_index().astype(float)


def simulate(daily: pd.Series, k: int, dd: int, target: int):
    pnl = (daily * k).to_numpy()
    n = len(pnl)
    outcomes = []  # (kind, time_days) kind: P/B/-
    for s in range(n):
        bal = peak = 0.0
        kind = None
        for off in range(CUTOFFS[-1]):
            i = s + off
            if i >= n:
                break
            bal += pnl[i]
            peak = max(peak, bal)
            if bal <= -dd or peak - bal > dd:
                kind = ("B", off + 1); break
            if bal >= target:
                kind = ("P", off + 1); break
        outcomes.append(kind or ("-", CUTOFFS[-1]))
    return outcomes


def report(daily, k, dd, target=DEF_TARGET, label=""):
    out = simulate(daily, k, dd, target)
    arr = pd.DataFrame(out, columns=["kind", "days"])
    line = []
    for c in CUTOFFS:
        p_pass = len(arr[(arr.kind == "P") & (arr.days <= c)]) / len(arr)
        line.append(f"pass<={c}d={p_pass:.0%}")
    p_blow = (arr.kind == "B").mean()
    p_pass30 = len(arr[(arr.kind == "P") & (arr.days <= 30)]) / len(arr)
    ev = p_pass30 * target * PAYOUT - EVAL_COST
    print(f"  {label:34s} k={k:2d} dd=${dd:,.0f} | "
          + "  ".join(line) + f" | blow={p_blow:.0%} EV30=${ev:+,.0f}")
    return ev


def main():
    daily = load_daily()
    t = pd.read_csv("data/oos_trades.csv")
    risk1 = (t["entry"] - t["sl"]).abs() * POINT_V
    print(f"daily 1-MNQ pnl: mean ${daily.mean():+,.0f}/day, std ${daily.std():+.0f}, "
          f"maxL ${daily.min():,.0f}\n")

    print(f"=== sizing sweep (target ${DEF_TARGET:,}, {PAYOUT:.0%} payout, "
          f"${EVAL_COST} eval) ===")
    best = []
    for dd in DD_TIERS:
        for k in K_RANGE:
            out = simulate(daily, k, dd, DEF_TARGET)
            arr = pd.DataFrame(out, columns=["kind", "days"])
            p30 = len(arr[(arr.kind == "P") & (arr.days <= 30)]) / len(arr)
            ev = p30 * DEF_TARGET * PAYOUT - EVAL_COST
            best.append((ev, dd, k))
    best.sort(reverse=True)
    for ev, dd, k in best[:12]:
        report(daily, k, dd, label="best30")

    print("\n=== what 30-day success actually needs (dd=$5,000) ===")
    for k in (4, 6, 8, 12):
        report(daily, k, 5_000, label="d=$5k")

    print("\n=== does raising the trade frequency help? quick check: ===")
    tw = t.copy()
    tw["day"] = tw["exit_ts"].dt.date
    tpd = tw.groupby("day")["r"].count().mean()
    print(f"  current cadence: {tpd:.2f} trades/day "
          f"({len(tw)} over {len(tw.day.unique())} days)")


if __name__ == "__main__":
    main()