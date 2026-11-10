"""Prop-account sizing study (MNQ granularity).

ES fills the whole drawdown in one trade, so sizing below 1 ES contract
is impossible. MNQ ($5/pt, 1/10 ES) lets us actually choose risk per trade.
We rebuild per-trade MNQ PnL from the OOS trade file, aggregate to daily,
and sweep micro-contract counts x firm drawdown tiers to find the
settings with the best mix of pass probability and EV per evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POINT_V = 5.0      # MNQ point value
COST_M = 2.5       # $/RT commission (micro)
SLIP_M = 0.25      # tick (~$1.25/side)

FIRM_TEMPLATE = {
    "target": 3_000,
    "payout": 0.90,
    "cost": 110,       # eval fee guesses: Apex-style
}
DD_TIERS = [2_000, 2_500, 3_000, 4_000, 5_000, 7_500]
WK_DAYS_CAP = 540


def load_trade_pnls() -> pd.Series:
    t = pd.read_csv("data/oos_trades.csv", parse_dates=["exit_ts"])
    risk = (t["entry"] - t["sl"]).abs() * POINT_V
    pnl = t["r"] * risk - (2 * SLIP_M + COST_M)
    t["pnl_m"] = pnl
    t["day"] = t["exit_ts"].dt.date
    return t.groupby("day")["pnl_m"].sum().sort_index().astype(float)


def simulate(daily: pd.Series, k: int, dd: int, daily_ll: int, target: int):
    pnl = daily * k
    passes = blows = cuts = 0
    pass_days = []
    n = len(pnl)
    for s in range(n):
        bal = 0.0
        peak = 0.0
        out = None
        for off in range(WK_DAYS_CAP):
            i = s + off
            if i >= n:
                break
            bal += pnl.iloc[i]
            peak = max(peak, bal)
            if daily_ll and pnl.iloc[i] <= -daily_ll:
                out = "B"; break
            if bal <= -dd:
                out = "B"; break
            if peak - bal > dd:
                out = "B"; break
            if bal >= target:
                out = "P"; break
        if out == "P":
            passes += 1; pass_days.append(off)
        elif out == "B":
            blows += 1
        else:
            cuts += 1
    total = passes + blows + cuts
    p_pass = passes / total
    p_blow = blows / total
    ev = p_pass * target * FIRM_TEMPLATE["payout"] - FIRM_TEMPLATE["cost"]
    med = float(np.median(pass_days)) if pass_days else float("nan")
    return p_pass, p_blow, med, ev


def main():
    daily = load_trade_pnls()
    t = pd.read_csv("data/oos_trades.csv")
    risk1 = (t["entry"] - t["sl"]).abs() * POINT_V
    print(f"OOS trades: {len(t)} | MNQ per-trade risk p50=${risk1.median():,.0f} "
          f"p95=${risk1.quantile(0.95):,.0f} | daily max win ${daily.max():+,.0f} "
          f"max loss ${daily.min():+,.0f}\n")

    best = []
    for dd in DD_TIERS:
        for daily_ll in (0, dd):
            for k in (2, 4, 6, 8, 12):
                pp, pb, med, ev = simulate(daily, k, dd, daily_ll,
                                           FIRM_TEMPLATE["target"])
                best.append((pp, pb, med, ev, dd, daily_ll, k))

    best.sort(reverse=True)
    print("=== top configs by EV/eval (pass prob >= 30%) ===")
    shown = 0
    for pp, pb, med, ev, dd, dll, k in best:
        if pp < 0.30:
            break
        print(f"  dd=${dd:,.0f} dailyLL=${dll:,.0f} k={k:2d} MNQ "
              f"| pass={pp:.0%} blow={pb:.0%} med_pass={med:.0f}d EV=${ev:+,.0f}")
        shown += 1
        if shown >= 14:
            break

    print("\n=== risk/hazard: does dailyLL hurt or help? (dd=$4,000, k=4) ===")
    for dll in (0, 4000):
        pp, pb, med, ev = simulate(daily, 4, 4000, dll, FIRM_TEMPLATE["target"])
        print(f"  dailyLL=${dll:,.0f}: pass={pp:.0%} blow={pb:.0%} med_pass={med:.0f}d EV=${ev:+,.0f}")

    # per-trade stop distances, to hand-size contracts by $ risk
    print("\n=== sizing map: largest k such that p95 risk <= budget ===")
    for budget in (125, 250, 375, 500, 750):
        k = max(1, int(budget // risk1.quantile(0.95)))
        tot = k * risk1.quantile(0.95)
        print(f"  budget ${budget:,} risk/trade -> k={k} MNQ (p95 risk ${tot:,.0f})")


if __name__ == "__main__":
    main()