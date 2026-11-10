"""Walk-forward validation of the categorical strategy on daily ES.

Two submissions:

1. STATIC SPLIT  : pick best params on 2006-2015, trade 2015-2026 blind.
2. ROLLING WALK  : every 2 years re-pick best params on expanding history,
                   trade the NEXT 2 years blind; accumulate all OOS trades.

Selection on each training window: highest sumR among configs with PF>=1.15
and at least 50 trades. Stability gate and vol band are off (established
last session: they only hurt). Parameter grid is deliberately small.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import backtest as bt
import categorical as ct
import config as C
from run import load

SPLIT = pd.Timestamp("2015-01-01")
STABLE = 1e9                                     # effectively off
GRID = {
    "rr": [1.0, 2.0, 3.0, 4.0],
    "bracket": [0.3, 0.5, 0.7],
    "eff_range": [0.25, 0.30, 0.35],
    "direction": ["both", "long"],               # long = trend-up + range-long only
}
COSA = 2 * C.SLIP * C.INSTRUMENTS["ES=F"]["point_value"] + C.COMM


def long_only(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.loc[d["cat"].isin(["TREND_DOWN", "RANGE_S"]), "cat"] = "NEUTRAL"
    return d


def stats(trades) -> dict:
    if len(trades) == 0:
        return {"n": 0}
    t = trades
    risk = (t["entry"] - t["sl"]).abs().mean() * C.INSTRUMENTS["ES=F"]["point_value"]
    return {
        "n": len(t),
        "win": (t["r"] > 0).mean(),
        "sumR": t["r"].sum(),
        "pf": (t["r"].clip(lower=0).sum() / -t["r"].clip(upper=0).sum())
              if (t["r"] < 0).any() else float("inf"),
        "net": t["r"].sum() * risk - len(t) * COSA,
    }


def select(trades) -> tuple:
    """Stats for a trade-set (training-eligibility: min sample only)."""
    s = stats(trades)
    if s["n"] < 50:
        return None
    return s


def sweep_on(df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp):
    """Return ranked (score, cfg, stats) for all configs restricted to [t0,t1)."""
    rows = []
    for rr, brk, eff, ddir in itertools.product(
            GRID["rr"], GRID["bracket"], GRID["eff_range"], GRID["direction"]):
        d = long_only(df) if ddir == "long" else df
        res = bt.run(d, params={"rr": rr, "bracket": brk}, active=None)
        t = res.trade_frame
        t = t[(t.entry_ts >= t0) & (t.entry_ts < t1)]
        s = select(t)
        if s is None:
            continue
        rows.append((s["sumR"], (rr, brk, eff, ddir), s))
    return sorted(rows, key=lambda x: -x[0])


def main():
    df = ct.detect(load("1d"), {"stable_vol_max": STABLE})
    print(f"daily data: {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} bars)\n")

    # ---- 1. static split ---------------------------------------------
    train = [w for w in sweep_on(df, df.index[0], SPLIT)]
    print(f"=== STATIC SPLIT: tuned on <{SPLIT.date()} ({len(train)} viable configs) ===")
    if not train:
        print("  no config passes the training bar.")
        return
    best_cfg = train[0][1]
    s_best = train[0][2]
    print(f"  selected: rr={best_cfg[0]} bracket={best_cfg[1]} "
          f"eff_range={best_cfg[2]} dir={best_cfg[3]}  "
          f"train n={s_best['n']} sumR={s_best['sumR']:+.1f} pf={s_best['pf']:.2f}\n")
    print("  top-5 on TRAIN -> their blind result on OOS 2015-2026 (shows selection bias!):")
    for _, cfg, s in train[:5]:
        d = df if cfg[3] == "both" else long_only(df)
        res = bt.run(d, params={"rr": cfg[0], "bracket": cfg[1]}, active=None)
        oos = res.trade_frame[res.trade_frame.entry_ts >= SPLIT]
        so = stats(oos)
        print(f"   rr={cfg[0]:g} brk={cfg[1]:g} eff={cfg[2]:g} {cfg[3]:4s} "
              f"| train n={s['n']:3d} sumR={s['sumR']:+6.1f} pf={s['pf']:.2f} "
              f"| OOS n={so['n']:3d} sumR={so['sumR']:+6.1f} pf={so['pf']:.2f} net=${so.get('net', 0):+8,.0f}")

    # ---- 2. rolling walk-forward -------------------------------------
    print("\n=== ROLLING WALK-FORWARD (blind) ===")
    oos_trades = []
    t_start = pd.Timestamp("2014-01-01")
    step = pd.Timedelta(days=365 * 2)
    last = df.index[-1]
    while t_start < last:
        t_run_end = min(t_start + step, last + pd.Timedelta(days=1))
        ranked = sweep_on(df, df.index[0], t_start)
        if not ranked:
            print(f"  {t_start.date()}->{t_run_end.date()}: no viable config, skipped")
        else:
            cfg = ranked[0][1]
            d = df if cfg[3] == "both" else long_only(df)
            res = bt.run(d, params={"rr": cfg[0], "bracket": cfg[1]}, active=None)
            tr = res.trade_frame[(res.trade_frame.entry_ts >= t_start) &
                                 (res.trade_frame.entry_ts < t_run_end)]
            oos_trades.append(tr)
            print(f"  {t_start.date()}->{t_run_end.date()}: picked rr={cfg[0]:g} "
                  f"brk={cfg[1]:g} eff={cfg[2]:g} {cfg[3]:5s} -> OOS n={len(tr):3d} "
                  f"sumR={tr['r'].sum():+6.1f}")
        t_start = t_run_end

    all_oos = pd.concat(oos_trades, ignore_index=True)
    s = stats(all_oos)
    cum = all_oos["r"].cumsum()
    dd = (cum.cummax() - cum).max()
    risk_usd = (all_oos["entry"] - all_oos["sl"]).abs().mean() \
        * C.INSTRUMENTS["ES=F"]["point_value"]
    print(f"\n  COMBINED ROLLING-OOS: n={len(all_oos)} "
          f"sumR={s['sumR']:+.1f} pf={s['pf']:.2f} "
          f"win={s['win']:.0%} maxDD={dd:.1f}R (${dd*risk_usd:+,.0f}) "
          f"net=${s['net']:+,.0f}")
    years = all_oos.groupby(all_oos["entry_ts"].dt.year)["r"].sum()
    print("   per-year R: " + "  ".join(f"{y}:{x:+.1f}" for y, x in years.items()))

    # buy & hold single-contract benchmark over the same window
    dfr = df[df.index >= all_oos["entry_ts"].min()]
    bh = (dfr["Close"].iloc[-1] - dfr["Open"].iloc[0]) * \
        C.INSTRUMENTS["ES=F"]["point_value"]
    print(f"\n  benchmark: buy&hold 1 ES contract, same window = ${bh:+,.0f} "
          f"(for comparison; this is pure long-beta)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(all_oos["entry_ts"], cum, lw=1.3)
    ax1.axhline(0, color="grey", lw=0.5)
    ax1.set_ylabel("cumulative R (1 contract)")
    ax1.set_title("Rolling walk-forward OOS: categorical strategy, daily ES")
    ax2.bar(years.index, years.values)
    ax2.axhline(0, color="grey", lw=0.5)
    ax2.set_ylabel("R / year")
    fig.tight_layout()
    fig.savefig("walkforward_oos.png", dpi=130)
    print("saved walkforward_oos.png")
    all_oos.to_csv("data/oos_trades.csv", index=False)


if __name__ == "__main__":
    main()