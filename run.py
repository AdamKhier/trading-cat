"""Run the categorical backtest across ES timeframes and report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import categorical as ct
import config as C
from fetch import fetch

DATA = Path(__file__).parent / "data"
TICKER = "ES=F"
PV = C.INSTRUMENTS[TICKER]["point_value"]

INTRA = ["1m", "5m", "15m", "30m", "1h"]
ALL_TF = INTRA + ["1d"]


def load(tf: str) -> pd.DataFrame:
    df = fetch(TICKER, tf)
    if C.RTH_ONLY and tf != "1d":
        t = df.index.tz_localize("UTC").tz_convert("America/New_York")
        hm = pd.Series(t.strftime("%H:%M"), index=df.index)
        df = df[hm.between(C.RTH_START, C.RTH_END)].copy()
    return df


def metrics(res: bt.Result) -> dict:
    tf = res.trade_frame
    n = len(tf)
    if n == 0:
        return {"n": 0}
    wins = tf[tf["r"] > 0]
    cum = tf["r"].cumsum()
    peak = cum.cummax()
    dd = (peak - cum).max()
    pf = abs(tf[tf["r"] > 0]["r"].sum()) / abs(tf[tf["r"] < 0]["r"].sum()) \
        if (tf["r"] < 0).any() else np.inf
    daily = res.df.index.tz_localize("UTC").tz_convert("America/New_York") \
        .strftime("%Y-%m-%d").nunique()
    return {
        "n": n,
        "winrate": (wins["r"] > 0).sum() / n if n else 0.0,
        "expectancy_r": tf["r"].mean(),
        "sum_r": tf["r"].sum(),
        "profit_factor": pf,
        "max_dd_r": dd,
        "trades_per_day": n / max(daily, 1),
        "pnl": tf["pnl"].sum(),
    }


def cat_breakdown(res: bt.Result) -> pd.DataFrame:
    tf = res.trade_frame
    if tf.empty:
        return pd.DataFrame()
    g = tf.groupby("cat").agg(
        n=("r", "size"),
        winrate=("r", lambda x: (x > 0).mean()),
        avg_r=("r", "mean"),
        sum_r=("r", "sum"),
    ).sort_values("n", ascending=False)
    return g


def vol_summary(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for tf, d in dfs.items():
        tr = d["tr"].dropna()
        if len(tr) < 10:
            continue
        norm = tr / d["Close"]
        voo = tr.rolling(20).std() / tr.rolling(20).mean()
        active = ct.active_tf(d)
        rows[tf] = {
            "bars": len(d),
            "mean_norm_atr": float(norm.mean() * 100),
            "voo": float(voo.mean()),
            "vol_stable_%": float((voo <= C.STABLE_VOL_MAX).mean() * 100),
            "active_%": float(active.mean() * 100),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> None:
    print("fetching + filtering...")
    dfs = {tf: load(tf) for tf in ALL_TF}

    print("\n=== volatility across timeframes (automated timeframe pick) ===")
    dfs = {tf: ct.detect(dfs[tf]) for tf in ALL_TF}
    print(vol_summary(dfs).round(2).to_string())

    results = {}
    for tf in ALL_TF:
        df = dfs[tf]
        active = ct.active_tf(df)
        res = bt.run(df, active=active)
        results[tf] = res

    print("\n=== performance (1 contract, no position reuse) ===")
    rows = {}
    for tf in ALL_TF:
        m = metrics(results[tf])
        rows[tf] = m
    perf = pd.DataFrame.from_dict(rows, orient="index")
    print(perf.round(3).to_string())

    print("\n=== best performer(s) ===")
    best = perf.loc[perf["sum_r"].idxmax()]
    best_tf = perf["sum_r"].idxmax()
    print(f"trade the {best_tf} chart (sum {best['sum_r']:.1f}R, "
          f"{best['n']} trades, win {best['winrate']:.0%})")

    for tf in ALL_TF:
        cb = cat_breakdown(results[tf])
        if not cb.empty:
            print(f"\n--- {tf}: category breakdown ---")
            print(cb.round(3).to_string())

    # equity curves (R-units, per timeframe)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for tf in ALL_TF:
        tfm = results[tf].trade_frame
        if tfm.empty:
            continue
        cum = tfm.set_index("exit_ts")["r"].cumsum()
        ax.plot(cum.index, cum.values, label=tf, lw=1.2)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title(f"{TICKER}: categorical strategy cumulative R by timeframe")
    ax.set_ylabel("cumulative R (1 contract)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DATA.parent / "equity_curves.png", dpi=130)
    print("\nsaved equity_curves.png")

    # dump trades for inspection
    for tf in ALL_TF:
        tfm = results[tf].trade_frame
        if not tfm.empty:
            tfm.to_csv(DATA / f"trades_{tf}.csv", index=False)


if __name__ == "__main__":
    main()