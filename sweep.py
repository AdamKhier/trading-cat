"""Grid sweep over volatility-stability leniency and R:R, per timeframe."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import backtest as bt
import categorical as ct
import config as C
from run import load

STABLE_VALS = [0.6, 1.0, 2.0, np.inf]            # std/mean TR cap (np.inf = off)
BANDS = [                                        # (vol_min, vol_max) normalised
    ("tight", 0.0004, 0.006),
    ("wide",  0.0002, 0.010),
    ("none",  0.0,    0.10),
]
RRS = [0.5, 1.0, 2.0]

TFS = ["1m", "5m", "15m", "30m", "1h"]


def evaluate(df, stable, band, rr, tf):
    dmin, dmax = dict(ct.detect.__globals__), {}
    act_params = {
        "stable_vol_max": stable,
        "vol_min": band[1],
        "vol_max": band[2],
        "voov_win": C.VOOV_WIN,
    }
    det_params = {"stable_vol_max": stable}
    detected = ct.detect(df, det_params)
    active = ct.active_tf(detected, act_params)
    res = bt.run(detected, params={"rr": rr}, active=active)
    t = res.trade_frame
    if t.empty:
        return None
    cum = t["r"].cumsum()
    dd = (cum.cummax() - cum).max()
    risk_usd = (t["entry"] - t["sl"]).abs() * C.INSTRUMENTS["ES=F"]["point_value"]
    cost = 2 * C.SLIP * C.INSTRUMENTS["ES=F"]["point_value"] + C.COMM
    net = t["r"].sum() * risk_usd.mean() - len(t) * cost
    return {
        "tf": tf,
        "stable": "off" if stable == np.inf else f"{stable:g}",
        "band": band[0],
        "rr": rr,
        "n": len(t),
        "win": (t["r"] > 0).mean(),
        "sumR": t["r"].sum(),
        "expR": t["r"].mean(),
        "pf": (t["r"].clip(lower=0).sum() / -t["r"].clip(upper=0).sum()
               if (t["r"] < 0).any() else np.inf),
        "ddR": dd,
        "net": net,
    }


def main():
    rows = []
    for tf in TFS:
        df = ct.detect(load(tf))
        for stable, band, rr in itertools.product(STABLE_VALS, BANDS, RRS):
            r = evaluate(df, stable, band, rr, tf)
            if r:
                rows.append(r)
    res = pd.DataFrame(rows)

    for tf in TFS:
        sub = res[res.tf == tf].sort_values("sumR", ascending=False)
        print(f"\n===== {tf} : top 12 by gross sumR =====")
        print(sub[["stable", "band", "rr", "n", "win", "sumR", "expR", "pf", "ddR", "net"]]
              .head(12).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n===== NET-positive after $30/trade costs (sorted by net) =====")
    good = res[res.net > 0].sort_values("net", ascending=False)
    if good.empty:
        print("  none.")
    else:
        print(good[["tf", "stable", "band", "rr", "n", "win", "sumR", "pf", "ddR", "net"]]
              .to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()