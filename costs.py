import pandas as pd
from run import load
import backtest as bt, categorical as ct, config as C

for tf in ["5m", "15m", "30m", "1h"]:
    df = ct.detect(load(tf))
    res = bt.run(df, active=ct.active_tf(df))
    t = res.trade_frame
    if len(t) == 0:
        continue
    risk_pts = (t.entry - t.sl).abs()
    R_dollars = risk_pts * C.INSTRUMENTS["ES=F"]["point_value"]
    cost = 2 * C.SLIP * C.INSTRUMENTS["ES=F"]["point_value"] + C.COMM
    gross = t.r.mean() * R_dollars.mean()
    print(
        f"{tf:3s}: trades={len(t):3d} mean_R=${R_dollars.mean():7.0f} "
        f"avg_gross_$={gross:+7.0f} cost=${cost:.0f}/trade "
        f"net_per_trade=${gross - cost:+7.0f}"
    )