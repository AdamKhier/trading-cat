import pandas as pd
import backtest as bt, categorical as ct, config as C
from run import load

print("=== 30m RR=2.0, stable=1.0, band=none: sample split + categories ===")
df = ct.detect(load("30m"), {"stable_vol_max": 1.0})
act = ct.active_tf(df, {"stable_vol_max": 1.0, "vol_min": 0.0, "vol_max": 0.10})
res = bt.run(df, params={"rr": 2.0}, active=act)
t = res.trade_frame
tmid = t.iloc[len(t) // 2]
for name, half in [("all", t), ("1st half", t[t.entry_ts <= tmid.entry_ts]),
                   ("2nd half", t[t.entry_ts > tmid.entry_ts])]:
    print(f"  {name:9s} n={len(half):3d} win={ (half.r>0).mean():.0%} "
          f"sumR={half.r.sum():+7.2f} pf={(half.r.clip(lower=0).sum()/-half.r.clip(upper=0).sum()):.2f}")
print("\n  by category:")
print(t.groupby("cat").agg(n=("r", "size"), win=("r", lambda x: (x > 0).mean()),
                           sum_r=("r", "sum")).round(2).to_string())

print("\n=== 1d (20 YEARS) RR sweep, stability off, band none ===")
dfd = ct.detect(load("1d"), {"stable_vol_max": 1e9})
actd = ct.active_tf(dfd, {"stable_vol_max": 1e9, "vol_min": 0.0, "vol_max": 10.0})
for rr in [1.0, 2.0, 3.0]:
    r = bt.run(dfd, params={"rr": rr}, active=actd)
    td = r.trade_frame
    if len(td):
        risk = (td.entry - td.sl).abs()
        net = td.r.sum() * risk.mean() * C.INSTRUMENTS["ES=F"]["point_value"] \
            - len(td) * (2 * C.SLIP * C.INSTRUMENTS["ES=F"]["point_value"] + C.COMM)
        print(f"  rr={rr} n={len(td):3d} win={(td.r>0).mean():.0%} sumR={td.r.sum():+7.2f} "
              f"pf={(td.r.clip(lower=0).sum()/-td.r.clip(upper=0).sum()):.2f} net=${net:+,.0f}")
    else:
        print(f"  rr={rr} -> no trades")

print("\n  1d category breakdown (rr=2):")
r = bt.run(dfd, params={"rr": 2.0}, active=actd)
print(r.trade_frame.groupby("cat").agg(n=("r", "size"), win=("r", lambda x: (x > 0).mean()),
                                       sum_r=("r", "sum")).round(2).to_string())