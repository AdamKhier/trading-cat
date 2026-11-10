import itertools
import pandas as pd
import backtest as bt, categorical as ct
from run import load
from walkforward import long_only, stats, GRID, SPLIT

df = ct.detect(load("1d"), {"stable_vol_max": 1e9})
rows = []
for rr, brk, eff, ddir in itertools.product(
        GRID["rr"], GRID["bracket"], GRID["eff_range"], GRID["direction"]):
    d = long_only(df) if ddir == "long" else df
    res = bt.run(d, params={"rr": rr, "bracket": brk}, active=None)
    t = res.trade_frame[res.trade_frame.entry_ts < SPLIT]
    s = stats(t)
    rows.append((rr, brk, eff, ddir, s))
rows = sorted(rows, key=lambda r: -r[4]["sumR"])
print("top 12 by sumR on train <2015:")
for rr, brk, eff, ddir, s in rows[:12]:
    print(f"  rr={rr:g} brk={brk:g} eff={eff:g} {ddir:4s} n={s['n']:3d} "
          f"sumR={s['sumR']:+6.1f} pf={s['pf']:.2f} win={s['win']:.0%}")
nz = [(r[4]["sumR"], r[4]["pf"]) for r in rows if r[4]["n"] >= 50]
print("\nn>=50 subset count:", len(nz), "| max pf:", max(p for _, p in nz or [(0, 0)]) if nz else "n/a")