"""Event-driven backtest of the categorical strategy on one timeframe."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import config as C


@dataclass
class Trade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    cat: str            # category at entry
    direction: int      # +1 long, -1 short
    entry: float
    exit: float
    sl: float
    tp: float
    r: float            # realised PnL in R-multiples
    pnl: float          # $ (1 contract less costs)
    exit_reason: str    # SL | TP | CAT
    bars_held: int = 0


@dataclass
class Result:
    trades: list = field(default_factory=list)
    df: pd.DataFrame = None

    @property
    def trade_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [t.__dict__ for t in self.trades],
            columns=["entry_ts", "exit_ts", "cat", "direction", "entry",
                     "exit", "sl", "tp", "r", "pnl", "exit_reason"],
        )


EXIT_REASONS = ("SL", "TP", "CAT")


def run(df: pd.DataFrame, params: dict | None = None,
        active: pd.Series | None = None) -> Result:
    p = {"rr": C.RR, "bracket": C.BRACKET, "range_sl_margin": C.RANGE_SL_MARGIN}
    p.update(params or {})

    if active is not None:
        df = df.copy()
        df["cat"] = df["cat"].where(active.reindex(df.index))
        df.loc[~active.reindex(df.index).fillna(False), "cat"] = pd.NA

    opens = df["Open"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()
    cats = df["cat"].to_numpy()
    mtr = df["mean_tr"].fillna(0).to_numpy()
    hhs = df["hh"].fillna(opens[0]).to_numpy()
    lls = df["ll"].fillna(opens[0]).to_numpy()
    mids = df["mid"].fillna(opens[0]).to_numpy()
    times = df.index

    res = Result(df=df)
    n = len(df)
    pos_dir = 0          # 0 flat, +1 long, -1 short
    entry = sl = tp = 0.0
    entry_cat = ""
    pos_ts = None
    pos_i = -1
    slip = C.SLIP

    for i in range(1, n):
        cat = cats[i - 1]                 # decision made on bar i-1 close
        cat_now = cats[i]                 # current bar's category (for exits)
        if pos_dir == 0:
            if cat is None or pd.isna(cat) or cat in ("CHAOS", "NEUTRAL"):
                continue
            if cat == "RANGE":
                lo, hi, mid, m = lls[i - 1], hhs[i - 1], mids[i - 1], mtr[i - 1]
                risk_base = m * p["range_sl_margin"]
                long_ok = lows[i] <= lo and closes[i] >= lo
                short_ok = highs[i] >= hi and closes[i] <= hi
                if long_ok:
                    e = opens[i]
                    stop = lo - risk_base
                    target = min(e + (e - stop) * p["rr"], (hi + lo) / 2)
                    if target > e:
                        pos_dir, entry, sl, tp = 1, e, stop, target
                        entry_cat, pos_ts, pos_i = "RANGE_L", times[i], i
                elif short_ok:
                    e = opens[i]
                    stop = hi + risk_base
                    target = max(e - (stop - e) * p["rr"], (hi + lo) / 2)
                    if target < e:
                        pos_dir, entry, sl, tp = -1, e, stop, target
                        entry_cat, pos_ts, pos_i = "RANGE_S", times[i], i
            elif cat in ("TREND_UP", "TREND_DOWN"):
                e = opens[i]
                d = 1 if cat == "TREND_UP" else -1
                stop = e - d * mtr[i - 1] * p["bracket"]
                target = e + d * (abs(e - stop) * p["rr"])
                pos_dir, entry, sl, tp = d, e, stop, target
                entry_cat, pos_ts, pos_i = cat, times[i], i
            continue

        # ---- manage open position -----------------------------------
        h, l, o, c = highs[i], lows[i], opens[i], closes[i]
        if pos_dir > 0:
            hit_stop = l <= sl
            hit_tp = h >= tp
            cat_change = cat_now in ("CHAOS",) or (cat_now == "TREND_DOWN") or \
                (entry_cat == "TREND_UP" and cat_now == "RANGE")
        else:
            hit_stop = h >= sl
            hit_tp = l <= tp
            cat_change = cat_now in ("CHAOS",) or (cat_now == "TREND_UP") or \
                (entry_cat == "TREND_DOWN" and cat_now == "RANGE")

        if hit_stop and hit_tp:           # conservative: assume SL first
            fill, reason = sl, "SL"
        elif hit_stop:
            fill, reason = sl, "SL"
        elif hit_tp:
            fill, reason = tp, "TP"
        elif cat_change:
            fill, reason = c, "CAT"
        else:
            continue

        risk = abs(entry - sl)
        r_units = (fill - entry) * pos_dir / risk
        pnl_r = r_units
        pnl = ((fill - entry) * pos_dir * C.INSTRUMENTS["ES=F"]["point_value"]
               - 2 * slip * C.INSTRUMENTS["ES=F"]["point_value"] - C.COMM)
        res.trades.append(Trade(
            entry_ts=pos_ts, exit_ts=times[i], cat=entry_cat, direction=pos_dir,
            entry=entry, exit=fill, sl=sl, tp=tp, r=pnl_r, pnl=pnl,
            exit_reason=reason, bars_held=i - pos_i,
        ))
        pos_dir = 0

    return res