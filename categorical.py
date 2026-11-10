"""Categorical detector: tags every bar as RANGE, TREND_UP, TREND_DOWN, or CHAOS.

Indicators use prior-bar info only (rolling windows shifted by 1 bar).
Two-scale-free measures drive the categories:

* range efficiency  = window_width / sum(TR over window)
    -> low  ~ price keeps returning to where it's been  (consolidation)
    -> high ~ price keeps reaching new areas             (direction)
* volatility-of-volatility = std(TR)/mean(TR) over a longer lookback
    -> high ~ "ATR is all over the place", be very cautious
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["Close"].shift(1)
    return pd.concat(
        [df["High"] - df["Low"],
         (df["High"] - pc).abs(),
         (df["Low"] - pc).abs()],
        axis=1,
    ).max(axis=1)


def _roll(src: pd.Series, win: int, fn: str) -> pd.Series:
    """Rolling fn with a few missing values tolerated at the start."""
    return getattr(src.rolling(win, min_periods=max(win // 3, 2)), fn)()


def detect(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    p = {
        "n_win": C.N_WIN,
        "eff_range": C.EFF_RANGE,
        "eff_trend": C.EFF_TREND,
        "whipsaw_look": C.WHIPSAW_LOOK,
        "stable_vol_max": C.STABLE_VOL_MAX,
        "voov_win": C.VOOV_WIN,
    }
    p.update(params or {})

    n, voov = p["n_win"], p["voov_win"]

    out = df[["Open", "High", "Low", "Close"]].copy()

    # --- volatility (ATR = period-1 true range) -----------------------
    out["tr"] = true_range(out)
    out["mean_tr"] = _roll(out["tr"], n, "mean").shift(1)
    out["sum_tr"] = _roll(out["tr"], n, "sum").shift(1)

    # normalised volatility (per-bar move as fraction of price), bar t-1
    out["norm"] = out["mean_tr"] / out["Close"].shift(1)
    # volatility-of-volatility from a longer lookback of TR
    tstd = _roll(out["tr"], voov, "std").shift(1)
    tmean = _roll(out["tr"], voov, "mean").shift(1)
    out["voo"] = tstd / tmean
    out["vol_stable"] = out["voo"] <= p["stable_vol_max"]

    # --- structure window (prior bars only) ---------------------------
    out["hh"] = _roll(out["High"], n, "max").shift(1)
    out["ll"] = _roll(out["Low"], n, "min").shift(1)
    out["width"] = out["hh"] - out["ll"]
    out["mid"] = (out["hh"] + out["ll"]) / 2

    # range efficiency: overlap vs. exploration
    out["eff"] = out["width"] / out["sum_tr"]
    out["k_width"] = out["width"] / out["mean_tr"]

    # fresh breaks of prior-window structure (price at new areas)
    out["break_up"] = out["Close"] > out["hh"]
    out["break_dn"] = out["Close"] < out["ll"]

    # whipsaw: both a break-up and a break-down recently => two-sided chop
    up_recent = _roll(out["break_up"].astype(float), p["whipsaw_look"], "max")
    dn_recent = _roll(out["break_dn"].astype(float), p["whipsaw_look"], "max")
    out["whipsaw"] = (up_recent > 0) & (dn_recent > 0)

    # --- category -----------------------------------------------------
    cat = pd.Series("NEUTRAL", index=out.index)
    cat[out["whipsaw"] | (~out["vol_stable"])] = "CHAOS"
    cat[(out["Close"] > out["hh"]) & (cat == "NEUTRAL")] = "TREND_UP"
    cat[(out["Close"] < out["ll"]) & (cat == "NEUTRAL")] = "TREND_DOWN"
    cat[(out["eff"] <= p["eff_range"]) & (cat == "NEUTRAL")] = "RANGE"
    out["cat"] = cat

    out[["whipsaw", "vol_stable", "break_up", "break_dn"]] = out[
        ["whipsaw", "vol_stable", "break_up", "break_dn"]
    ].fillna(False).astype(bool)
    return out


def active_tf(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Volatility filter that automates Iman's timeframe choice.
    True where this timeframe's normalised ATR sits in the workable band
    and volatility is stable.
    """
    p = {"vol_min": C.VOL_MIN, "vol_max": C.VOL_MAX,
         "stable_vol_max": C.STABLE_VOL_MAX, "voov_win": C.VOOV_WIN}
    p.update(params or {})
    tr = df["tr"]
    w = p["voov_win"]
    norm = tr.rolling(20, min_periods=5).mean() / df["Close"]
    voo = tr.rolling(w, min_periods=w // 3).std() / \
        tr.rolling(w, min_periods=w // 3).mean()
    return (norm.between(p["vol_min"], p["vol_max"])) & (voo <= p["stable_vol_max"])