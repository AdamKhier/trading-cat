"""Quantified version of Iman's categorical trading.

All parameters are Iman-inspired defaults; tweak freely.
Point conventions: ES = $50/pt. Slippage and commission in $.
"""

INSTRUMENTS = {
    "ES=F": {"point_value": 50.0},
}

# ---- timeframe / volatility automation -------------------------------
# Iman: adjust the chart's timeframe so volatility is in a workable band.
# Normalised volatility = mean(ATR1) / price, per bar.
VOL_MIN = 0.0004      # below this, bars are too quiet to give an edge
VOL_MAX = 0.006       # above this, bars are too wild / hard to categorise
STABLE_VOL_MAX = 1.0  # max std(TR)/mean(TR); above => "ATR is all over the place"
VOOV_WIN = 40         # bars used for the volatility-of-volatility measure

# ---- category detection ----------------------------------------------
N_WIN = 20            # bars in the range/structure window
EFF_RANGE = 0.30      # width / sum(TR) <= this => consolidation
EFF_TREND = 0.55      # (informational) efficiency above this => directional probe
WHIPSAW_LOOK = 8      # break both ways within this many bars => chaotic

# ---- trade rules -----------------------------------------------------
BRACKET = 0.50        # stop distance = BRACKET * mean(TR)  (Iman: ~half candle)
RR = 1.0              # risk:reward, start 1:1 per Iman
RANGE_SL_MARGIN = 0.25  # range stop sits this many mean(TR) outside the band

# ---- costs -----------------------------------------------------------
SLIP = 0.25           # points, per side (ES 1 tick)
COMM = 5.00           # $, per round turn

# ---- sessions --------------------------------------------------------
RTH_ONLY = True       # trade only RTH bars for intraday timeframes
RTH_START = "09:30"
RTH_END = "16:00"