# CAT — Categorical Futures Backtester (ES/MES/NQ/MNQ/CL/GC/YM)

Quantified implementation of categorical price-action trading: classify each bar into categories, backtest continuation/mean-reversion edge per timeframe, walk-forward out-of-sample validation, prop-firm sizing.

## Strategy in one paragraph

Bars are bucketed by range/structure into categories; per-category expectancy is measured on train data, then traded on unseen data with RTH filtering, volatility gating (ATR regime), costs (slippage + commission), and prop-firm position sizing. `walkforward.py` produces OOS equity curves.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy matplotlib
python3 run.py            # categorical backtest across ES timeframes
python3 walkforward.py    # walk-forward OOS + equity_curves.png
python3 backtest.py       # single backtest engine
```

## Files

| Path | What it is |
| --- | --- |
| `run.py` | Run categorical backtest across timeframes |
| `categorical.py` | Category detection logic |
| `backtest.py` / `mtf30.py` / `prop30.py` / `propsize.py` | Engines + prop sizing |
| `config.py` / `costs.py` / `fetch.py` | Instruments, costs, data loading |
| `sweep.py` / `spotcheck.py` / `freq.py` / `dbg_train.py` | Parameter sweeps + diagnostics |
| `walkforward.py` | Walk-forward OOS validation |
| `data/` | Sample futures CSVs (ES/MES/NQ/MNQ/CL/GC/YM) |
| `equity_curves.png` / `walkforward_oos.png` | Example outputs |

## Data note

CSVs in `data/` are a sample cache. Point values default to ES = $50/pt (see `config.py`). RTH filtering and volatility bands are tunable in `config.py`.
