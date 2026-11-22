# CAT — Categorical Futures Backtester — Project Summary

> Reconstructed 22 Nov 2026 from the repository itself, not from chat history.

## What it is
A quantified implementation of categorical price-action trading for ES/MES/NQ/MNQ/CL/GC/YM: classify each bar into categories, measure per-category expectancy on train data, then trade it out-of-sample with RTH filtering, ATR volatility gating, costs and prop-firm sizing.

## Stack
Python 3 + pandas, numpy, matplotlib. No package or lock file — a plain venv with those three installed.

## How to run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy matplotlib
python3 run.py            # categorical backtest across ES timeframes
python3 walkforward.py    # walk-forward OOS + equity_curves.png
python3 backtest.py       # single backtest engine
```

## Layout
| File | Purpose |
| --- | --- |
| `run.py` | Backtest across timeframes |
| `categorical.py` | Category detection |
| `backtest.py`, `mtf30.py`, `prop30.py`, `propsize.py` | Engines and prop sizing |
| `config.py`, `costs.py`, `fetch.py` | Instruments, costs, data loading |
| `sweep.py`, `spotcheck.py`, `freq.py`, `dbg_train.py` | Sweeps and diagnostics |
| `walkforward.py` | OOS validation |
| `data/` | Sample CSVs (ES/MES/NQ/MNQ/CL/GC/YM) |

## Status
Research harness with sample data only — `data/` is a cache from Sep 2026, not a live feed. Point values default to ES = $50/pt in `config.py`.

## Gaps
- No dependency manifest; environment must be rebuilt by hand.
- Sample CSVs are a subset — no multi-year dataset.

## Git
`AdamKhier/trading-cat` — public.
