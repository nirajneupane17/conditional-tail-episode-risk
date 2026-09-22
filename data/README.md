# Data

All inputs to the CTER pipeline. The **per-series files in this directory are
the authoritative inputs**; `archive/` holds derived/optional files only.

## Files and coverage

| File | Rows | Start | End |
| --- | --- | --- | --- |
| `returns_sp500.csv` | 6,717 | 2000-01-04 | 2026-09-18 |
| `returns_nasdaq.csv` | 6,717 | 2000-01-04 | 2026-09-18 |
| `returns_tlt_bonds.csv` | 6,073 | 2002-07-31 | 2026-09-18 |
| `returns_eurusd.csv` | 6,692 | 2000-01-04 | 2026-09-11 |
| `returns_usdjpy.csv` | 6,692 | 2000-01-04 | 2026-09-11 |
| `returns_btc.csv` | 4,386 | 2014-09-18 | 2026-09-20 |
| `stress_vix.csv` | 6,751 | 2000-01-03 | 2026-09-17 |
| `stress_nfci.csv` | 1,393 | 2000-01-07 | 2026-09-11 |
| `stress_stlfsi.csv` | 1,393 | 2000-01-07 | 2026-09-11 |

TLT begins in 2002 (ETF inception) and Bitcoin in 2014; the equity and FX
series begin in 2000. Coverage differences are why out-of-sample windows are
**asset-specific** (see the main README).

## Field dictionary

**Return series** — `returns_<asset>.csv`

| Column | Type | Description |
| --- | --- | --- |
| `date` | ISO date | Trading date on the asset's own calendar |
| `logret` | float | Daily log return, `ln(P_t / P_{t-1})` |

The loaders in `src/lib.py` rename `logret` to `r` internally. Losses used
throughout are `L = -logret`.

**Stress series**

| File | Column | Description | Frequency |
| --- | --- | --- | --- |
| `stress_vix.csv` | `vix` | CBOE Volatility Index (level) | daily |
| `stress_nfci.csv` | `nfci` | Chicago Fed National Financial Conditions Index | weekly |
| `stress_stlfsi.csv` | `stlfsi` | St. Louis Fed Financial Stress Index | weekly |

## Weekly-to-daily alignment (LOCF)

NFCI and STLFSI are published weekly. They are aligned to the daily return
calendar by **last-observation-carried-forward (LOCF)**: each daily row takes
the most recent *already-released* weekly value. This is strictly backward-
looking, so no future information enters the feature set at any forecast origin.
This alignment is performed inside `src/lib.py::load_stress()` from the three
per-series files above.

## Calendar convention

Exchange-traded assets trade on business days; **Bitcoin trades 24/7**. All
horizons and rolling windows are therefore defined **in observations on each
asset's own calendar**, not in fixed calendar time. A "10-day observation
horizon" is 10 trading days for exchange-traded assets and 10 calendar days for
Bitcoin.

## `archive/` (not core inputs)

- `stress_all.csv` — a merged convenience file of the three stress series. It
  necessarily contains weekly `NaN`s for NFCI/STLFSI before LOCF and should
  **not** be used as a primary source; the per-series files are authoritative.
- `regime_calendar_sp500.csv`, `regime_vol_sp500.csv` — supplementary S&P 500
  regime diagnostics, not used by the committed pipeline.

## Sources and licensing

Market prices and the VIX are from standard market-data providers; NFCI and
STLFSI are published by the Federal Reserve Banks of Chicago and St. Louis
(FRED). Raw third-party market data may be subject to the original providers'
licensing terms; the derived log-return series are provided here for research
reproducibility.
