# U.S. Treasury ETF Rotation Strategy — Backtest Framework

A rules-based rotation strategy across six iShares U.S. Treasury ETFs spanning the full maturity spectrum, benchmarked against the Bloomberg U.S. Treasury Total Return Index (LUATTRUU).

## Strategy Overview

### ETF Universe

| Ticker | ETF | Maturity Segment |
|--------|---------------------------------------------|------------------|
| SHV | iShares Short Treasury Bond ETF | 1–12 months |
| SHY | iShares 1–3 Year Treasury Bond ETF | 1–3 years |
| IEI | iShares 3–7 Year Treasury Bond ETF | 3–7 years |
| IEF | iShares 7–10 Year Treasury Bond ETF | 7–10 years |
| TLH | iShares 10–20 Year Treasury Bond ETF | 10–20 years |
| TLT | iShares 20+ Year Treasury Bond ETF | 20+ years |

### Rotation Rules

At each rebalancing date, the six ETFs are ranked by their **total return in the previous period**. The top, middle, and bottom two are assigned to three strategies:

- **Winners (W):** The 2 ETFs with the **highest** prior-period returns → long duration momentum
- **Median (M):** The 2 ETFs in the **middle** of the ranking → intermediate, behaviorally neutral
- **Losers (L):** The 2 ETFs with the **lowest** prior-period returns → short duration / mean-reversion

Each selected pair is held at **equal weight (50/50)** for the next rebalancing interval. There is **no look-ahead bias** — the first period uses equal-weight across all six ETFs.

### Rebalancing Frequencies Tested

| Frequency | Unit | Interval |
|------------|--------|----------|
| Annual | months | 12 |
| Semi-Annual| months | 6 |
| Quarterly | months | 3 |
| Monthly | months | 1 |
| Bi-Weekly | weeks | 2 |
| Weekly | weeks | 1 |
| Daily | days | 1 |

### Benchmark

**Bloomberg U.S. Treasury Total Return Index (LUATTRUU)** — a passive buy-and-hold benchmark representing the broad U.S. Treasury market.

### Key Finding

The **Median strategy with semi-annual rebalancing** achieves the best balance between return and risk. By investing in mid-performing maturities, it avoids overexposure to duration risk while capturing consistent, compounding growth — much like the statistical median filters out extreme outliers.

### Evaluation Period

January 2008 – March 2025 (initial investment: $100)

### Performance Metrics

- Cumulative portfolio value
- Annualized return (CAGR)
- Annualized volatility
- Sharpe ratio
- Maximum drawdown (MDD)

---

## Project Structure

```
us_treasury_rotation/
├── data/                       # 
│   ├── SHV_weekly_return_detailed.csv
│   ├── SHY_weekly_return_detailed.csv
│   ├── IEI_weekly_return_detailed.csv
│   ├── IEF_weekly_return_detailed.csv
│   ├── TLH_weekly_return_detailed.csv
│   ├── TLT_weekly_return_detailed.csv
│   └── LUATTRUU_clean.xlsx
│
├── results/                       # ← All outputs go here
│   ├── tables/                    #    CSV summaries per frequency
│   │   ├── annual_summary_12m.csv
│   │   ├── annual_summary_6m.csv
│   │   ├── ...
│   │   └── semi_annual_memberships.csv
│   └── visualizations/            #    Charts and figures
│       ├── final_portfolio_values.png
│       ├── Growth_Curve_Semi-Annual.png
│       ├── Growth_Curve_Quarterly.png
│       ├── Growth_Curves_All_Frequencies.pdf
│       ├── Annual_Return_pct_Rebalancing.png
│       ├── Sharpe_Ratio_Rebalancing.png
│       ├── Volatility_pct_Rebalancing.png
│       └── Max_Drawdown_pct_Rebalancing.png
│
├── src/                           # ← Modular source code
│   ├── __init__.py
│   ├── config.py                  #    Constants, paths, ETF list, settings
│   ├── utils.py                   #    Risk metrics, date helpers, data loaders
│   ├── backtest.py                #    Core rotation backtest engine
│   ├── selection_table.py         #    Semi-annual membership table builder
│   └── visualizations.py          #    All plotting functions
│
├── run_backtest.py                # ← CLI entry point (runs full pipeline)
├── results_notebook.ipynb         # ← Jupyter notebook for presenting results
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your data

Place the ETF CSV files and LUATTRUU Excel file in the `data/` directory. Each ETF CSV must contain at minimum `Date` and `Adj Close` columns.

### 3. Run the backtest

**Option A — Command line:**

```bash
python run_backtest.py          # all 7 frequencies
python run_backtest.py --quick  # Annual + Semi-Annual only
```

**Option B — Jupyter notebook:**

```bash
jupyter notebook results_notebook.ipynb
```

Run all cells to execute the backtest and display tables + visualizations inline.

### 4. View results

- **Tables:** `results/tables/`
- **Charts:** `results/visualizations/`

---

## Data Requirements

| File | Required Columns |
|------|-----------------|
| `{TICKER}_weekly_return_detailed.csv` | `Date`, `Adj Close` |
| `LUATTRUU_clean.xlsx` | `Date`, `TR_DAILY` (or `PX_LAST`) |

