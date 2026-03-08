"""
config.py
Central configuration for the U.S. Treasury ETF Rotation Backtest.

All constants consumed by backtest.py, transaction_costs.py,
statistical_tests.py, and visualizations.py live here — one place to
change anything that touches multiple modules.
"""
import os
import numpy as np

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
VIZ_DIR      = os.path.join(RESULTS_DIR, "visualizations")
TABLES_DIR   = os.path.join(RESULTS_DIR, "tables")

# ── ETF universe ───────────────────────────────────────────────────────────
ETFS = ["SHV", "SHY", "IEI", "IEF", "TLH", "TLT"]

ETF_DESCRIPTIONS = {
    "SHV": "iShares Short Treasury Bond ETF (1–12 months)",
    "SHY": "iShares 1–3 Year Treasury Bond ETF",
    "IEI": "iShares 3–7 Year Treasury Bond ETF",
    "IEF": "iShares 7–10 Year Treasury Bond ETF",
    "TLH": "iShares 10–20 Year Treasury Bond ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
}

# ── Benchmark ──────────────────────────────────────────────────────────────
LUATTRUU_FILENAME = "LUATTRUU_clean.xlsx"

# ── Backtest parameters ────────────────────────────────────────────────────
START_YEAR   = 2008
END_YEAR     = 2025
INITIAL_CASH = 100
RF_ANNUAL    = 0.0

# ── Transaction cost assumptions ──────────────────────────────────────────
# Empirical bid-ask half-spreads (bps, one-way) sourced from Bloomberg/iShares.
ETF_SPREADS_BPS = {
    "SHV": 0.5,   # ultra-short, extremely liquid
    "SHY": 1.0,   # short duration, very liquid
    "IEI": 1.5,   # intermediate, liquid
    "IEF": 2.0,   # intermediate-long, liquid
    "TLH": 3.0,   # long duration, moderately liquid
    "TLT": 2.5,   # long duration, highly traded
}

_avg_spread          = np.mean(list(ETF_SPREADS_BPS.values()))
BASE_ROUND_TRIP_BPS  = round(2 * _avg_spread, 1)   # ≈ 3.5 bps

SCENARIO_MULTIPLIERS = {
    "Low (0.5×)": 0.5,
    "Base (1×)":  1.0,
    "High (2×)":  2.0,
}

# ── Rebalancing frequencies ────────────────────────────────────────────────
REBALANCING_SETTINGS = [
    ("Annual",      "months", 12),
    ("Semi-Annual", "months",  6),
    ("Quarterly",   "months",  3),
    ("Monthly",     "months",  1),
    ("Bi-Weekly",   "weeks",   2),
    ("Weekly",      "weeks",   1),
    ("Daily",       "days",    1),
]

# ── Strategy names and colors ──────────────────────────────────────────────
STRATEGIES = ["Winners", "Median", "Losers", "BuyHold_2"]

COLORS = {
    "Winners":   "#008000",
    "Median":    "#000000",
    "Losers":    "#0000FF",
    "BuyHold_2": "#FF0000",
}

STRAT_COLORS = {
    "Winners": "#008000",
    "Median":  "#000000",
    "Losers":  "#0000FF",
}
