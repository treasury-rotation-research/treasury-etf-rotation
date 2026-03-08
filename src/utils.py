"""
utils.py
Utility functions: risk metrics, date generation, data loading.
"""
import os
import numpy as np
import pandas as pd
from typing import List, Tuple

from src.config import DATA_DIR


# ═══════════════════════════════════════════════════════════════════════════
# Annualisation helper
# ═══════════════════════════════════════════════════════════════════════════

def infer_ann_factor(idx) -> int:
    """
    Infer the annualisation factor from a date index by inspecting the
    median gap between observations.

    Returns 252 (daily), 52 (weekly), 12 (monthly), or the nearest integer
    implied by the median gap in days.
    """
    try:
        sidx = pd.DatetimeIndex(idx).sort_values()
    except Exception:
        return 252
    if len(sidx) < 3:
        return 252
    median_days = pd.Series(sidx).diff().dropna().dt.days.median()
    if pd.isna(median_days):
        return 252
    if median_days <= 3:
        return 252
    if median_days <= 8:
        return 52
    if median_days <= 20:
        return 12
    return int(round(365.0 / max(median_days, 1.0)))


# ═══════════════════════════════════════════════════════════════════════════
# Risk / performance helpers  (all accept percentage-unit return series)
# ═══════════════════════════════════════════════════════════════════════════

def compute_mdd_from_pct(pct_series: pd.Series) -> float:
    """
    Maximum drawdown (%) from a series of *percentage* returns.
    Converts to decimal → builds cumulative wealth → computes drawdown.
    """
    s = pd.Series(pct_series).dropna().astype(float) / 100.0
    if s.empty:
        return np.nan
    wealth = (1.0 + s).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    return float(dd.min() * 100.0)


def annualized_stats_from_pct(
    pct_series: pd.Series,
    rf_ann: float = 0.0,
) -> Tuple[float, float]:
    """
    Return (annualized_volatility_%, sharpe_ratio) from a *percentage* return series.

    Volatility is annualized by multiplying the sample std by sqrt(ann_factor).
    Sharpe = (annualized_mean - rf_ann) / annualized_vol.
    Returns (nan, nan) for empty or constant series.
    """
    s = pd.Series(pct_series).dropna().astype(float) / 100.0
    if s.empty:
        return np.nan, np.nan
    ann      = infer_ann_factor(s.index)
    sig_ann  = float(s.std() * ann ** 0.5)
    mu_ann   = float(s.mean() * ann)
    vol_pct  = sig_ann * 100.0
    sharpe   = (
        np.nan if (sig_ann == 0.0 or not np.isfinite(sig_ann))
        else (mu_ann - rf_ann) / sig_ann
    )
    return vol_pct, sharpe


def sortino_from_pct(
    pct_series: pd.Series,
    rf_ann: float = 0.0,
) -> float:
    """
    Annualized Sortino ratio from a *percentage* return series.

    Sortino = (annualized_mean - rf_ann) / downside_deviation
    Downside deviation: sqrt(mean(r² for r < 0)) * sqrt(ann_factor).
    Uses MAR = 0 (returns below zero count as downside).

    Returns nan when fewer than 2 observations or no downside returns.
    """
    s = pd.Series(pct_series).dropna().astype(float) / 100.0
    if len(s) < 2:
        return np.nan
    ann         = infer_ann_factor(s.index)
    mu_ann      = float(s.mean() * ann)
    downside    = s[s < 0.0]
    if downside.empty:
        return np.nan
    dd_ann = float(np.sqrt((downside ** 2).mean()) * ann ** 0.5)
    if dd_ann == 0.0 or not np.isfinite(dd_ann):
        return np.nan
    return (mu_ann - rf_ann) / dd_ann


def information_ratio_from_pct(
    strategy_pct: pd.Series,
    benchmark_pct: pd.Series,
    rf_ann: float = 0.0,   # kept for API symmetry; IR does not use rf
) -> float:
    """
    Annualized Information Ratio from two *percentage* return series.

    IR = annualized_active_return / annualized_tracking_error
    where tracking_error = std(active_returns, ddof=1) * sqrt(ann_factor).

    Series are aligned on their common index before calculation.
    Returns nan when fewer than 2 overlapping observations or TE = 0.
    """
    s = pd.Series(strategy_pct).dropna().astype(float) / 100.0
    b = pd.Series(benchmark_pct).dropna().astype(float) / 100.0
    common = s.index.intersection(b.index)
    if len(common) < 2:
        return np.nan
    active = s.loc[common] - b.loc[common]
    ann    = infer_ann_factor(common)
    te     = float(active.std(ddof=1) * ann ** 0.5)
    if te == 0.0 or not np.isfinite(te):
        return np.nan
    return float(active.mean() * ann / te)


def tracking_error_from_pct(
    strategy_pct: pd.Series,
    benchmark_pct: pd.Series,
) -> float:
    """
    Annualized Tracking Error from two *percentage* return series.

    TE = std(active_returns, ddof=1) * sqrt(ann_factor)
    where active_return = strategy_return - benchmark_return per period.

    Series are aligned on their common index before calculation.
    Returns nan when fewer than 2 overlapping observations.
    """
    s = pd.Series(strategy_pct).dropna().astype(float) / 100.0
    b = pd.Series(benchmark_pct).dropna().astype(float) / 100.0
    common = s.index.intersection(b.index)
    if len(common) < 2:
        return np.nan
    active = s.loc[common] - b.loc[common]
    ann    = infer_ann_factor(common)
    te     = float(active.std(ddof=1) * ann ** 0.5)
    return te * 100.0  # return as percentage, consistent with Volatility_%


def cagr_from_values(series: pd.Series) -> float:
    """
    CAGR (decimal) from a portfolio-value series.
    Uses the number of observations minus 1 as the number of periods (years).
    """
    s = series.dropna().astype(float)
    if len(s) < 2:
        return np.nan
    return (s.iloc[-1] / s.iloc[0]) ** (1.0 / (len(s) - 1)) - 1.0


def mdd_from_values(series: pd.Series) -> float:
    """Max drawdown (%) from a portfolio *value* series (not returns)."""
    s = series.dropna().astype(float)
    if s.empty:
        return np.nan
    dd = s / s.cummax() - 1.0
    return float(dd.min() * 100.0)


# ═══════════════════════════════════════════════════════════════════════════
# Date / period helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_period_ends(
    start_year: int,
    end_year: int,
    frequency: int,
    unit: str,
) -> List[pd.Timestamp]:
    """
    Generate an ordered, deduplicated list of rebalancing boundary dates.

    The first date is always Dec 31 of (start_year - 1); the last is always
    Dec 31 of end_year.  Intermediate dates are determined by unit/frequency.

    Parameters
    ----------
    unit : 'months' | 'weeks' | 'days'
    """
    first = pd.Timestamp(f"{start_year - 1}-12-31")
    last  = pd.Timestamp(f"{end_year}-12-31")

    freq_map = {
        "months": f"{frequency}ME",
        "weeks":  f"{frequency}W-FRI",
        "days":   f"{frequency}D",
    }
    if unit not in freq_map:
        raise ValueError(f"unit must be one of {list(freq_map)}; got '{unit}'.")

    dates = [first] + list(pd.date_range(start=first, end=last, freq=freq_map[unit]))
    if dates[-1] != last:
        dates.append(last)
    return list(pd.Series(dates).drop_duplicates().sort_values())


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_etf_data(etfs: List[str]) -> dict:
    """
    Load ETF price CSVs from DATA_DIR.
    Expects one file per ticker: {ticker}_weekly_return_detailed.csv
    Returns {ticker: DataFrame} sorted by Date.
    """
    etf_data = {}
    for etf in etfs:
        path = os.path.join(DATA_DIR, f"{etf}_weekly_return_detailed.csv")
        df   = pd.read_csv(path, parse_dates=["Date"])
        df["Date"] = pd.to_datetime(df["Date"])
        etf_data[etf] = df.sort_values("Date").reset_index(drop=True)
    return etf_data


def load_luattruu(filename: str, start_year: int):
    """
    Load the LUATTRUU Bloomberg U.S. Treasury index from an Excel workbook.

    Returns
    -------
    lu_daily_pct : pd.Series  — daily percentage returns, indexed by Date
    lu_level     : pd.Series  — cumulative level series, indexed by Date

    Raises ImportError with an actionable message if openpyxl is missing.
    Raises ValueError if neither TR_DAILY nor PX_LAST column is present.
    """
    path = os.path.join(DATA_DIR, filename)
    try:
        lu = pd.read_excel(path, parse_dates=["Date"]).sort_values("Date")
    except ImportError as exc:
        raise ImportError(
            "Cannot read the LUATTRUU Excel file — 'openpyxl' is not installed. "
            "Run: pip install openpyxl"
        ) from exc

    # Keep one year of history before start_year for warm-up
    lu = lu[lu["Date"] >= pd.Timestamp(f"{start_year}-01-01") - pd.Timedelta(days=365)]
    lu.set_index("Date", inplace=True)

    if "TR_DAILY" in lu.columns and lu["TR_DAILY"].notna().any():
        lu_daily_pct = lu["TR_DAILY"].astype(float)
        lu_level     = (1.0 + lu_daily_pct.fillna(0.0) / 100.0).cumprod()
    elif "PX_LAST" in lu.columns:
        px           = lu["PX_LAST"].astype(float)
        lu_level     = px / float(px.iloc[0])
        lu_daily_pct = lu_level.pct_change() * 100.0
    else:
        raise ValueError(
            "LUATTRUU file must contain a 'TR_DAILY' or 'PX_LAST' column."
        )

    return lu_daily_pct, lu_level
