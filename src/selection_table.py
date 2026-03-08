"""
selection_table.py
Statistical Inference Module — publication-grade implementation.

Provides:
  1. Newey-West HAC t-test on period-level excess returns
  2. Block-bootstrap Sharpe difference test
  3. Convenience wrapper: semiannual_selection_table (ETF membership)

Implementation notes
--------------------
- HAC is applied to raw period-level returns, NOT to pre-annualized values.
- Annualization is performed AFTER inference (mean / CI are then scaled).
- Risk-free rate is subtracted at the period frequency before inference.
- Standard deviation uses ddof=1 (sample).
- Bootstrap uses circular block resampling to preserve autocorrelation.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple

from scipy.stats import norm

from src.config import RF_ANNUAL, REBALANCING_SETTINGS, ETFS, START_YEAR, END_YEAR
from src.utils import build_period_ends, load_etf_data


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

STRATS = ["Winners", "Median", "Losers"]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _periods_per_year(unit: str, freq: int) -> float:
    """Return the number of rebalancing periods in one calendar year."""
    return {"months": 12.0 / freq, "weeks": 52.0 / freq, "days": 252.0 / freq}[unit]


def _extract_period_returns(wealth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a wealth-level DataFrame into period-level returns.
    Expects columns: {strategy}_Net and BuyHold.
    Returns a DataFrame of pct_change() columns aligned on a common index.
    """
    cols = {s: wealth_df[f"{s}_Net"].pct_change() for s in STRATS if f"{s}_Net" in wealth_df}
    cols["BuyHold"] = wealth_df["BuyHold"].pct_change()
    return pd.DataFrame(cols).dropna()


# ═══════════════════════════════════════════════════════════════════════════
# 1. Newey-West HAC t-test
# ═══════════════════════════════════════════════════════════════════════════

def _newey_west_mean_test(
    x: np.ndarray,
    max_lag: int = None,
) -> Tuple[float, float, float, Tuple[float, float]]:
    """
    Newey-West HAC test of H0: mean(x) = 0.

    Returns
    -------
    mean, t_stat, p_value, (ci_lower, ci_upper)
    """
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan, (np.nan, np.nan)

    if max_lag is None:
        max_lag = max(1, int(n ** (1.0 / 3.0)))

    mu    = x.mean()
    resid = x - mu
    nw_var = float(np.dot(resid, resid) / n)
    for j in range(1, max_lag + 1):
        w       = 1.0 - j / (max_lag + 1.0)
        nw_var += 2.0 * w * float(np.dot(resid[j:], resid[:-j]) / n)

    se     = np.sqrt(max(nw_var, 0.0) / n)
    t_stat = mu / se if se > 0 else np.nan
    p_val  = 2.0 * (1.0 - norm.cdf(abs(t_stat)))
    return mu, t_stat, p_val, (mu - 1.96 * se, mu + 1.96 * se)


def run_newey_west_tests(
    frequencies=None,
    run_backtest_fn=None,
) -> pd.DataFrame:
    """
    Run Newey-West excess-return tests for all strategies × frequencies.

    Parameters
    ----------
    frequencies      : list of (name, unit, freq) — defaults to REBALANCING_SETTINGS
    run_backtest_fn  : callable(unit, freq) → (wealth_df, groups)
                       Must be supplied by the caller (avoids circular imports).

    Returns
    -------
    pd.DataFrame with columns:
      Frequency, Strategy, N, Mean_Excess_Ann_%, NW_t_stat,
      p_value, CI_95_Lower_%, CI_95_Upper_%, Significant_5%
    """
    if frequencies is None:
        frequencies = REBALANCING_SETTINGS
    if run_backtest_fn is None:
        raise ValueError("run_backtest_fn must be provided.")

    rows = []
    for name, unit, freq in frequencies:
        wealth_df, _ = run_backtest_fn(rebalancing_unit=unit, rebalancing_frequency=freq)
        returns      = _extract_period_returns(wealth_df)
        ppy          = _periods_per_year(unit, freq)

        for strat in STRATS:
            if strat not in returns or "BuyHold" not in returns:
                continue
            ex = (returns[strat] - returns["BuyHold"]).dropna().values
            mu_p, t, p_val, (lo_p, hi_p) = _newey_west_mean_test(ex)

            rows.append({
                "Frequency":        name,
                "Strategy":         strat,
                "N":                len(ex),
                "Mean_Excess_Ann_%": round(mu_p * ppy * 100, 4),
                "NW_t_stat":        round(t, 4),
                "p_value":          round(p_val, 4),
                "CI_95_Lower_%":    round(lo_p * ppy * 100, 4),
                "CI_95_Upper_%":    round(hi_p * ppy * 100, 4),
                "Significant_5%":   p_val < 0.05,
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Block-bootstrap Sharpe difference test
# ═══════════════════════════════════════════════════════════════════════════

def _sharpe_ratio(r: np.ndarray, rf_period: float) -> float:
    """Sample Sharpe ratio at the period frequency (ddof=1)."""
    excess = r - rf_period
    std    = np.std(excess, ddof=1)
    return np.mean(excess) / std if std > 0 else np.nan


def _block_bootstrap_sharpe_diff(
    strat_r: np.ndarray,
    bm_r: np.ndarray,
    rf_period: float,
    n_boot: int = 10_000,
    block_size: int = None,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """
    Circular block bootstrap test: H0: Sharpe(strategy) - Sharpe(benchmark) = 0.

    Returns
    -------
    observed_diff, p_value, ci_lower, ci_upper
    """
    rng = np.random.default_rng(seed)
    n   = len(strat_r)

    if block_size is None:
        block_size = max(2, int(n ** (1.0 / 3.0)))

    obs_diff = (_sharpe_ratio(strat_r, rf_period) -
                _sharpe_ratio(bm_r,    rf_period))

    boot_diffs = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, n)
            idx.extend((start + j) % n for j in range(block_size))
        idx = idx[:n]
        d = (_sharpe_ratio(strat_r[idx], rf_period) -
             _sharpe_ratio(bm_r[idx],    rf_period))
        boot_diffs.append(d)

    boot_diffs = np.array(boot_diffs)
    boot_diffs = boot_diffs[~np.isnan(boot_diffs)]
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    p_val = 2.0 * min((boot_diffs <= 0).mean(), (boot_diffs >= 0).mean())

    return float(obs_diff), float(p_val), float(ci_lo), float(ci_hi)


def run_bootstrap_sharpe_tests(
    frequencies=None,
    run_backtest_fn=None,
) -> pd.DataFrame:
    """
    Run block-bootstrap Sharpe difference tests for all strategies × frequencies.

    Parameters
    ----------
    frequencies      : list of (name, unit, freq)
    run_backtest_fn  : callable(unit, freq) → (wealth_df, groups)

    Returns
    -------
    pd.DataFrame with columns:
      Frequency, Strategy, Sharpe_Diff, CI_95_Lower, CI_95_Upper,
      p_value, Significant_5%
    """
    if frequencies is None:
        frequencies = REBALANCING_SETTINGS
    if run_backtest_fn is None:
        raise ValueError("run_backtest_fn must be provided.")

    rows = []
    for name, unit, freq in frequencies:
        wealth_df, _ = run_backtest_fn(rebalancing_unit=unit, rebalancing_frequency=freq)
        returns      = _extract_period_returns(wealth_df)
        ppy          = _periods_per_year(unit, freq)
        rf_period    = RF_ANNUAL / ppy

        for strat in STRATS:
            if strat not in returns or "BuyHold" not in returns:
                continue
            sr = returns[strat].dropna().values
            br = returns["BuyHold"].dropna().values
            obs, p_val, lo, hi = _block_bootstrap_sharpe_diff(sr, br, rf_period)

            rows.append({
                "Frequency":    name,
                "Strategy":     strat,
                "Sharpe_Diff":  round(obs,   4),
                "CI_95_Lower":  round(lo,    4),
                "CI_95_Upper":  round(hi,    4),
                "p_value":      round(p_val, 4),
                "Significant_5%": p_val < 0.05,
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Semi-annual ETF membership / selection table
# ═══════════════════════════════════════════════════════════════════════════

def semiannual_selection_table(
    etfs: List[str] = None,
    start_year: int = None,
    end_year: int = None,
    drop_years: List[int] = None,
) -> pd.DataFrame:
    """
    Build a semi-annual ETF membership table showing which ETFs were
    selected into Winners / Median / Losers at each rebalancing date.

    Parameters
    ----------
    etfs       : list of ticker symbols (defaults to config.ETFS)
    start_year : first backtest year (defaults to config.START_YEAR)
    end_year   : last backtest year  (defaults to config.END_YEAR)
    drop_years : list of years to exclude from the output (e.g. [START_YEAR])
                 because the first period has no prior-period ranking signal.

    Returns
    -------
    pd.DataFrame with index = rebalancing dates and columns per ETF showing
    the strategy bucket it was assigned to at that date.
    """
    if etfs is None:
        etfs = ETFS
    if start_year is None:
        start_year = START_YEAR
    if end_year is None:
        end_year = END_YEAR
    if drop_years is None:
        drop_years = []

    period_ends    = build_period_ends(start_year, end_year, 6, "months")
    report_periods = period_ends[1:]
    etf_data       = load_etf_data(etfs)

    # Per-period ETF returns
    returns = pd.DataFrame(index=report_periods, columns=etfs, dtype=float)
    for etf, df in etf_data.items():
        price = df.set_index("Date")["Adj Close"].astype(float).sort_index()
        for i in range(1, len(period_ends)):
            s_date, e_date = period_ends[i - 1], period_ends[i]
            try:
                returns.loc[e_date, etf] = (
                    price.loc[:e_date].iloc[-1] / price.loc[:s_date].iloc[-1] - 1.0
                )
            except (IndexError, KeyError):
                returns.loc[e_date, etf] = np.nan

    k = max(1, len(etfs) // 3)
    rows = []

    for i, end in enumerate(report_periods):
        if end.year in drop_years:
            continue
        if i == 0:
            sel = {g: etfs[:] for g in STRATS}
        else:
            ranked = returns.loc[report_periods[i - 1]].dropna().sort_values()
            sel = {
                "Losers":  list(ranked.index[:k]),
                "Median":  list(ranked.index[k:2*k]) if len(ranked) >= 2*k else list(ranked.index),
                "Winners": list(ranked.index[2*k:])  if len(ranked) >= 2*k else list(ranked.index),
            }

        row = {"Date": end}
        for g, members in sel.items():
            for etf in members:
                row[etf] = g
        rows.append(row)

    df_out = pd.DataFrame(rows).set_index("Date")
    return df_out[etfs]   # preserve original ETF column order
