"""
backtest.py
Core backtest engine for the Winner / Median / Loser Treasury ETF rotation strategy.

Produces a year-indexed annual summary DataFrame with the following columns
for each strategy (Winners, Median, Losers, BuyHold_2):

  <Strategy>              — portfolio wealth level at year-end
  Return_<Strategy>       — year-over-year return (%)
  Volatility_<Strategy>  — annualized volatility (%), from daily returns
  Sharpe_<Strategy>      — annualized Sharpe ratio, from daily returns
  MDD_<Strategy>         — max drawdown within the year (%), from daily returns
  Sortino_<Strategy>     — annualized Sortino ratio, from daily returns
  TE_<Strategy>          — annualized Tracking Error (%) vs LUATTRUU, daily
                           (0.0 for BuyHold_2 — TE vs itself is zero)
  IR_<Strategy>          — full-period annualized Information Ratio vs LUATTRUU,
                           computed over ALL holding periods (single scalar,
                           broadcast to every row). Using full-period avoids
                           inflated values from slicing to 1–2 obs per year.
                           (0.0 for BuyHold_2 — IR vs itself is zero)

Note on IR methodology
----------------------
IR is computed using period-level (holding-period) portfolio returns vs the
benchmark's return over each identical period — not daily ETF constituent
returns. This correctly captures the strategy's active return signal: how
much better/worse did the rotation decision perform each holding period vs
simply holding the benchmark.
"""
import os
import warnings
import numpy as np
import pandas as pd

from src.config import TABLES_DIR
from src.utils import (
    build_period_ends,
    load_etf_data,
    load_luattruu,
    annualized_stats_from_pct,
    compute_mdd_from_pct,
    sortino_from_pct,
    information_ratio_from_pct,
    tracking_error_from_pct,
)

warnings.filterwarnings("ignore")

# Ordered list of all strategy keys (strategies + benchmark)
_ALL_STRATS = ["Winners", "Median", "Losers", "BuyHold_2"]
_ROT_STRATS = ["Winners", "Median", "Losers"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _buyhold_values(
    dates: list,
    level: pd.Series,
    start_cash: float,
    base_date: pd.Timestamp,
) -> list:
    """
    Map a normalised level series to dollar wealth at each checkpoint date.

    Parameters
    ----------
    dates      : list of pd.Timestamp  — evaluation dates
    level      : pd.Series             — cumulative price level, indexed by Date
    start_cash : float                 — starting portfolio value
    base_date  : pd.Timestamp          — date on which the strategy is set to start_cash
    """
    ps   = level.sort_index()
    base = (
        float(ps.loc[:base_date].iloc[-1])
        if not ps.loc[:base_date].empty
        else float(ps.iloc[0])
    )
    norm = ps / base
    return [
        start_cash * float(norm.loc[:d].iloc[-1])
        if not norm.loc[:d].empty else np.nan
        for d in dates
    ]


def _year_stats(
    y: int,
    ret_panel: pd.DataFrame,
    periods: list,
    groups: dict,
    lu_col: str,
    rf_annual: float,
) -> dict:
    """
    Compute per-year risk metrics for all strategies and the benchmark
    using daily return series.

    Computes: vol, sharpe, mdd, sortino, tracking_error
    IR is intentionally excluded here — it is computed at the period level
    in run_etf_rotation() where full holding-period returns are available.

    Returns
    -------
    dict mapping strategy → (vol, sharpe, mdd, sortino, te)
    """
    stats = {}

    for g in _ROT_STRATS:
        segs = []
        for seg_start, seg_end, idx in periods:
            if seg_end.year != y:
                continue
            cols = groups[g][idx] if idx < len(groups[g]) else []
            if not cols:
                continue
            mask   = (ret_panel.index > seg_start) & (ret_panel.index <= seg_end)
            ew_pct = ret_panel[cols].loc[mask].mean(axis=1).dropna()
            if not ew_pct.empty:
                segs.append(ew_pct)

        if segs:
            s           = pd.concat(segs).sort_index()
            vol, sharpe = annualized_stats_from_pct(s, rf_ann=rf_annual)
            mdd         = compute_mdd_from_pct(s)
            sortino     = sortino_from_pct(s, rf_ann=rf_annual)
            lu_seg      = ret_panel[lu_col].reindex(s.index)
            te          = tracking_error_from_pct(s, lu_seg)
        else:
            vol = sharpe = mdd = sortino = te = np.nan

        stats[g] = (vol, sharpe, mdd, sortino, te)

    # Benchmark — TE vs itself = 0 by definition
    y_mask  = (ret_panel.index >= pd.Timestamp(f"{y}-01-01")) & \
              (ret_panel.index <= pd.Timestamp(f"{y}-12-31"))
    lu_year = ret_panel[lu_col].loc[y_mask].dropna()

    if not lu_year.empty:
        vol, sharpe = annualized_stats_from_pct(lu_year, rf_ann=rf_annual)
        mdd         = compute_mdd_from_pct(lu_year)
        sortino     = sortino_from_pct(lu_year, rf_ann=rf_annual)
    else:
        vol = sharpe = mdd = sortino = np.nan

    stats["BuyHold_2"] = (vol, sharpe, mdd, sortino, 0.0)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_etf_rotation(
    etfs: list,
    start_year: int,
    end_year: int,
    initial_cash: float,
    rebalancing_unit: str,
    rebalancing_frequency: int,
    luattruu_filename: str = "LUATTRUU_clean.xlsx",
    rf_annual: float = 0.0,
    verbose: bool = True,
    save_csv: bool = True,
):
    """
    Bond ETF rotation backtest: Winners / Median / Losers vs LUATTRUU.

    At every rebalancing date the ETFs are ranked by their prior-period
    return (ascending).  Bottom tercile → Losers; middle → Median;
    top → Winners.  Each portfolio is equal-weighted within its tercile.
    The first period uses all ETFs equally (no look-ahead).

    Parameters
    ----------
    etfs                  : list[str]  — ticker symbols
    start_year, end_year  : int
    initial_cash          : float      — starting portfolio value (e.g. 100)
    rebalancing_unit      : str        — 'months' | 'weeks' | 'days'
    rebalancing_frequency : int        — e.g. 12 for annual, 6 for semi-annual
    luattruu_filename     : str        — Excel file in data/
    rf_annual             : float      — annualized risk-free rate (decimal)
    verbose               : bool       — print annual summary table
    save_csv              : bool       — write CSV to results/tables/

    Returns
    -------
    annual_df : pd.DataFrame  — year-indexed summary (see module docstring)
    groups    : dict          — per-period ETF selections per strategy
    """
    # ── Rebalancing dates ──────────────────────────────────────────────────
    period_ends    = build_period_ends(start_year, end_year, rebalancing_frequency, rebalancing_unit)
    report_periods = period_ends[1:]

    # ── ETF price data ─────────────────────────────────────────────────────
    etf_data = load_etf_data(etfs)

    # ── Per-period ETF returns ─────────────────────────────────────────────
    returns = pd.DataFrame(index=report_periods, columns=etfs, dtype=float)
    for etf, df in etf_data.items():
        price = df.set_index("Date")["Adj Close"].astype(float).sort_index()
        for i in range(1, len(period_ends)):
            s_date, e_date = period_ends[i - 1], period_ends[i]
            try:
                p0 = price.loc[:s_date].iloc[-1]
                p1 = price.loc[:e_date].iloc[-1]
                returns.loc[e_date, etf] = p1 / p0 - 1.0
            except (IndexError, KeyError):
                returns.loc[e_date, etf] = np.nan

    # ── Rotation simulation (no look-ahead) ───────────────────────────────
    k      = max(1, len(etfs) // 3)
    cash   = {g: initial_cash for g in _ROT_STRATS}
    groups = {g: [] for g in _ROT_STRATS}
    wealth = {g: [] for g in _ROT_STRATS}
    dates  = []

    for i, end in enumerate(report_periods):
        if i == 0:
            sel = {g: etfs[:] for g in _ROT_STRATS}
        else:
            ranked = returns.loc[report_periods[i - 1]].dropna().sort_values()
            if ranked.empty:
                sel = {g: etfs[:] for g in _ROT_STRATS}
            else:
                sel = {
                    "Losers":  list(ranked.index[:k]),
                    "Median":  list(ranked.index[k:2*k]) if len(ranked) >= 2*k else list(ranked.index),
                    "Winners": list(ranked.index[2*k:])  if len(ranked) >= 2*k else list(ranked.index),
                }

        for g in _ROT_STRATS:
            groups[g].append(sel[g])
            vals = returns.loc[end, sel[g]].dropna()
            cash[g] *= 1.0 + (float(vals.mean()) if not vals.empty else 0.0)
            wealth[g].append(cash[g])
        dates.append(end)

    # ── LUATTRUU buy-and-hold ──────────────────────────────────────────────
    lu_daily_pct, lu_level = load_luattruu(luattruu_filename, start_year)
    bh_vals = _buyhold_values(dates, lu_level, initial_cash, period_ends[0])

    # ── Wealth DataFrame ───────────────────────────────────────────────────
    df_all = pd.DataFrame(
        {**{g: wealth[g] for g in _ROT_STRATS}, "BuyHold_2": bh_vals},
        index=pd.DatetimeIndex(dates),
    )

    # ── Daily return panel for risk metrics ───────────────────────────────
    ret_panel = pd.DataFrame()
    for etf, df in etf_data.items():
        px = (
            df[["Date", "Adj Close"]].copy()
              .assign(Date=lambda d: pd.to_datetime(d["Date"]))
              .set_index("Date")["Adj Close"]
              .pct_change() * 100.0
        )
        ret_panel = ret_panel.join(px.rename(etf), how="outer")
    ret_panel["LUATTRUU"] = lu_daily_pct

    # ── Per-period index for yearly risk slicing ───────────────────────────
    periods_idx = [
        (period_ends[i], period_ends[i + 1], i)
        for i in range(len(period_ends) - 1)
    ]

    # ── Annual summary ─────────────────────────────────────────────────────
    rows = []
    for y in range(start_year, end_year + 1):
        yr = df_all[df_all.index.year == y]
        if not yr.empty:
            rows.append(yr.iloc[-1])
    annual_df = pd.DataFrame(rows)
    annual_df.index = list(range(start_year, start_year + len(annual_df)))
    annual_df.index.name = "Year"

    # ── Daily-based risk metrics (vol, sharpe, mdd, sortino, te) ──────────
    for y in annual_df.index:
        ys = _year_stats(y, ret_panel, periods_idx, groups, "LUATTRUU", rf_annual)
        for g in _ALL_STRATS:
            vol, sharpe, mdd, sortino, te = ys[g]
            annual_df.loc[y, f"Volatility_{g}"] = vol
            annual_df.loc[y, f"Sharpe_{g}"]     = sharpe
            annual_df.loc[y, f"MDD_{g}"]         = mdd
            annual_df.loc[y, f"Sortino_{g}"]     = sortino
            annual_df.loc[y, f"TE_{g}"]          = te

    # ── Year-over-year returns ─────────────────────────────────────────────
    for g in _ALL_STRATS:
        annual_df[f"Return_{g}"] = annual_df[g].pct_change() * 100.0

    # ── Full-sample IR (holding-period portfolio returns vs benchmark) ────────
    # IR is a single full-period statistic — computing it on per-year slices
    # produces only 1–2 data points per year, inflating the ratio.
    # We compute once over ALL holding periods, then broadcast to every row.
    period_rets_bh = df_all["BuyHold_2"].pct_change() * 100.0  # % units

    for g in _ROT_STRATS:
        period_rets_g = df_all[g].pct_change() * 100.0  # % units
        ir_full = information_ratio_from_pct(
            period_rets_g.dropna(), period_rets_bh.dropna()
        )
        annual_df[f"IR_{g}"] = ir_full   # same scalar broadcast to all year rows

    # BuyHold IR vs itself = 0 by definition
    annual_df["IR_BuyHold_2"] = 0.0

    # ── Verbose output ─────────────────────────────────────────────────────
    if verbose:
        label = f"{rebalancing_frequency} {rebalancing_unit.upper()}"
        print(f"\n{'='*80}")
        print(f"ANNUAL SUMMARY — Rebalancing every {label}")
        print("=" * 80)
        show = ["Winners", "Median", "Losers", "BuyHold_2",
                "Return_Winners", "Return_Median", "Return_Losers", "Return_BuyHold_2"]
        print(annual_df[[c for c in show if c in annual_df.columns]].to_string())

    # ── Save CSV ───────────────────────────────────────────────────────────
    if save_csv:
        os.makedirs(TABLES_DIR, exist_ok=True)
        fname = f"annual_summary_{rebalancing_frequency}{rebalancing_unit[0]}.csv"
        annual_df.to_csv(os.path.join(TABLES_DIR, fname))
        if verbose:
            print(f"  → Saved: results/tables/{fname}")

    return annual_df, groups