"""
Transaction Cost Analysis for the U.S. Treasury ETF Rotation Strategy.

This module extends the base backtest with:
  1. Turnover tracking at each rebalancing date
  2. Net-of-cost wealth paths under configurable cost assumptions
  3. Net-of-cost risk metrics (Sharpe, volatility, MDD)
  4. Cost-as-percentage-of-alpha analysis
  5. Scenario analysis (low / base / high cost)
  6. Break-even cost computation (with extrapolation)
  7. Cross-frequency cost comparison
  8. Summary tables and visualisations saved to results/transaction_costs/
     Tables (8 CSV): annual_summary_with_costs, turnover_summary, etf_spreads,
                     net_risk_metrics, scenario_comparison, cost_as_pct_of_alpha,
                     cross_frequency_cost, and one per-strategy run via compute_breakeven_cost
     Charts (7 PNG): net_wealth_combined, cumulative_cost_drag, turnover_grouped_bars,
                     scenario_comparison_cagr, breakeven_median, breakeven_all_strategies,
                     cross_frequency_cost
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from src.config import (
    ETFS, START_YEAR, END_YEAR, INITIAL_CASH, RF_ANNUAL,
    LUATTRUU_FILENAME, COLORS, TABLES_DIR, VIZ_DIR,
    REBALANCING_SETTINGS,
)
from src.utils import (
    build_period_ends, load_etf_data, load_luattruu,
    compute_mdd_from_pct, annualized_stats_from_pct,
    cagr_from_values, mdd_from_values, infer_ann_factor,
)

# ── Output directories ────────────────────────────────────────────────────
TC_DIR = os.path.join(os.path.dirname(TABLES_DIR), "transaction_costs")
TC_TABLES = os.path.join(TC_DIR, "tables")
TC_VIZ = os.path.join(TC_DIR, "visualizations")

for _d in [TC_DIR, TC_TABLES, TC_VIZ]:
    os.makedirs(_d, exist_ok=True)

# ── Empirical bid-ask half-spreads (bps, one-way) — Bloomberg/iShares ─────
ETF_SPREADS_BPS = {
    "SHV": 0.5,   # extremely liquid, ultra-short
    "SHY": 1.0,   # very liquid, short duration
    "IEI": 1.5,   # liquid, intermediate
    "IEF": 2.0,   # liquid, intermediate-long
    "TLH": 3.0,   # moderately liquid, long
    "TLT": 2.5,   # highly traded, long duration
}

_avg_spread = np.mean(list(ETF_SPREADS_BPS.values()))
BASE_ROUND_TRIP_BPS = round(2 * _avg_spread, 1)        # ≈ 3.5 bps

SCENARIO_MULTIPLIERS = {
    "Low (0.5×)": 0.5,
    "Base (1×)":  1.0,
    "High (2×)":  2.0,
}

STRAT_COLORS = {"Winners": "#008000", "Median": "#000000", "Losers": "#0000FF"}
STRATS = ["Winners", "Median", "Losers"]


# ══════════════════════════════════════════════════════════════════════════
# 1. Core backtest engine with transaction costs
# ══════════════════════════════════════════════════════════════════════════

def _compute_turnover(prev_sel: List[str], curr_sel: List[str]) -> float:
    """Fraction of the portfolio that turns over (0.0 / 0.5 / 1.0)."""
    if not prev_sel:
        return 0.0
    changed = len(set(prev_sel).symmetric_difference(set(curr_sel))) / 2
    return changed / max(len(curr_sel), 1)


def _buyhold_value_at(dates, level_series, start_cash, base_date=None):
    ps = level_series.sort_index()
    if not dates or ps.empty:
        return []
    if base_date is None:
        base_date = dates[0]
    base = ps.loc[:base_date].iloc[-1] if not ps.loc[:base_date].empty else ps.iloc[0]
    norm = ps / float(base)
    return [
        start_cash * float(norm.loc[:d].iloc[-1]) if not norm.loc[:d].empty else np.nan
        for d in dates
    ]


def run_backtest_with_costs(
    etfs: List[str] = ETFS,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    initial_cash: float = INITIAL_CASH,
    rebalancing_unit: str = "months",
    rebalancing_frequency: int = 6,
    cost_bps: float = BASE_ROUND_TRIP_BPS,
    rf_annual: float = RF_ANNUAL,
    signal_lag: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the rotation backtest with transaction-cost deduction.

    Parameters
    ----------
    signal_lag : int, default 0
        Number of periods to delay the ranking signal. 0 = base case (rank
        on the immediately preceding period). 1 = skip-period (rank on the
        period *before* the preceding one), testing execution-lag robustness.

    Returns (wealth_df, turnover_df).
    """
    period_ends = build_period_ends(start_year, end_year, rebalancing_frequency, rebalancing_unit)
    report_periods = period_ends[1:]
    etf_data = load_etf_data(etfs)

    # Period returns
    returns = pd.DataFrame(index=report_periods, columns=etfs, dtype=float)
    for etf, df in etf_data.items():
        price = df.set_index("Date")["Adj Close"].astype(float).sort_index()
        for i in range(1, len(period_ends)):
            s, e = period_ends[i - 1], period_ends[i]
            try:
                p0 = price.loc[:s].iloc[-1]
                p1 = price.loc[:e].iloc[-1]
                returns.loc[e, etf] = (p1 / p0) - 1.0
            except Exception:
                returns.loc[e, etf] = np.nan

    N = len(etfs)
    k = max(1, N // 3)
    cost_frac = cost_bps / 10_000.0

    cash_gross = {s: initial_cash for s in STRATS}
    cash_net = {s: initial_cash for s in STRATS}
    prev_sel = {s: [] for s in STRATS}

    records, turnover_records = [], []

    for idx_i, end in enumerate(report_periods):
        # Determine which period to use for ranking signal
        rank_idx = idx_i - 1 - signal_lag
        if idx_i == 0 or rank_idx < 0:
            w_sel = m_sel = l_sel = etfs[:]
        else:
            prev_row = returns.loc[report_periods[rank_idx]].dropna().sort_values()
            if prev_row.empty:
                w_sel = m_sel = l_sel = etfs[:]
            else:
                l_sel = list(prev_row.index[:k])
                m_sel = list(prev_row.index[k:2*k]) if len(prev_row) >= 2*k else list(prev_row.index)
                w_sel = list(prev_row.index[2*k:]) if len(prev_row) >= 2*k else list(prev_row.index)

        sel_map = {"Winners": w_sel, "Median": m_sel, "Losers": l_sel}
        r_now = returns.loc[end]

        for strat in STRATS:
            sel = sel_map[strat]
            to = _compute_turnover(prev_sel[strat], sel)
            vals = r_now[sel].dropna()
            period_ret = float(vals.mean()) if not vals.empty else 0.0

            cash_gross[strat] *= (1.0 + period_ret)
            tc_dollar = cash_net[strat] * to * cost_frac
            cash_net[strat] -= tc_dollar
            cash_net[strat] *= (1.0 + period_ret)

            turnover_records.append({
                "Period": end, "Strategy": strat,
                "Turnover": to, "Cost_Dollar": tc_dollar,
                "Cost_Bps_Effective": to * cost_bps,
                "Selection": ", ".join(sel),
            })
            prev_sel[strat] = sel

        records.append({
            "Period": end,
            **{f"{s}_Gross": cash_gross[s] for s in STRATS},
            **{f"{s}_Net": cash_net[s] for s in STRATS},
        })

    wealth_df = pd.DataFrame(records).set_index("Period")
    lu_daily_pct, lu_level = load_luattruu(LUATTRUU_FILENAME, start_year)
    wealth_df["BuyHold"] = _buyhold_value_at(
        list(wealth_df.index), lu_level, initial_cash, period_ends[0]
    )

    return wealth_df, pd.DataFrame(turnover_records)


# ══════════════════════════════════════════════════════════════════════════
# 2. Scenario & break-even analysis
# ══════════════════════════════════════════════════════════════════════════

def run_scenario_analysis(rebalancing_unit="months", rebalancing_frequency=6):
    results = {}
    for label, mult in SCENARIO_MULTIPLIERS.items():
        wdf, _ = run_backtest_with_costs(
            rebalancing_unit=rebalancing_unit,
            rebalancing_frequency=rebalancing_frequency,
            cost_bps=BASE_ROUND_TRIP_BPS * mult,
        )
        results[label] = wdf
    return results


def compute_breakeven_cost(
    strategy: str = "Median",
    rebalancing_unit: str = "months",
    rebalancing_frequency: int = 6,
    search_max_bps: float = 200.0,
    step_bps: float = 0.5,
) -> Tuple[float, pd.DataFrame]:
    """
    Find the round-trip cost (bps) at which strategy net CAGR = Buy & Hold CAGR.
    Uses linear extrapolation if the crossing falls beyond search_max_bps.
    """
    costs = np.arange(0, search_max_bps + step_bps, step_bps)
    rows = []
    for c in costs:
        wdf, _ = run_backtest_with_costs(
            rebalancing_unit=rebalancing_unit,
            rebalancing_frequency=rebalancing_frequency,
            cost_bps=c,
        )
        sv = wdf[f"{strategy}_Net"].dropna()
        bv = wdf["BuyHold"].dropna()
        ny = (sv.index[-1] - sv.index[0]).days / 365.25
        if ny <= 0:
            continue
        cs = (sv.iloc[-1] / sv.iloc[0]) ** (1 / ny) - 1
        cb = (bv.iloc[-1] / bv.iloc[0]) ** (1 / ny) - 1
        rows.append({
            "Cost_bps": c,
            f"CAGR_{strategy}_Net": cs * 100,
            "CAGR_BuyHold": cb * 100,
            "Excess_Return_pct": (cs - cb) * 100,
        })

    curve_df = pd.DataFrame(rows)
    excess = curve_df["Excess_Return_pct"].values

    # Look for actual crossing within the scanned range
    breakeven = np.nan
    for i in range(1, len(excess)):
        if excess[i - 1] >= 0 and excess[i] < 0:
            c0, c1 = curve_df["Cost_bps"].iloc[i - 1], curve_df["Cost_bps"].iloc[i]
            e0, e1 = excess[i - 1], excess[i]
            breakeven = c0 + (0 - e0) * (c1 - c0) / (e1 - e0)
            break

    # If no crossing found, extrapolate linearly from the curve endpoints
    if np.isnan(breakeven) and len(curve_df) >= 2:
        e0 = curve_df["Excess_Return_pct"].iloc[0]
        e1 = curve_df["Excess_Return_pct"].iloc[-1]
        c0 = curve_df["Cost_bps"].iloc[0]
        c1 = curve_df["Cost_bps"].iloc[-1]
        if e0 > 0 and e0 != e1:
            slope = (e1 - e0) / (c1 - c0)
            breakeven = c0 - e0 / slope

    return breakeven, curve_df


# ══════════════════════════════════════════════════════════════════════════
# 3. Summary tables
# ══════════════════════════════════════════════════════════════════════════

def build_annual_summary_with_costs(wealth_df, turnover_df):
    """Year-end wealth (gross/net), drag %, turnover, and YoY net returns."""
    wc = wealth_df.copy()
    wc["Year"] = wc.index.year
    rows = []
    for y in sorted(wc["Year"].unique()):
        yr = wc[wc["Year"] == y]
        if yr.empty:
            continue
        last = yr.iloc[-1]
        row = {"Year": y}
        for s in STRATS:
            row[f"{s}_Gross"] = last[f"{s}_Gross"]
            row[f"{s}_Net"] = last[f"{s}_Net"]
            row[f"{s}_Drag_pct"] = (
                (last[f"{s}_Gross"] - last[f"{s}_Net"]) / last[f"{s}_Gross"] * 100
                if last[f"{s}_Gross"] > 0 else np.nan
            )
        row["BuyHold"] = last["BuyHold"]
        yr_to = turnover_df[pd.to_datetime(turnover_df["Period"]).dt.year == y]
        for s in STRATS:
            st = yr_to[yr_to["Strategy"] == s]
            row[f"{s}_Turnover"] = st["Turnover"].sum()
            row[f"{s}_Cost_Dollar"] = st["Cost_Dollar"].sum()
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Year")
    for s in STRATS + ["BuyHold"]:
        col = f"{s}_Net" if s != "BuyHold" else "BuyHold"
        yoy = []
        for i in range(len(df)):
            if i == 0:
                yoy.append(np.nan)
            else:
                p, c = float(df[col].iloc[i - 1]), float(df[col].iloc[i])
                yoy.append((c / p - 1) * 100 if p > 0 else np.nan)
        df[f"Return_{s}_Net" if s != "BuyHold" else "Return_BuyHold"] = yoy
    return df


def build_turnover_summary(turnover_df):
    """Aggregate turnover statistics per strategy."""
    rows = []
    for s in STRATS:
        sdf = turnover_df[turnover_df["Strategy"] == s]
        n = len(sdf)
        nt = (sdf["Turnover"] > 0).sum()
        rows.append({
            "Strategy": s,
            "Total_Periods": n,
            "Periods_With_Trades": nt,
            "Trade_Frequency_pct": round(nt / n * 100, 1) if n > 0 else 0,
            "Avg_Turnover_Per_Period": round(sdf["Turnover"].mean() * 100, 1),
            "Cumulative_Turnover": round(sdf["Turnover"].sum() * 100, 1),
            "Total_Cost_Dollar": round(sdf["Cost_Dollar"].sum(), 4),
        })
    return pd.DataFrame(rows).set_index("Strategy")


def build_etf_spread_table():
    rows = []
    for etf, hs in ETF_SPREADS_BPS.items():
        rows.append({"ETF": etf, "Half_Spread_bps": hs, "Round_Trip_bps": 2 * hs})
    rows.append({
        "ETF": "Average",
        "Half_Spread_bps": round(_avg_spread, 1),
        "Round_Trip_bps": BASE_ROUND_TRIP_BPS,
    })
    return pd.DataFrame(rows)


def build_scenario_comparison_table(scenario_results):
    rows = []
    for label, wdf in scenario_results.items():
        ny = (wdf.index[-1] - wdf.index[0]).days / 365.25
        row = {"Scenario": label}
        for s in STRATS:
            fg, fn = wdf[f"{s}_Gross"].iloc[-1], wdf[f"{s}_Net"].iloc[-1]
            row[f"{s}_Final_Gross"] = round(fg, 2)
            row[f"{s}_Final_Net"] = round(fn, 2)
            row[f"{s}_CAGR_Gross"] = round(((fg / INITIAL_CASH) ** (1 / ny) - 1) * 100, 2)
            row[f"{s}_CAGR_Net"] = round(((fn / INITIAL_CASH) ** (1 / ny) - 1) * 100, 2)
            row[f"{s}_Cost_Drag"] = round(fg - fn, 2)
        bh = wdf["BuyHold"].iloc[-1]
        row["BuyHold_Final"] = round(bh, 2)
        row["BuyHold_CAGR"] = round(((bh / INITIAL_CASH) ** (1 / ny) - 1) * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Scenario")


# ── NEW TABLE: Net-of-cost risk metrics (Sharpe, Vol, MDD) ───────────────

def build_net_risk_metrics_table(wealth_df):
    """
    Full risk metrics for each strategy (gross & net) and the benchmark.
    Computed from semi-annual period returns.
    """
    ny = (wealth_df.index[-1] - wealth_df.index[0]).days / 365.25
    rows = []
    for label, col_g, col_n in [
        ("Winners", "Winners_Gross", "Winners_Net"),
        ("Median",  "Median_Gross",  "Median_Net"),
        ("Losers",  "Losers_Gross",  "Losers_Net"),
    ]:
        for suffix, col in [("Gross", col_g), ("Net", col_n)]:
            vals = wealth_df[col].dropna().astype(float)
            cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / ny) - 1
            pct_ret = vals.pct_change().dropna() * 100
            vol, sharpe = annualized_stats_from_pct(pct_ret, rf_ann=RF_ANNUAL)
            mdd = mdd_from_values(vals)
            rows.append({
                "Strategy": f"{label} ({suffix})",
                "Final_Value": round(vals.iloc[-1], 2),
                "CAGR_pct": round(cagr * 100, 2),
                "Volatility_pct": round(vol, 2) if not np.isnan(vol) else np.nan,
                "Sharpe_Ratio": round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
                "Max_Drawdown_pct": round(mdd, 2) if not np.isnan(mdd) else np.nan,
            })

    bh = wealth_df["BuyHold"].dropna().astype(float)
    cagr_bh = (bh.iloc[-1] / bh.iloc[0]) ** (1 / ny) - 1
    pct_bh = bh.pct_change().dropna() * 100
    vol_bh, sh_bh = annualized_stats_from_pct(pct_bh, rf_ann=RF_ANNUAL)
    mdd_bh = mdd_from_values(bh)
    rows.append({
        "Strategy": "Buy & Hold",
        "Final_Value": round(bh.iloc[-1], 2),
        "CAGR_pct": round(cagr_bh * 100, 2),
        "Volatility_pct": round(vol_bh, 2) if not np.isnan(vol_bh) else np.nan,
        "Sharpe_Ratio": round(sh_bh, 3) if not np.isnan(sh_bh) else np.nan,
        "Max_Drawdown_pct": round(mdd_bh, 2) if not np.isnan(mdd_bh) else np.nan,
    })
    return pd.DataFrame(rows).set_index("Strategy")


# ── NEW TABLE: Cost as % of alpha ─────────────────────────────────────────

def build_cost_alpha_table(wealth_df, scenario_results):
    """
    For each strategy × scenario: gross alpha over benchmark,
    cost drag on CAGR, and what % of alpha is consumed by costs.
    """
    ny = (wealth_df.index[-1] - wealth_df.index[0]).days / 365.25
    bh_cagr = ((wealth_df["BuyHold"].iloc[-1] / INITIAL_CASH) ** (1 / ny) - 1) * 100

    rows = []
    for label, wdf in scenario_results.items():
        for s in STRATS:
            fg = wdf[f"{s}_Gross"].iloc[-1]
            fn = wdf[f"{s}_Net"].iloc[-1]
            cg = ((fg / INITIAL_CASH) ** (1 / ny) - 1) * 100
            cn = ((fn / INITIAL_CASH) ** (1 / ny) - 1) * 100
            gross_alpha = cg - bh_cagr
            cost_drag = cg - cn
            pct_consumed = (cost_drag / abs(gross_alpha) * 100) if gross_alpha != 0 else np.nan
            rows.append({
                "Scenario": label, "Strategy": s,
                "CAGR_Gross_pct": round(cg, 3),
                "CAGR_Net_pct": round(cn, 3),
                "CAGR_BuyHold_pct": round(bh_cagr, 3),
                "Gross_Alpha_pct": round(gross_alpha, 3),
                "Cost_Drag_CAGR_pct": round(cost_drag, 4),
                "Cost_as_pct_of_Alpha": round(pct_consumed, 1) if not np.isnan(pct_consumed) else "N/A",
                "Net_Alpha_pct": round(cn - bh_cagr, 3),
            })
    return pd.DataFrame(rows)


# ── NEW TABLE: Cross-frequency cost comparison ────────────────────────────

def build_cross_frequency_cost_table():
    """
    Run the Median strategy across ALL rebalancing frequencies with base costs.
    Shows how turnover and cost drag scale with frequency.
    """
    rows = []
    for name, unit, freq in REBALANCING_SETTINGS:
        try:
            wdf, tdf = run_backtest_with_costs(
                rebalancing_unit=unit, rebalancing_frequency=freq,
                cost_bps=BASE_ROUND_TRIP_BPS,
            )
        except Exception:
            continue
        ny = (wdf.index[-1] - wdf.index[0]).days / 365.25
        if ny <= 0:
            continue
        mt = tdf[tdf["Strategy"] == "Median"]
        fg = wdf["Median_Gross"].iloc[-1]
        fn = wdf["Median_Net"].iloc[-1]
        bh = wdf["BuyHold"].iloc[-1]
        cg = ((fg / INITIAL_CASH) ** (1 / ny) - 1) * 100
        cn = ((fn / INITIAL_CASH) ** (1 / ny) - 1) * 100
        cbh = ((bh / INITIAL_CASH) ** (1 / ny) - 1) * 100
        rows.append({
            "Frequency": name,
            "Rebalancing_Periods": len(mt),
            "Periods_With_Trades": int((mt["Turnover"] > 0).sum()),
            "Cumulative_Turnover_pct": round(mt["Turnover"].sum() * 100, 0),
            "Total_Cost_Dollar": round(mt["Cost_Dollar"].sum(), 2),
            "CAGR_Gross_pct": round(cg, 2),
            "CAGR_Net_pct": round(cn, 2),
            "Cost_Drag_CAGR_bps": round((cg - cn) * 100, 1),
            "Net_vs_BuyHold_pct": round(cn - cbh, 2),
            "Beats_BuyHold": "Yes" if cn > cbh else "No",
        })
    return pd.DataFrame(rows).set_index("Frequency")


# ══════════════════════════════════════════════════════════════════════════
# 4. Visualizations
# ══════════════════════════════════════════════════════════════════════════

def plot_combined_net_wealth(wealth_df, save=True):
    """Single chart: all three net-of-cost strategies + Buy & Hold."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in STRATS:
        ax.plot(wealth_df.index, wealth_df[f"{s}_Net"],
                label=f"{s} (Net)", color=STRAT_COLORS[s], linewidth=2)
    ax.plot(wealth_df.index, wealth_df["BuyHold"],
            label="Buy & Hold", color=COLORS["BuyHold_2"], linewidth=2, linestyle="--")
    ax.set_title("Net-of-Cost Portfolio Value — Semi-Annual Rebalancing", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(TC_VIZ, "net_wealth_combined.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)





def plot_cumulative_cost_drag(wealth_df, save=True):
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in STRATS:
        drag = wealth_df[f"{s}_Gross"] - wealth_df[f"{s}_Net"]
        ax.plot(wealth_df.index, drag, label=s, color=STRAT_COLORS[s], linewidth=2)
    ax.set_title("Cumulative Transaction Cost Drag — Semi-Annual Rebalancing", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cost Drag ($)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(TC_VIZ, "cumulative_cost_drag.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_turnover_grouped(turnover_df, save=True):
    """Single grouped-bar chart (replaces the old 3-panel layout)."""
    pivot = turnover_df.pivot(index="Period", columns="Strategy", values="Turnover")
    pivot = pivot[STRATS] * 100
    fig, ax = plt.subplots(figsize=(14, 5))
    periods = pivot.index
    x = np.arange(len(periods))
    bw = 0.25
    for i, s in enumerate(STRATS):
        ax.bar(x + (i - 1) * bw, pivot[s].values, bw,
               label=s, color=STRAT_COLORS[s], alpha=0.85)
    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(d.date()) for d in periods[::2]], rotation=45, fontsize=8)
    ax.set_ylabel("Turnover (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Portfolio Turnover at Each Rebalancing Date — Semi-Annual", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(TC_VIZ, "turnover_grouped_bars.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_scenario_comparison(scenario_results, save=True):
    """Scenario comparison using CAGR (not final value) with value labels."""
    labels = list(scenario_results.keys())
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, strat in enumerate(STRATS):
        vals = []
        for l in labels:
            wdf = scenario_results[l]
            ny = (wdf.index[-1] - wdf.index[0]).days / 365.25
            vals.append(((wdf[f"{strat}_Net"].iloc[-1] / INITIAL_CASH) ** (1 / ny) - 1) * 100)
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=strat, color=STRAT_COLORS[strat])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=8)

    # Buy & Hold reference line
    wdf0 = list(scenario_results.values())[0]
    ny = (wdf0.index[-1] - wdf0.index[0]).days / 365.25
    bh_cagr = ((wdf0["BuyHold"].iloc[-1] / INITIAL_CASH) ** (1 / ny) - 1) * 100
    ax.axhline(bh_cagr, color=COLORS["BuyHold_2"], linewidth=2, linestyle="--",
               label=f"Buy & Hold ({bh_cagr:.2f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CAGR (%)")
    ax.set_title("Net CAGR Under Cost Scenarios — Semi-Annual Rebalancing", fontsize=13)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(TC_VIZ, "scenario_comparison_cagr.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_breakeven_curve(breakeven_bps, curve_df, strategy="Median", save=True):
    fig, ax = plt.subplots(figsize=(12, 6))
    sc = f"CAGR_{strategy}_Net"

    ax.plot(curve_df["Cost_bps"], curve_df[sc],
            label=f"{strategy} Net CAGR", color=STRAT_COLORS.get(strategy, "black"), linewidth=2)
    ax.plot(curve_df["Cost_bps"], curve_df["CAGR_BuyHold"],
            label="Buy & Hold CAGR", color=COLORS["BuyHold_2"], linewidth=2, linestyle="--")

    ax.fill_between(curve_df["Cost_bps"], curve_df[sc], curve_df["CAGR_BuyHold"],
                    where=curve_df[sc] >= curve_df["CAGR_BuyHold"],
                    alpha=0.15, color="green", label="Strategy advantage")
    ax.fill_between(curve_df["Cost_bps"], curve_df[sc], curve_df["CAGR_BuyHold"],
                    where=curve_df[sc] < curve_df["CAGR_BuyHold"],
                    alpha=0.15, color="red", label="Buy & Hold advantage")

    bh_cagr = curve_df["CAGR_BuyHold"].iloc[0]
    if not np.isnan(breakeven_bps):
        if breakeven_bps <= curve_df["Cost_bps"].max():
            ax.axvline(breakeven_bps, color="gray", linewidth=1.5, linestyle=":")
            ax.annotate(f"Break-even: {breakeven_bps:.0f} bps",
                        xy=(breakeven_bps, bh_cagr),
                        xytext=(min(breakeven_bps + 10, curve_df["Cost_bps"].max() - 15),
                                bh_cagr + 0.3),
                        fontsize=11, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="gray"))
        else:
            ax.annotate(f"Break-even ≈ {breakeven_bps:.0f} bps\n(extrapolated beyond scan range)",
                        xy=(curve_df["Cost_bps"].max() * 0.75, bh_cagr + 0.15),
                        fontsize=11, fontweight="bold", color="gray",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Mark base cost
    ax.axvline(BASE_ROUND_TRIP_BPS, color="blue", linewidth=1, linestyle="--", alpha=0.5)
    ax.annotate(f"Base cost\n({BASE_ROUND_TRIP_BPS} bps)",
                xy=(BASE_ROUND_TRIP_BPS, curve_df[sc].iloc[0] - 0.2),
                fontsize=9, color="blue", ha="center")

    ax.set_xlabel("Round-Trip Transaction Cost (bps)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title(f"Break-Even Analysis: {strategy} Strategy vs. Buy & Hold", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(TC_VIZ, f"breakeven_{strategy.lower()}.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_breakeven_all_strategies(save=True):
    """Break-even curves for all three strategies on one chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    bh_cagr = None

    for strat in STRATS:
        be, curve = compute_breakeven_cost(strategy=strat, search_max_bps=150, step_bps=1.0)
        if bh_cagr is None:
            bh_cagr = curve["CAGR_BuyHold"].iloc[0]

        if not np.isnan(be) and be <= 150:
            bel = f"{be:.0f} bps"
        elif not np.isnan(be):
            bel = f"≈{be:.0f} bps"
        else:
            bel = ">200 bps"

        ax.plot(curve["Cost_bps"], curve[f"CAGR_{strat}_Net"],
                label=f"{strat} (BE={bel})", color=STRAT_COLORS[strat], linewidth=2)
        if not np.isnan(be) and be <= 150:
            ax.plot(be, bh_cagr, "o", color=STRAT_COLORS[strat], markersize=8, zorder=5)

    ax.axhline(bh_cagr, color=COLORS["BuyHold_2"], linewidth=2, linestyle="--",
               label=f"Buy & Hold ({bh_cagr:.2f}%)")
    ax.axvspan(0, 10, alpha=0.06, color="green", label="Realistic cost range (0–10 bps)")

    ax.set_xlabel("Round-Trip Transaction Cost (bps)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title("Break-Even Cost: All Strategies vs. Buy & Hold — Semi-Annual", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(TC_VIZ, "breakeven_all_strategies.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_cross_frequency_cost(cross_freq_df, save=True):
    """Two-panel chart: Median CAGR by frequency + cost drag scaling."""
    df = cross_freq_df.reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(df))
    w = 0.30

    # Left panel: Gross vs Net CAGR
    axes[0].bar(x - w / 2, df["CAGR_Gross_pct"], w, label="Gross", color="#333333")
    axes[0].bar(x + w / 2, df["CAGR_Net_pct"], w, label="Net", color="#888888")
    bh_line = df["CAGR_Net_pct"].iloc[0] - df["Net_vs_BuyHold_pct"].iloc[0]
    axes[0].axhline(bh_line, color=COLORS["BuyHold_2"], linewidth=2,
                    linestyle="--", label="Buy & Hold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["Frequency"], rotation=30, fontsize=9)
    axes[0].set_ylabel("CAGR (%)")
    axes[0].set_title("Median CAGR by Frequency")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.2, axis="y")

    # Right panel: Cost drag in bps
    bars = axes[1].bar(x, df["Cost_Drag_CAGR_bps"], 0.5, color="#CC6600")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["Frequency"], rotation=30, fontsize=9)
    axes[1].set_ylabel("Cost Drag (bps of CAGR)")
    axes[1].set_title("Transaction Cost Drag by Frequency")
    axes[1].grid(alpha=0.2, axis="y")
    for bar, v in zip(bars, df["Cost_Drag_CAGR_bps"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                     f"{v:.1f}", ha="center", fontsize=9)

    fig.suptitle("Median Strategy — Cross-Frequency Cost Comparison",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(TC_VIZ, "cross_frequency_cost.png"),
                    dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# 5. Master runner
# ══════════════════════════════════════════════════════════════════════════

def run_full_transaction_cost_analysis(verbose=True):
    """
    Execute the complete transaction cost analysis.
    Saves all tables to results/transaction_costs/tables/
    and all charts to results/transaction_costs/visualizations/.

    Returns a dict of all key result objects.
    """
    print("=" * 80)
    print("TRANSACTION COST ANALYSIS — Semi-Annual Rebalancing")
    print("=" * 80)

    # ── 1. Base-case backtest ──────────────────────────────────────────────
    print("\n[1/10] Base-case backtest with costs...")
    wealth_df, turnover_df = run_backtest_with_costs()

    # ── 2. Annual summary ─────────────────────────────────────────────────
    print("[2/10] Annual summary with costs...")
    annual_tc = build_annual_summary_with_costs(wealth_df, turnover_df)
    annual_tc.to_csv(os.path.join(TC_TABLES, "annual_summary_with_costs.csv"))

    # ── 3. Turnover summary ───────────────────────────────────────────────
    print("[3/10] Turnover summary...")
    turnover_summary = build_turnover_summary(turnover_df)
    turnover_summary.to_csv(os.path.join(TC_TABLES, "turnover_summary.csv"))
    if verbose:
        print(turnover_summary.to_string())

    # ── 4. ETF spread reference ───────────────────────────────────────────
    print("[4/10] ETF spread table...")
    spread_table = build_etf_spread_table()
    spread_table.to_csv(os.path.join(TC_TABLES, "etf_spreads.csv"), index=False)

    # ── 5. Net-of-cost risk metrics ───────────────────────────────────────
    print("[5/10] Net-of-cost risk metrics...")
    risk_metrics = build_net_risk_metrics_table(wealth_df)
    risk_metrics.to_csv(os.path.join(TC_TABLES, "net_risk_metrics.csv"))
    if verbose:
        print(risk_metrics.to_string())

    # ── 6. Scenario analysis ──────────────────────────────────────────────
    print("[6/10] Scenario analysis (Low / Base / High)...")
    scenario_results = run_scenario_analysis()
    scenario_table = build_scenario_comparison_table(scenario_results)
    scenario_table.to_csv(os.path.join(TC_TABLES, "scenario_comparison.csv"))

    # ── 7. Cost as % of alpha ─────────────────────────────────────────────
    print("[7/10] Cost as % of alpha...")
    cost_alpha = build_cost_alpha_table(wealth_df, scenario_results)
    cost_alpha.to_csv(os.path.join(TC_TABLES, "cost_as_pct_of_alpha.csv"), index=False)
    if verbose:
        median_rows = cost_alpha[cost_alpha["Strategy"] == "Median"]
        print(median_rows[["Scenario", "Gross_Alpha_pct", "Cost_Drag_CAGR_pct",
                           "Cost_as_pct_of_Alpha", "Net_Alpha_pct"]].to_string(index=False))

    # ── 8. Break-even analysis ────────────────────────────────────────────
    print("[8/10] Break-even analysis...")
    breakevens = {}
    for strat in STRATS:
        be, curve = compute_breakeven_cost(
            strategy=strat, search_max_bps=200, step_bps=0.5
        )
        breakevens[strat] = {"breakeven_bps": be, "curve": curve}
        bel = f"{be:.1f} bps" if not np.isnan(be) else ">200 bps"
        if verbose:
            print(f"       {strat}: break-even = {bel}")

    # ── 9. Cross-frequency cost comparison ────────────────────────────────
    print("[9/10] Cross-frequency cost comparison...")
    cross_freq = build_cross_frequency_cost_table()
    cross_freq.to_csv(os.path.join(TC_TABLES, "cross_frequency_cost.csv"))
    if verbose:
        print(cross_freq[["Cumulative_Turnover_pct", "Cost_Drag_CAGR_bps",
                           "Net_vs_BuyHold_pct", "Beats_BuyHold"]].to_string())

    # ── 10. Visualizations ────────────────────────────────────────────────
    print("[10/10] Generating visualizations...")
    plot_combined_net_wealth(wealth_df)
    plot_cumulative_cost_drag(wealth_df)
    plot_turnover_grouped(turnover_df)
    plot_scenario_comparison(scenario_results)
    plot_breakeven_curve(
        breakevens["Median"]["breakeven_bps"],
        breakevens["Median"]["curve"],
        "Median",
    )
    plot_breakeven_all_strategies()
    plot_cross_frequency_cost(cross_freq)

    print(f"\n{'=' * 80}")
    print("Done! All outputs saved to: results/transaction_costs/")
    print(f"  Tables:         results/transaction_costs/tables/  (8 CSV files)")
    print(f"  Visualizations: results/transaction_costs/visualizations/  (7 charts)")
    print("=" * 80)

    return {
        "wealth_df": wealth_df,
        "turnover_df": turnover_df,
        "annual_summary": annual_tc,
        "turnover_summary": turnover_summary,
        "spread_table": spread_table,
        "risk_metrics": risk_metrics,
        "scenario_results": scenario_results,
        "scenario_table": scenario_table,
        "cost_alpha": cost_alpha,
        "breakevens": breakevens,
        "cross_freq": cross_freq,
    }