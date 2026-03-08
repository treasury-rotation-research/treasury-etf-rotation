"""
robustness.py
Start-Date Sensitivity, Rolling-End-Date Persistence, and Coefficient-of-Variation
Analysis for the U.S. Treasury ETF Rotation Strategy.

Five-part robustness suite:
    1. Start-Date Sensitivity  — 12 monthly calendar anchors per frequency
    2. Rolling-End-Date Persistence — does Median rank #1 across rolling endpoints?
    3. Coefficient-of-Variation Summary — how stable is each strategy's terminal value?
    4. Visualizations:
        a. Final-value dot-strip plot (redesigned)
        b. Risk metrics boxplots (redesigned)
        c. Median win-rate heatmap (NEW)
        d. Rolling-end-date persistence chart (NEW)
        e. CV summary bar chart (NEW)

Output: results/robustness/{tables/, visualizations/}
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from typing import List, Dict, Optional, Tuple

from src.config import (
    ETFS, START_YEAR, END_YEAR, INITIAL_CASH, RF_ANNUAL,
    LUATTRUU_FILENAME, COLORS, STRAT_COLORS,
)
from src.utils import (
    load_etf_data, load_luattruu, mdd_from_values,
    annualized_stats_from_pct, compute_mdd_from_pct,
)

warnings.filterwarnings("ignore")

# ── Output directories ────────────────────────────────────────────────────
_RESULTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
ROB_DIR    = os.path.join(_RESULTS_ROOT, "robustness")
ROB_TABLES = os.path.join(ROB_DIR, "tables")
ROB_VIZ    = os.path.join(ROB_DIR, "visualizations")

for _d in (ROB_DIR, ROB_TABLES, ROB_VIZ):
    os.makedirs(_d, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
ROBUSTNESS_FREQUENCIES = [
    ("Annual",      12),
    ("Semi-Annual",  6),
    ("Quarterly",    3),
]

STRATS = ["Winners", "Median", "Losers"]

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# ── Unified colors from config.py ─────────────────────────────────────────
_STRAT_COLORS = {
    "Winners": STRAT_COLORS.get("Winners", COLORS.get("Winners", "#008000")),
    "Median":  STRAT_COLORS.get("Median",  COLORS.get("Median",  "#000000")),
    "Losers":  STRAT_COLORS.get("Losers",  COLORS.get("Losers",  "#0000FF")),
}
_BH_COLOR = COLORS.get("BuyHold_2", "#FF0000")

FREQ_COLORS = {
    "Annual":      "#1a6faf",
    "Semi-Annual": "#e07b00",
    "Quarterly":   "#2e8b57",
}

# ── Plot style ────────────────────────────────────────────────────────────
BG_COLOR = "white"
plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.18,
    "grid.linestyle":     "--",
    "figure.dpi":         100,
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.labelsize":     10,
})




# ══════════════════════════════════════════════════════════════════════════
# 1. Rolling-start backtest engine  (GROSS — no transaction costs)
# ══════════════════════════════════════════════════════════════════════════

def _build_offset_period_ends(
    start_year: int,
    end_year: int,
    holding_months: int,
    offset_month: int,
) -> List[pd.Timestamp]:
    """Build rebalancing dates for a given holding period and start-month offset."""
    anchor = (
        pd.Timestamp(year=start_year - 1, month=offset_month, day=1)
        + pd.offsets.MonthEnd(0)
    )
    dates = [anchor]
    while True:
        nxt = (dates[-1] + pd.DateOffset(months=holding_months)
               + pd.offsets.MonthEnd(0))
        if nxt.year > end_year + 1:
            break
        dates.append(nxt)

    last_needed = pd.Timestamp(f"{end_year}-03-31")
    if dates[-1] < last_needed:
        dates.append(
            dates[-1] + pd.DateOffset(months=holding_months)
            + pd.offsets.MonthEnd(0)
        )
    return sorted(set(dates))


def _build_annual_summary_for_offset(
    etfs: List[str],
    holding_months: int,
    offset_month: int,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    initial_cash: float = INITIAL_CASH,
    rf_annual: float = RF_ANNUAL,
) -> Optional[pd.DataFrame]:
    """
    Run the full backtest for one start-month offset and return a
    year-indexed annual summary DataFrame.
    """
    period_ends = _build_offset_period_ends(
        start_year, end_year, holding_months, offset_month
    )
    if len(period_ends) < 3:
        return None

    report_periods = period_ends[1:]
    etf_data = load_etf_data(etfs)

    # ── Period returns ─────────────────────────────────────────────────────
    returns = pd.DataFrame(index=report_periods, columns=etfs, dtype=float)
    for etf, df in etf_data.items():
        price = df.set_index("Date")["Adj Close"].astype(float).sort_index()
        for i in range(1, len(period_ends)):
            s, e = period_ends[i - 1], period_ends[i]
            try:
                p0 = price.loc[:s].iloc[-1]
                p1 = price.loc[:e].iloc[-1]
                returns.loc[e, etf] = p1 / p0 - 1.0
            except Exception:
                returns.loc[e, etf] = np.nan

    # ── Rotation (GROSS) ──────────────────────────────────────────────────
    k = max(1, len(etfs) // 3)
    cash   = {s: initial_cash for s in STRATS}
    groups = {s: [] for s in STRATS}
    wealth_hist = {"Period": [], **{s: [] for s in STRATS}}

    for idx_i, end in enumerate(report_periods):
        if idx_i == 0:
            w_sel = m_sel = l_sel = etfs[:]
        else:
            prev_row = returns.loc[report_periods[idx_i - 1]].dropna().sort_values()
            if prev_row.empty:
                w_sel = m_sel = l_sel = etfs[:]
            else:
                l_sel = list(prev_row.index[:k])
                m_sel = list(prev_row.index[k:2*k]) if len(prev_row) >= 2*k else list(prev_row.index)
                w_sel = list(prev_row.index[2*k:])  if len(prev_row) >= 2*k else list(prev_row.index)

        sel_map = {"Winners": w_sel, "Median": m_sel, "Losers": l_sel}
        for s in STRATS:
            groups[s].append(sel_map[s])

        r_now = returns.loc[end]
        for strat in STRATS:
            vals = r_now[sel_map[strat]].dropna()
            cash[strat] *= 1.0 + (float(vals.mean()) if not vals.empty else 0.0)

        wealth_hist["Period"].append(end)
        for s in STRATS:
            wealth_hist[s].append(cash[s])

    # ── Benchmark (LUATTRUU) ──────────────────────────────────────────────
    _, lu_level = load_luattruu(LUATTRUU_FILENAME, start_year)
    lu_level = lu_level.sort_index()
    base_date = period_ends[0]
    base_val = (lu_level.loc[:base_date].iloc[-1]
                if not lu_level.loc[:base_date].empty else lu_level.iloc[0])
    lu_norm = lu_level / float(base_val)

    bh_values = [
        initial_cash * float(lu_norm.loc[:d].iloc[-1])
        if not lu_norm.loc[:d].empty else np.nan
        for d in wealth_hist["Period"]
    ]

    # ── Wealth DataFrame ──────────────────────────────────────────────────
    df_all = pd.DataFrame(
        {**{s: wealth_hist[s] for s in STRATS}, "BuyHold_2": bh_values},
        index=pd.DatetimeIndex(wealth_hist["Period"]),
    )

    # ── Daily return panel for yearly risk metrics ─────────────────────────
    lu_daily_pct, _ = load_luattruu(LUATTRUU_FILENAME, start_year)
    ret_panel = pd.DataFrame()
    for etf, df in etf_data.items():
        tmp = df[["Date", "Adj Close"]].copy()
        tmp["Date"] = pd.to_datetime(tmp["Date"])
        tmp.set_index("Date", inplace=True)
        ret_panel = ret_panel.join(
            (tmp["Adj Close"].pct_change() * 100.0).rename(etf), how="outer"
        )
    ret_panel["LUATTRUU"] = lu_daily_pct

    # ── Yearly risk metrics ────────────────────────────────────────────────
    periods_list = [
        (period_ends[i], period_ends[i + 1], i)
        for i in range(len(period_ends) - 1)
    ]
    strats_plus_bm = STRATS + ["BuyHold_2"]
    year_stats = {g: {} for g in strats_plus_bm}

    for y in range(start_year, end_year + 1):
        for g in STRATS:
            segs = []
            for seg_start, seg_end, i in periods_list:
                if seg_end.year != y:
                    continue
                cols = groups[g][i] if i < len(groups[g]) else []
                mask = (ret_panel.index > seg_start) & (ret_panel.index <= seg_end)
                if not cols:
                    continue
                ew_pct = ret_panel[cols].loc[mask].mean(axis=1)
                if not ew_pct.dropna().empty:
                    segs.append(ew_pct.dropna())
            if segs:
                s = pd.concat(segs).sort_index()
                vol, sh = annualized_stats_from_pct(s, rf_ann=rf_annual)
                mdd = compute_mdd_from_pct(s)
            else:
                vol = sh = mdd = np.nan
            year_stats[g][y] = (vol, sh, mdd)

        lu_year = ret_panel["LUATTRUU"].loc[
            (ret_panel.index >= pd.Timestamp(f"{y}-01-01")) &
            (ret_panel.index <= pd.Timestamp(f"{y}-12-31"))
        ].dropna()
        if not lu_year.empty:
            vol, sh = annualized_stats_from_pct(lu_year, rf_ann=rf_annual)
            mdd = compute_mdd_from_pct(lu_year)
        else:
            vol = sh = mdd = np.nan
        year_stats["BuyHold_2"][y] = (vol, sh, mdd)

    # ── Annual summary DataFrame ───────────────────────────────────────────
    annual_rows = []
    for y in range(start_year, end_year + 1):
        yr = df_all[df_all.index.year == y]
        if not yr.empty:
            annual_rows.append(yr.iloc[-1])
    annual_df = pd.DataFrame(annual_rows)
    annual_df.index = list(range(start_year, start_year + len(annual_df)))
    annual_df.index.name = "Year"

    for g in strats_plus_bm:
        annual_df[f"Volatility_{g}"] = [
            year_stats[g].get(y, (np.nan,) * 3)[0] for y in annual_df.index
        ]
        annual_df[f"Sharpe_{g}"] = [
            year_stats[g].get(y, (np.nan,) * 3)[1] for y in annual_df.index
        ]
        annual_df[f"MDD_{g}"] = [
            year_stats[g].get(y, (np.nan,) * 3)[2] for y in annual_df.index
        ]

    for g in strats_plus_bm:
        yoy = []
        for y in annual_df.index:
            if (y - 1) in annual_df.index:
                prev = float(annual_df.loc[y - 1, g])
                curr = float(annual_df.loc[y, g])
                yoy.append((curr / prev - 1.0) * 100.0)
            else:
                yoy.append(np.nan)
        annual_df[f"Return_{g}"] = yoy

    return annual_df


# ══════════════════════════════════════════════════════════════════════════
# 2. Build summary tables  (one per frequency)
# ══════════════════════════════════════════════════════════════════════════

def build_rolling_summary_table(
    holding_months: int,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For a given holding period, run the backtest for all 12 start-month
    offsets and extract the END_YEAR row from each annual summary.
    """
    rows = []
    for offset in range(1, 13):
        month_label = MONTH_LABELS[offset - 1]
        annual_df = _build_annual_summary_for_offset(
            etfs=ETFS,
            holding_months=holding_months,
            offset_month=offset,
        )
        if annual_df is None or END_YEAR not in annual_df.index:
            if verbose:
                print(f"    [{month_label}] No data for {END_YEAR} — skipping")
            continue

        row = annual_df.loc[END_YEAR].to_dict()
        row["Offset_Month"] = month_label
        rows.append(row)
        if verbose:
            print(f"    [{month_label}] OK")

    df = pd.DataFrame(rows)
    cols = ["Offset_Month"] + [c for c in df.columns if c != "Offset_Month"]
    return df[cols].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════
# 3. Newey-West t-statistic  (no statsmodels dependency)
# ══════════════════════════════════════════════════════════════════════════

def _newey_west_tstat(
    series: np.ndarray,
    max_lag: int = None,
) -> Tuple[float, float, Tuple[float, float]]:
    """Newey-West HAC adjusted t-statistic for the mean of a series."""
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, (np.nan, np.nan)

    if max_lag is None:
        max_lag = max(1, int(n ** (1.0 / 3.0)))

    mean  = x.mean()
    resid = x - mean
    nw_var = float(np.dot(resid, resid) / n)
    for j in range(1, max_lag + 1):
        w = 1.0 - j / (max_lag + 1.0)
        nw_var += 2.0 * w * float(np.dot(resid[j:], resid[:-j]) / n)

    nw_se  = np.sqrt(max(nw_var, 0.0) / n)
    t_stat = mean / nw_se if nw_se > 0 else np.nan
    return float(mean), float(t_stat), (
        float(mean - 1.96 * nw_se), float(mean + 1.96 * nw_se)
    )


# ══════════════════════════════════════════════════════════════════════════
# 4. NEW — Rolling-End-Date Persistence Analysis
# ══════════════════════════════════════════════════════════════════════════

def build_rolling_endpoint_analysis(
    holding_months: int = 6,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Test how often the Median strategy ranks #1 across rolling end-year
    endpoints (2015 through END_YEAR). For each endpoint, run the full
    backtest and record the final value, CAGR, and Sharpe for each strategy.

    This addresses the concern that the Median result is an artifact of
    the specific end date chosen.
    """
    rows = []
    for end_yr in range(2015, END_YEAR + 1):
        annual_df = _build_annual_summary_for_offset(
            etfs=ETFS,
            holding_months=holding_months,
            offset_month=12,  # standard Dec alignment
            start_year=START_YEAR,
            end_year=end_yr,
        )
        if annual_df is None or end_yr not in annual_df.index:
            continue

        row_data = {"End_Year": end_yr}
        for s in STRATS + ["BuyHold_2"]:
            row_data[f"FinalVal_{s}"] = annual_df.loc[end_yr, s] if s in annual_df.columns else np.nan
            sharpe_col = f"Sharpe_{s}"
            row_data[f"Sharpe_{s}"] = (
                annual_df[sharpe_col].mean() if sharpe_col in annual_df.columns else np.nan
            )

        # Rank strategies by final value (higher = better rank)
        strat_vals = {s: row_data[f"FinalVal_{s}"] for s in STRATS}
        ranked = sorted(strat_vals, key=lambda x: strat_vals.get(x, -999), reverse=True)
        for rank, s in enumerate(ranked, 1):
            row_data[f"Rank_{s}"] = rank

        # Does Median beat BuyHold?
        row_data["Median_Beats_BH"] = (
            row_data.get("FinalVal_Median", 0) > row_data.get("FinalVal_BuyHold_2", 999)
        )

        rows.append(row_data)
        if verbose:
            ranks_str = "  ".join(f"{s}=#{row_data[f'Rank_{s}']}" for s in STRATS)
            print(f"    End {end_yr}: {ranks_str}  Median>BH={'✓' if row_data['Median_Beats_BH'] else '✗'}")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 5. NEW — Coefficient-of-Variation Summary
# ══════════════════════════════════════════════════════════════════════════

def build_cv_summary(
    tables: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Coefficient of Variation of terminal portfolio value across 12 start months.
    Lower CV = more stable result regardless of calendar alignment.
    """
    rows = []
    for freq_label, df in tables.items():
        if df is None or df.empty:
            continue
        for s in STRATS + ["BuyHold_2"]:
            if s not in df.columns:
                continue
            vals = df[s].dropna()
            if len(vals) < 2:
                continue
            rows.append({
                "Frequency": freq_label,
                "Strategy":  s,
                "Mean":      round(vals.mean(), 2),
                "Std":       round(vals.std(), 2),
                "CV_pct":    round(vals.std() / vals.mean() * 100, 2) if vals.mean() != 0 else np.nan,
                "Min":       round(vals.min(), 2),
                "Max":       round(vals.max(), 2),
                "Range":     round(vals.max() - vals.min(), 2),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 6. Visualizations  (redesigned + new charts)
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
# Design constants for charts
# ══════════════════════════════════════════════════════════════════════════

_BG_COLOR    = "#ffffff"
_GRID_ALPHA  = 0.12
_GRID_COLOR  = "#888888"
_EDGE_COLOR  = "#dddddd"
_TITLE_SZ    = 13
_LABEL_SZ    = 10
_TICK_SZ     = 9
_FREQS_LIST  = ["Annual", "Semi-Annual", "Quarterly"]
_FREQ_TICK   = {"Annual": "Annual\n(12m)", "Semi-Annual": "Semi-Ann\n(6m)", "Quarterly": "Quarterly\n(3m)"}


def _apply_style(ax, grid_axis="both"):
    ax.set_facecolor(_BG_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#555555", labelsize=_TICK_SZ)
    ax.grid(axis=grid_axis, alpha=_GRID_ALPHA, color=_GRID_COLOR, linestyle="-")


def _rob_save(fig, fname):
    fig.tight_layout()
    fig.savefig(os.path.join(ROB_VIZ, fname),
                dpi=300, bbox_inches="tight", facecolor=_BG_COLOR)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Visualizations
# ══════════════════════════════════════════════════════════════════════════

def plot_final_value_dotplot(tables, save=True):
    """
    Spaghetti/fan chart: Median cumulative growth paths across all 12
    start-month offsets vs Buy & Hold. Shows that Median consistently
    beats B&H regardless of calendar alignment.
    One panel per frequency.
    """
    freq_display = {"Annual": "Annual (12m)", "Semi-Annual": "Semi-Annual (6m)",
                    "Quarterly": "Quarterly (3m)"}

    # We need full annual paths, not just the final row.
    # Rebuild from the offset engine for each frequency.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, facecolor=_BG_COLOR)

    for ax, freq_label in zip(axes, _FREQS_LIST):
        _apply_style(ax, "both")
        df = tables.get(freq_label)
        if df is None or df.empty:
            continue

        # Simple bar comparison: Median final value per month vs B&H
        df = df.copy()
        df["Offset_Month"] = pd.Categorical(df["Offset_Month"], categories=MONTH_LABELS, ordered=True)
        df = df.sort_values("Offset_Month")
        x = np.arange(len(df))

        bh_vals = df["BuyHold_2"].values
        med_vals = df["Median"].values

        # Median bars
        ax.bar(x, med_vals, width=0.7, color=_STRAT_COLORS["Median"], alpha=0.7,
               edgecolor="white", lw=0.8, label="Median", zorder=3)

        # B&H reference line
        bh_mean = np.nanmean(bh_vals)
        ax.axhline(bh_mean, color=_BH_COLOR, linewidth=2, linestyle="--",
                    alpha=0.7, zorder=5, label=f"B&H avg (${bh_mean:.0f})")

        # Highlight months where Median > B&H
        for i in range(len(df)):
            if med_vals[i] > bh_vals[i]:
                ax.bar(x[i], med_vals[i], width=0.7, color=_STRAT_COLORS["Median"],
                       alpha=0.9, edgecolor="white", lw=0.8, zorder=3)

        # Win rate annotation
        win_rate = (med_vals > bh_vals).sum() / len(med_vals) * 100
        ax.text(0.97, 0.95, f"Median > B&H:\n{win_rate:.0f}% of months",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                fontweight="bold", color=_STRAT_COLORS["Median"],
                bbox=dict(facecolor="white", edgecolor=_EDGE_COLOR,
                          alpha=0.95, pad=4, boxstyle="round,pad=0.3"))

        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_LABELS, fontsize=7.5, rotation=45)
        ax.set_title(freq_display[freq_label], fontsize=11, fontweight="bold",
                     color=FREQ_COLORS[freq_label])
        if ax == axes[0]:
            ax.set_ylabel("Final Portfolio Value ($)", fontsize=_LABEL_SZ)
            ax.legend(fontsize=8, framealpha=0.95, edgecolor=_EDGE_COLOR)
        ax.set_ylim(90, 220)

    fig.suptitle("Median Strategy: Final Value by Start Month vs Buy & Hold",
                 fontsize=_TITLE_SZ, fontweight="bold", y=1.01)
    fig.text(0.5, 0.96, f"{START_YEAR}–{END_YEAR}  |  Gross of costs  |  darker bars = beats B&H",
             fontsize=9, color="#888888", ha="center")

    if save:
        _rob_save(fig, "final_value_dotplot.png")
    else:
        plt.close(fig)


def plot_risk_boxplots(tables, save=True):
    """Clean 1x3 boxplots - Median strategy only. No strip dots."""
    metric_cfg = [
        ("Sharpe_Median",     "Sharpe_BuyHold_2",     "Sharpe Ratio",        True),
        ("Volatility_Median", "Volatility_BuyHold_2", "Volatility (%)",      False),
        ("MDD_Median",        "MDD_BuyHold_2",        "Max Drawdown (%)",    False),
    ]

    med_color = "#222222"

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), facecolor="white")

    for ax, (col, bh_col, ylabel, has_zero) in zip(axes, metric_cfg):
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors="#555555", labelsize=9)
        ax.grid(axis="y", alpha=0.15, linestyle="--")

        data_list, labels, bh_vals = [], [], []
        for freq_label in _FREQS_LIST:
            df = tables.get(freq_label)
            if df is None or col not in df.columns:
                continue
            data_list.append(df[col].dropna().values)
            labels.append(_FREQ_TICK[freq_label])
            if bh_col in df.columns:
                bh_vals.append(df[bh_col].dropna().mean())

        if not data_list:
            continue

        bp = ax.boxplot(
            data_list, tick_labels=labels, patch_artist=True, widths=0.42,
            showmeans=True, showfliers=False,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor=med_color, markersize=6,
                           markeredgewidth=1.5),
            medianprops=dict(color="white", linewidth=2.5),
            whiskerprops=dict(linewidth=1.3, color=med_color, alpha=0.5),
            capprops=dict(linewidth=1.3, color=med_color, alpha=0.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(med_color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(med_color)
            patch.set_linewidth(1.5)

        # B&H reference line
        if bh_vals:
            bh_mean = np.mean(bh_vals)
            ax.axhline(bh_mean, color=_BH_COLOR, linewidth=2, linestyle="--",
                        alpha=0.65, zorder=1,
                        label="Buy & Hold" if ax == axes[0] else "")

        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")

    axes[0].legend(fontsize=9, framealpha=0.95, edgecolor="#dddddd", loc="best")

    fig.suptitle("Median Strategy: Risk Metrics Across 12 Start-Month Anchors",
                 fontsize=13, fontweight="bold", y=0.98)
    
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    if save:
        _rob_save(fig, "risk_boxplots.png")
    else:
        plt.close(fig)


def plot_median_win_rate_heatmap(tables, save=True):
    """Landscape heatmap — Median rank by month × frequency."""
    from matplotlib.colors import ListedColormap

    rank_data = pd.DataFrame(index=MONTH_LABELS, columns=_FREQS_LIST, dtype=float)
    beats_bh  = pd.DataFrame(index=MONTH_LABELS, columns=_FREQS_LIST, dtype=bool)

    for freq_label in _FREQS_LIST:
        df = tables.get(freq_label)
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            month = row["Offset_Month"]
            strat_vals = {s: row.get(s, np.nan) for s in STRATS}
            ranked = sorted(strat_vals, key=lambda x: strat_vals.get(x, -999), reverse=True)
            rank_data.loc[month, freq_label] = ranked.index("Median") + 1
            beats_bh.loc[month, freq_label] = row.get("Median", 0) > row.get("BuyHold_2", 999)

    rank_data = rank_data.astype(float)

    # Landscape: frequencies as rows, months as columns
    rank_t  = rank_data.T
    beats_t = beats_bh.T

    fig, ax = plt.subplots(figsize=(14, 3.2), facecolor=_BG_COLOR)
    cmap = ListedColormap(["#1a1a1a", "#e8b930", "#cc4444"])

    sns.heatmap(rank_t, ax=ax, annot=False, cmap=cmap, vmin=0.5, vmax=3.5,
                linewidths=2.5, linecolor="white",
                cbar_kws={"shrink": 0.5, "label": "", "ticks": [1, 2, 3]},
                square=True)

    for i, freq in enumerate(_FREQS_LIST):
        for j, month in enumerate(MONTH_LABELS):
            rank = rank_t.iloc[i, j]
            star = "★" if beats_t.iloc[i, j] else ""
            text = f"{int(rank)}{star}" if not np.isnan(rank) else ""
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white")

    ax.set_title("Median Strategy Rank by Start Month & Frequency",
                 fontsize=_TITLE_SZ, fontweight="bold", pad=10)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=_TICK_SZ)

    fig.text(0.5, -0.04,
             "1 = best among W/M/L  |  ★ = also beats Buy & Hold  |  "
             "green = rank 1, yellow = rank 2, red = rank 3",
             fontsize=9, color="#888888", ha="center")

    if save:
        _rob_save(fig, "median_rank_heatmap.png")
    else:
        plt.close(fig)


def plot_rolling_endpoint_persistence(endpoint_df, save=True):
    """Two-panel: final values over time + Median rank."""
    if endpoint_df.empty:
        return

    years = endpoint_df["End_Year"].values

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), facecolor=_BG_COLOR,
                             gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.25})

    ax = axes[0]
    _apply_style(ax, "both")
    for s in STRATS:
        col = f"FinalVal_{s}"
        if col in endpoint_df.columns:
            ax.plot(years, endpoint_df[col].values, color=_STRAT_COLORS[s], linewidth=2.2, label=s,
                    marker="o", markersize=5, markeredgecolor="white", markeredgewidth=1)
    if "FinalVal_BuyHold_2" in endpoint_df.columns:
        ax.plot(years, endpoint_df["FinalVal_BuyHold_2"].values,
                color=_BH_COLOR, linewidth=1.8, ls="--", label="Buy & Hold", marker="s", markersize=3.5)
    ax.set_ylabel("Final Portfolio Value ($)", fontsize=_LABEL_SZ)
    ax.set_title("Rolling-End-Date Persistence: Semi-Annual Strategy",
                 fontsize=_TITLE_SZ, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.95, edgecolor=_EDGE_COLOR, loc="upper left")

    ax = axes[1]
    _apply_style(ax, "y")
    if "Rank_Median" in endpoint_df.columns:
        ranks = endpoint_df["Rank_Median"].values
        colors = [_STRAT_COLORS["Median"] if r == 1 else "#e8b930" if r == 2 else "#cc4444" for r in ranks]
        ax.bar(years, ranks, color=colors, edgecolor="white", linewidth=0.8, width=0.7)
        ax.set_ylim(0.4, 3.6); ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["1st", "2nd", "3rd"], fontsize=_TICK_SZ)
        ax.invert_yaxis()
        ax.set_ylabel("Median Rank", fontsize=_LABEL_SZ)
        ax.set_xlabel("Evaluation End Year", fontsize=_LABEL_SZ)

        win_rate = (ranks == 1).sum() / len(ranks) * 100
        ax.text(0.98, 0.2, f"Rank #1 in {win_rate:.0f}% of endpoints",
                transform=ax.transAxes, ha="right", fontsize=10, fontweight="bold", color=_STRAT_COLORS["Median"],
                bbox=dict(facecolor="white", edgecolor=_STRAT_COLORS["Median"], alpha=0.9, pad=4, boxstyle="round,pad=0.4"))

    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)

    if save:
        _rob_save(fig, "rolling_endpoint_persistence.png")
    else:
        plt.close(fig)


def plot_cv_summary(cv_df, save=True):
    """CV bar chart — lower = more stable."""
    if cv_df.empty:
        return

    all_strats = STRATS + ["BuyHold_2"]
    labels = {"Winners": "Winners", "Median": "Median", "Losers": "Losers", "BuyHold_2": "Buy & Hold"}

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=_BG_COLOR)
    _apply_style(ax, "y")

    x = np.arange(len(_FREQS_LIST))
    n = len(all_strats)
    w = 0.17

    for i, s in enumerate(all_strats):
        cvs = []
        for freq in _FREQS_LIST:
            row = cv_df[(cv_df["Frequency"] == freq) & (cv_df["Strategy"] == s)]
            cvs.append(row["CV_pct"].iloc[0] if not row.empty else np.nan)
        color = _STRAT_COLORS.get(s, _BH_COLOR)
        offset = (i - (n - 1) / 2.0) * w
        bars = ax.bar(x + offset, cvs, w, label=labels[s], color=color, edgecolor="white", lw=0.8, alpha=0.85)
        for bar, v in zip(bars, cvs):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15,
                        f"{v:.1f}", ha="center", fontsize=7.5, fontweight="bold", color="#555")

    ax.set_xticks(x); ax.set_xticklabels(_FREQS_LIST)
    ax.set_ylabel("Coefficient of Variation (%)", fontsize=_LABEL_SZ)
    ax.set_title("Terminal Value Stability Across 12 Start Months",
                 fontsize=_TITLE_SZ, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.95, edgecolor=_EDGE_COLOR)

    fig.text(0.5, -0.02, "Lower CV = more stable regardless of calendar alignment",
             fontsize=9, color="#888888", ha="center")

    if save:
        _rob_save(fig, "cv_summary.png")
    else:
        plt.close(fig)

def run_full_robustness_analysis(verbose: bool = True) -> Dict:
    """
    Execute the full 5-part robustness analysis.

    Saves
    -----
    Tables (results/robustness/tables/):
        rolling_results_12mo.csv, rolling_results_6mo.csv, rolling_results_3mo.csv
        rolling_endpoint_persistence.csv  (NEW)
        cv_summary.csv                    (NEW)

    Visualizations (results/robustness/visualizations/):
        final_value_dotplot.png        — Dot strip plot
        risk_boxplots.png              — Volatility, MDD & Sharpe boxplots
        median_rank_heatmap.png        — NEW: Median rank by month × frequency
        rolling_endpoint_persistence.png — NEW: End-date persistence
        cv_summary.png                 — NEW: CV bar chart
    """
    print("=" * 70)
    print("ROBUSTNESS ANALYSIS  (5-part suite)")
    print(f"Strategies : {', '.join(STRATS)}")
    print(f"Frequencies: {', '.join(f for f, _ in ROBUSTNESS_FREQUENCIES)}")
    print(f"Anchors    : 12 monthly offsets (Jan-Dec)")
    print(f"Period     : {START_YEAR}-{END_YEAR}  |  Rf = {RF_ANNUAL}")
    print("NOTE: Start-date analysis uses GROSS returns (no transaction costs)")
    print("=" * 70)

    freq_file_map = {
        "Annual":      ("rolling_results_12mo.csv", 12),
        "Semi-Annual": ("rolling_results_6mo.csv",   6),
        "Quarterly":   ("rolling_results_3mo.csv",   3),
    }

    tables = {}

    # ── Part 1: Build start-date summary tables ──────────────────────────
    step = 1
    for freq_label, (fname, holding_months) in freq_file_map.items():
        print(f"\n[{step}/7] {freq_label} ({holding_months}m) -- building summary table...")
        tbl = build_rolling_summary_table(
            holding_months=holding_months,
            verbose=verbose,
        )
        path = os.path.join(ROB_TABLES, fname)
        tbl.to_csv(path, index=False)
        tables[freq_label] = tbl
        print(f"       -> saved: {fname}  ({len(tbl)} rows x {len(tbl.columns)} cols)")
        step += 1

    # ── Part 2: Rolling-end-date persistence ─────────────────────────────
    print(f"\n[{step}/7] Rolling-end-date persistence (Semi-Annual)...")
    endpoint_df = build_rolling_endpoint_analysis(holding_months=6, verbose=verbose)
    endpoint_df.to_csv(os.path.join(ROB_TABLES, "rolling_endpoint_persistence.csv"), index=False)
    step += 1

    # ── Part 3: CV summary ───────────────────────────────────────────────
    print(f"\n[{step}/7] Coefficient-of-Variation summary...")
    cv_df = build_cv_summary(tables)
    cv_df.to_csv(os.path.join(ROB_TABLES, "cv_summary.csv"), index=False)
    if verbose:
        print(cv_df.to_string(index=False))
    step += 1

    # ── Visualizations ────────────────────────────────────────────────────
    print(f"\n[{step}/7] Generating visualizations...")

    plot_risk_boxplots(tables)
    print("       -> risk_boxplots.png  (rolling-endpoint persistence)")

    plot_median_win_rate_heatmap(tables)
    print("       -> median_rank_heatmap.png")

    plot_rolling_endpoint_persistence(endpoint_df)
    print("       -> rolling_endpoint_persistence.png")

    plot_cv_summary(cv_df)
    print("       -> cv_summary.png")

    print(f"\n{'=' * 70}")
    print("Done! All outputs saved to: results/robustness/")
    print(f"  Tables         : results/robustness/tables/         (5 CSVs)")
    print(f"  Visualizations : results/robustness/visualizations/ (4 PNGs)")
    print("=" * 70)

    return {
        "tables": tables,
        "endpoint_persistence": endpoint_df,
        "cv_summary": cv_df,
    }


# ── CLI entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    run_full_robustness_analysis(verbose=True)
