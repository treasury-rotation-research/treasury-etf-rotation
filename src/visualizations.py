"""
visualizations.py
Visualization functions for the Treasury ETF rotation backtest.

All figures are saved as PNG (or PDF for the multi-panel dashboard) to
results/visualizations/.

Public API
----------
  build_ref_buyhold(runs, prefer)    — build benchmark reference dict
  plot_final_values(runs, ref)       — bar: final portfolio value by frequency
  plot_growth_curve(runs, ref, name) — line: growth curve for one frequency
  plot_all_growth_curves(runs, ref)  — 2×3 dashboard of growth curves
  plot_metric_bars(runs, ref)        — 6 bar charts: CAGR, Sharpe, Vol,
                                       MDD, Sortino, IR
  plot_sortino_ir_bars(runs, ref)    — bar: Sortino & IR only
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

from src.config import STRATEGIES, COLORS, VIZ_DIR, REBALANCING_SETTINGS
from src.utils import cagr_from_values, mdd_from_values

# ── Plot style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.dpi":        100,
})

_ROTATION_STRATS = ["Winners", "Median", "Losers"]


def _ensure_viz_dir() -> None:
    os.makedirs(VIZ_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Internal: generic grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════

def _save_barplot(
    metric_name: str,
    data_by_group: dict,
    xlabels: list,
    zero_line: bool = False,
    save: bool = True,
) -> None:
    """
    Render a grouped bar chart for `metric_name` across rebalancing frequencies.

    Parameters
    ----------
    metric_name   : y-axis label and plot title suffix
    data_by_group : {strategy: [value_per_frequency, ...]}
    xlabels       : frequency labels for the x-axis
    zero_line     : draw a horizontal line at y=0
    save          : write PNG to VIZ_DIR
    """
    _ensure_viz_dir()
    groups = list(data_by_group.keys())
    n_grp  = len(groups)
    x      = np.arange(len(xlabels))
    width  = 0.18

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, g in enumerate(groups):
        offset = (i - (n_grp - 1) / 2.0) * width
        ax.bar(x + offset, data_by_group[g], width, label=g,
               color=COLORS[g], edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f"{metric_name} by Rebalancing Frequency", fontsize=13)
    ax.legend(title="Strategy", fontsize=9)
    if zero_line:
        ax.axhline(0, linewidth=0.9, color="black", alpha=0.4)
    fig.tight_layout()

    if save:
        safe = (metric_name.replace(" ", "_")
                           .replace("%", "pct")
                           .replace("(", "").replace(")", "")
                           .replace("/", "_"))
        fig.savefig(os.path.join(VIZ_DIR, f"{safe}_Rebalancing.png"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Internal: collect per-metric data across all frequencies
# ═══════════════════════════════════════════════════════════════════════════

def _collect_metric(
    runs: dict,
    ref: dict,
    col_template: str,
    ref_key: str,
    agg: str = "mean",   # 'mean' | 'last' | 'cagr' | 'mdd_values'
) -> Tuple[list, dict]:
    """
    Iterate over REBALANCING_SETTINGS (excluding Daily) and build per-frequency
    lists for each strategy.

    Parameters
    ----------
    col_template : column name pattern with {g} for strategy, e.g. 'Sharpe_{g}'
    ref_key      : key into ref dict for the BuyHold_2 value
    agg          : aggregation applied to the annual column
                   'mean'       → df[col].mean()
                   'last'       → df[col].iloc[-1]
                   'cagr'       → cagr_from_values(df[g]) * 100
                   'mdd_values' → mdd_from_values(df[g])

    Returns
    -------
    freq_labels, data_by_group
    """
    freq_labels   = []
    data_by_group = {g: [] for g in STRATEGIES}

    for name, unit, freq in REBALANCING_SETTINGS:
        key = (unit, freq)
        if key not in runs or name == "Daily":
            continue
        freq_labels.append(name)
        _, df = runs[key]

        for g in STRATEGIES:
            if g == "BuyHold_2":
                data_by_group[g].append(ref.get(ref_key, np.nan))
                continue

            col = col_template.format(g=g)
            if agg == "mean":
                val = float(df[col].mean()) if col in df.columns else np.nan
            elif agg == "last":
                val = float(df[col].iloc[-1]) if col in df.columns else np.nan
            elif agg == "cagr":
                c = cagr_from_values(df[g]) if g in df.columns else np.nan
                val = c * 100.0 if not np.isnan(c) else np.nan
            elif agg == "mdd_values":
                val = mdd_from_values(df[g]) if g in df.columns else np.nan
            else:
                val = np.nan
            data_by_group[g].append(val)

    return freq_labels, data_by_group


# ═══════════════════════════════════════════════════════════════════════════
# 1. Final portfolio value bar chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_final_values(runs: dict, ref: dict, save: bool = True) -> None:
    """
    Grouped bar chart of final portfolio value across rebalancing frequencies.

    Parameters
    ----------
    runs : {(unit, freq): (label, annual_df), ...}
    ref  : benchmark reference dict from build_ref_buyhold()
    """
    freq_labels, data = _collect_metric(runs, ref, "{g}", "final", agg="last")
    _save_barplot("Final Portfolio Value ($)", data, freq_labels, save=save)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Growth curve — single frequency
# ═══════════════════════════════════════════════════════════════════════════

def plot_growth_curve(
    runs: dict,
    ref: dict,
    freq_name: str,
    save: bool = True,
) -> None:
    """Line chart of year-end portfolio value for a single rebalancing frequency."""
    _ensure_viz_dir()
    match = [k for k, (n, _) in runs.items() if n == freq_name]
    if not match:
        print(f"[skip] No run found for '{freq_name}'")
        return

    _, df  = runs[match[0]]
    years  = df.index.astype(int)

    fig, ax = plt.subplots(figsize=(12, 6))
    for s in _ROTATION_STRATS:
        if s in df.columns:
            ax.plot(years, df[s].values, label=s, color=COLORS[s], linewidth=2)
    ax.plot(ref["years_index"], ref["curve"].values,
            label="BuyHold_2", color=COLORS["BuyHold_2"], linewidth=2)

    ax.set_title(f"Growth Curve — {freq_name} Rebalancing", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    fig.tight_layout()

    if save:
        fname = f"Growth_Curve_{freq_name.replace(' ', '_')}.png"
        fig.savefig(os.path.join(VIZ_DIR, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Growth curve dashboard — all frequencies
# ═══════════════════════════════════════════════════════════════════════════

def plot_all_growth_curves(runs: dict, ref: dict, save: bool = True) -> None:
    """2×3 subplot dashboard of growth curves (Daily excluded)."""
    _ensure_viz_dir()
    use = [(n, u, f) for n, u, f in REBALANCING_SETTINGS
           if n != "Daily" and (u, f) in runs]
    if not use:
        print("[skip] No runs to plot.")
        return

    rows, cols = 2, 3
    fig, axes  = plt.subplots(rows, cols, figsize=(18, 9), sharey=True)
    axes       = axes.ravel()

    for ax, (name, unit, freq) in zip(axes, use):
        _, df = runs[(unit, freq)]
        yrs   = df.index.astype(int)
        for s in _ROTATION_STRATS:
            ax.plot(yrs, df[s].values, color=COLORS[s], linewidth=2)
        ax.plot(ref["years_index"], ref["curve"].values,
                color=COLORS["BuyHold_2"], linewidth=2)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Year")
        ax.set_ylabel("Value ($)")

    for idx in range(len(use), rows * cols):
        axes[idx].set_visible(False)

    handles = [plt.Line2D([0], [0], color=COLORS[s], lw=3, label=s)
               for s in STRATEGIES]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               frameon=False, fontsize=10)
    fig.suptitle("Growth Curves (Year-end Equity) by Rebalancing Frequency",
                 y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        fig.savefig(os.path.join(VIZ_DIR, "Growth_Curves_All_Frequencies.pdf"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 4. All metric bar charts  (CAGR, Sharpe, Vol, MDD, Sortino, IR)
# ═══════════════════════════════════════════════════════════════════════════

def plot_metric_bars(runs: dict, ref: dict, save: bool = True) -> None:
    """
    Generate all seven metric bar charts across rebalancing frequencies:
      1. Annual Return (CAGR %)
      2. Sharpe Ratio
      3. Volatility (%)
      4. Max Drawdown (%)
      5. Sortino Ratio
      6. Tracking Error (%) vs LUATTRUU
      7. Information Ratio vs LUATTRUU
    """
    # 1. CAGR
    labels, data = _collect_metric(runs, ref, "{g}", "cagr_pct", agg="cagr")
    _save_barplot("Annual Return (%)", data, labels, zero_line=True, save=save)

    # 2. Sharpe
    labels, data = _collect_metric(runs, ref, "Sharpe_{g}", "sharpe", agg="mean")
    _save_barplot("Sharpe Ratio", data, labels, save=save)

    # 3. Volatility
    labels, data = _collect_metric(runs, ref, "Volatility_{g}", "vol_pct", agg="mean")
    _save_barplot("Volatility (%)", data, labels, save=save)

    # 4. Max Drawdown
    labels, data = _collect_metric(runs, ref, "{g}", "mdd_pct", agg="mdd_values")
    _save_barplot("Max Drawdown (%)", data, labels, zero_line=True, save=save)

    # 5. Sortino
    labels, data = _collect_metric(runs, ref, "Sortino_{g}", "sortino", agg="mean")
    _save_barplot("Sortino Ratio", data, labels, zero_line=True, save=save)


    # 7. Information Ratio — full-period scalar, same value in every row
    freq_labels = []
    ir_by_group = {g: [] for g in STRATEGIES}
    for name, unit, freq in REBALANCING_SETTINGS:
        key = (unit, freq)
        if key not in runs or name == "Daily":
            continue
        freq_labels.append(name)
        _, df = runs[key]
        for g in STRATEGIES:
            col = f"IR_{g}"
            # All rows hold the same full-period scalar — just take the first
            ir_by_group[g].append(
                float(df[col].iloc[0]) if col in df.columns else np.nan
            )
    _save_barplot("Information Ratio vs LUATTRUU (Full Period)", ir_by_group,
                  freq_labels, zero_line=True, save=save)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Sortino & IR only  (standalone, also called by plot_metric_bars)
# ═══════════════════════════════════════════════════════════════════════════

def plot_sortino_ir_bars(runs: dict, ref: dict, save: bool = True) -> None:
    """
    Standalone bar charts for Sortino Ratio, Tracking Error, and IR.
    Useful for calling independently without running all charts.
    """
    # Sortino
    labels, data = _collect_metric(runs, ref, "Sortino_{g}", "sortino", agg="mean")
    _save_barplot("Sortino Ratio", data, labels, zero_line=True, save=save)


    # IR — full-period scalar, same value in every row
    freq_labels = []
    ir_by_group = {g: [] for g in STRATEGIES}
    for name, unit, freq in REBALANCING_SETTINGS:
        key = (unit, freq)
        if key not in runs or name == "Daily":
            continue
        freq_labels.append(name)
        _, df = runs[key]
        for g in STRATEGIES:
            col = f"IR_{g}"
            ir_by_group[g].append(
                float(df[col].iloc[0]) if col in df.columns else np.nan
            )
    _save_barplot("Information Ratio vs LUATTRUU (Full Period)", ir_by_group,
                  freq_labels, zero_line=True, save=save)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Reference Buy&Hold dict
# ═══════════════════════════════════════════════════════════════════════════

def build_ref_buyhold(runs: dict, prefer: tuple = ("months", 12)) -> dict:
    """
    Extract Buy&Hold reference metrics from one chosen frequency's annual_df.

    Parameters
    ----------
    runs   : {(unit, freq): (label, annual_df)}
    prefer : (unit, freq) key to use as the reference run (default: Annual)

    Returns
    -------
    dict with keys: final, cagr_pct, mdd_pct, sharpe, vol_pct, sortino,
                    curve, years_index
    """
    _, ref_df = runs[prefer] if prefer in runs else next(iter(runs.values()))

    def _col_mean(col):
        return float(ref_df[col].mean()) if col in ref_df.columns else np.nan

    return {
        "final":       float(ref_df["BuyHold_2"].iloc[-1]),
        "cagr_pct":    cagr_from_values(ref_df["BuyHold_2"]) * 100.0,
        "mdd_pct":     mdd_from_values(ref_df["BuyHold_2"]),
        "sharpe":      _col_mean("Sharpe_BuyHold_2"),
        "vol_pct":     _col_mean("Volatility_BuyHold_2"),
        "sortino":     _col_mean("Sortino_BuyHold_2"),
        "te_pct":      0.0,   # BuyHold TE vs itself = 0
        "curve":       ref_df["BuyHold_2"].copy(),
        "years_index": ref_df.index.astype(int),
    }