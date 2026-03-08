"""
statistical_tests.py
Four-tier statistical validation of the U.S. Treasury ETF rotation strategy.

Test hierarchy (mirrors paper Section 4):

    🥇  Primary   — Newey-West HAC Excess Return Test (statsmodels + fallback)
    🥈  Secondary — Lo (2002) Sharpe SE  |  Skip-Period Comparison
    🥉  Support   — Bootstrap Sharpe Difference  |  Lookback Sensitivity

Everything runs across three strategies (Winners, Median, Losers) and three
rebalancing frequencies (Annual, Semi-Annual, Quarterly), then saves CSVs
and five Median-focused charts.

Usage:
    python run_statistical_tests.py
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
from scipy import stats as sp
from typing import Tuple, Dict

warnings.filterwarnings("ignore")

# ── Try importing statsmodels for robust NW-HAC ─────────────────────────
try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ── project imports ────────────────────────────────────────────────────────
from src.transaction_costs import run_backtest_with_costs
from src.config import (
    BASE_ROUND_TRIP_BPS, RF_ANNUAL, START_YEAR, END_YEAR, INITIAL_CASH,
    COLORS, STRAT_COLORS,
)

# ── output directories ─────────────────────────────────────────────────────
ST_TABLES = os.path.join("results", "statistical_tests", "tables")
ST_VIZ    = os.path.join("results", "statistical_tests", "visualizations")
os.makedirs(ST_TABLES, exist_ok=True)
os.makedirs(ST_VIZ,    exist_ok=True)

# ── strategy / frequency registry ─────────────────────────────────────────
STRATS = ["Winners", "Median", "Losers"]

#   (display name,  backtest unit,  backtest freq)
FREQS = [
    ("Annual",      "months", 12),
    ("Semi-Annual", "months",  6),
    ("Quarterly",   "months",  3),
]

# ── Unified colors from config.py ─────────────────────────────────────────
# Use config-defined palette everywhere for consistency
_COLORS = {
    "Winners":  STRAT_COLORS.get("Winners", COLORS.get("Winners", "#008000")),
    "Median":   STRAT_COLORS.get("Median",  COLORS.get("Median",  "#000000")),
    "Losers":   STRAT_COLORS.get("Losers",  COLORS.get("Losers",  "#0000FF")),
    "BuyHold":  COLORS.get("BuyHold_2", "#FF0000"),
}

# ── Plot style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "figure.dpi":         100,
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.labelsize":     10,
})

BG_COLOR = "white"


# ══════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════

def _period_returns(wealth_df: pd.DataFrame) -> pd.DataFrame:
    """Convert wealth index to per-period returns."""
    pret = wealth_df.pct_change().dropna()
    rename = {f"{s}_Net": s for s in STRATS}
    return pret.rename(columns=rename)


def _excess_returns(wealth_df: pd.DataFrame) -> pd.DataFrame:
    """Per-period excess return: strategy (net) minus buy-and-hold."""
    pret = _period_returns(wealth_df)
    if "BuyHold" not in pret.columns:
        raise KeyError("BuyHold column missing from backtest output.")
    available = [s for s in STRATS if s in pret.columns]
    return pret[available + ["BuyHold"]].sub(pret["BuyHold"], axis=0).drop(columns="BuyHold")


def _ppy(unit: str, freq: int) -> float:
    """Periods per year for a given time unit and frequency."""
    return {"months": 12.0 / freq, "weeks": 52.0 / freq, "days": 252.0 / freq}[unit]


def _annualize_sharpe(r: np.ndarray, ppy: float) -> float:
    """Annualized Sharpe from period-level returns (no risk-free deduction)."""
    s = r.std(ddof=1)
    return (r.mean() / s) * np.sqrt(ppy) if s > 0 else np.nan


def _nw_lag(n: int) -> int:
    """Newey-West (1994) data-driven lag selection."""
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))



# ══════════════════════════════════════════════════════════════════════════
#  🥇  PRIMARY: Newey-West HAC Excess Return Test
# ══════════════════════════════════════════════════════════════════════════

def _nw_test_statsmodels(x: np.ndarray) -> Tuple[float, int, float, float, Tuple[float, float]]:
    """
    Newey-West HAC t-test using statsmodels OLS with HAC covariance.
    H0: E[x] = 0  →  regress x on a constant, use NW-HAC standard errors.
    """
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        nans = np.nan
        return nans, 0, nans, nans, (nans, nans)

    lag = _nw_lag(n)
    X = sm.add_constant(np.ones(n))  # intercept only
    model = sm.OLS(x, X[:, :1])  # single constant regressor
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lag})

    xbar  = results.params[0]
    se    = results.bse[0]
    t     = results.tvalues[0]
    p     = results.pvalues[0]
    ci    = results.conf_int(alpha=0.05)[0]

    return float(xbar), lag, float(t), float(p), (float(ci[0]), float(ci[1]))


def _nw_test_manual(x: np.ndarray) -> Tuple[float, int, float, float, Tuple[float, float]]:
    """
    Manual Newey-West HAC t-test (fallback when statsmodels unavailable).
    Uses Bartlett kernel with Newey-West (1994) lag selection.
    """
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        nans = np.nan
        return nans, 0, nans, nans, (nans, nans)

    lag  = _nw_lag(n)
    xbar = x.mean()
    e    = x - xbar

    V = float(np.dot(e, e)) / n
    for j in range(1, lag + 1):
        w   = 1.0 - j / (lag + 1.0)
        cov = float(np.dot(e[j:], e[:-j])) / n
        V  += 2.0 * w * cov

    se     = np.sqrt(max(V, 1e-20) / n)
    t      = xbar / se
    p      = float(2.0 * sp.t.sf(abs(t), df=n - 1))
    t_crit = float(sp.t.ppf(0.975, df=n - 1))

    return float(xbar), lag, float(t), p, (xbar - t_crit * se, xbar + t_crit * se)


def _nw_test(x: np.ndarray) -> Tuple[float, int, float, float, Tuple[float, float]]:
    """Dispatch to statsmodels if available, otherwise manual."""
    if HAS_STATSMODELS:
        return _nw_test_statsmodels(x)
    return _nw_test_manual(x)


def run_newey_west() -> pd.DataFrame:
    """
    Newey-West HAC test on annualized excess returns.
    Also reports a plain t-test and Wilcoxon signed-rank test for comparison.
    """
    rows = []
    for freq_name, unit, freq in FREQS:
        wealth, _ = run_backtest_with_costs(rebalancing_unit=unit,
                                            rebalancing_frequency=freq)
        exc  = _excess_returns(wealth)
        ppy_ = _ppy(unit, freq)

        for strat in STRATS:
            if strat not in exc.columns:
                continue

            raw = exc[strat].dropna().values
            ann = raw * ppy_ * 100.0

            mean, lag, t, p_nw, (ci_lo, ci_hi) = _nw_test(ann)

            _, p_plain = sp.ttest_1samp(ann, 0.0) if len(ann) >= 3 else (np.nan, np.nan)

            p_wil = np.nan
            if len(ann) >= 10:
                try:
                    _, p_wil = sp.wilcoxon(ann, alternative="two-sided")
                except Exception:
                    pass

            def _r(v, d=4):
                return round(v, d) if (v is not None and not np.isnan(v)) else np.nan

            rows.append({
                "Frequency":        freq_name,
                "Strategy":         strat,
                "N":                len(ann),
                "NW_Lag":           lag,
                "Mean_Excess_pct":  _r(mean),
                "NW_t":             _r(t),
                "NW_p":             _r(p_nw),
                "CI_Lower_pct":     _r(ci_lo),
                "CI_Upper_pct":     _r(ci_hi),
                "Significant_5pct": "Yes" if (not np.isnan(p_nw) and p_nw < 0.05) else "No",
                "Plain_t_p":        _r(p_plain),
                "Wilcoxon_p":       _r(p_wil),
                "HAC_Engine":       "statsmodels" if HAS_STATSMODELS else "manual",
            })

    result = pd.DataFrame(rows)
    if result.empty:
        warnings.warn(
            "run_newey_west() produced no rows — no strategy columns found.",
            stacklevel=2,
        )
    return result


# ══════════════════════════════════════════════════════════════════════════
#  🥈  SECONDARY A: Lo (2002) Sharpe SE + Ljung-Box Autocorrelation
# ══════════════════════════════════════════════════════════════════════════

def run_lo_sharpe() -> pd.DataFrame:
    """
    Lo (2002) standard error for the annualized Sharpe ratio.
    Uses statsmodels Ljung-Box when available, manual Q(1) otherwise.
    """
    rows = []
    for freq_name, unit, freq in FREQS:
        ppy_  = _ppy(unit, freq)
        ann_f = np.sqrt(ppy_)

        wealth, _ = run_backtest_with_costs(rebalancing_unit=unit,
                                            rebalancing_frequency=freq)
        pret = _period_returns(wealth)

        all_strats = STRATS + (["BuyHold"] if "BuyHold" in pret.columns else [])

        for strat in all_strats:
            if strat not in pret.columns:
                continue
            r = pret[strat].dropna().values
            n = len(r)
            if n < 5:
                continue

            sr_p   = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else np.nan
            sr_ann = sr_p * ann_f if not np.isnan(sr_p) else np.nan

            rho1   = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if n >= 4 else np.nan

            # Ljung-Box via statsmodels if available
            if HAS_STATSMODELS and n >= 10:
                try:
                    lb_result = acorr_ljungbox(r, lags=[1], return_df=True)
                    q1   = float(lb_result["lb_stat"].iloc[0])
                    p_lb = float(lb_result["lb_pvalue"].iloc[0])
                except Exception:
                    q1   = n * (n + 2) * rho1**2 / (n - 1) if not np.isnan(rho1) else np.nan
                    p_lb = float(sp.chi2.sf(q1, df=1)) if not np.isnan(q1) else np.nan
            else:
                q1   = n * (n + 2) * rho1**2 / (n - 1) if not np.isnan(rho1) else np.nan
                p_lb = float(sp.chi2.sf(q1, df=1)) if not np.isnan(q1) else np.nan

            if not np.isnan(sr_p) and not np.isnan(rho1):
                iid_term = (1.0 / n) * (1.0 + 0.5 * sr_p**2) * ppy_
                ac_term  = (2.0 * rho1 / (1.0 - rho1)) * ppy_ / n if abs(rho1) < 0.999 else 0.0
                lo_se    = float(np.sqrt(max(iid_term + ac_term, 1e-20)))
                lo_t     = sr_ann / lo_se
                lo_p     = float(2.0 * sp.t.sf(abs(lo_t), df=n - 1))
            else:
                lo_se = lo_t = lo_p = np.nan

            rows.append({
                "Frequency":        freq_name,
                "Strategy":         strat,
                "N":                n,
                "Sharpe_Ann":       round(sr_ann, 4) if not np.isnan(sr_ann) else np.nan,
                "AR1_rho":          round(rho1,   4) if not np.isnan(rho1)   else np.nan,
                "LjungBox_Q1":      round(q1,     3) if not np.isnan(q1)     else np.nan,
                "LjungBox_p":       round(p_lb,   4) if not np.isnan(p_lb)   else np.nan,
                "AC_Present":       "Yes" if (not np.isnan(p_lb) and p_lb < 0.10) else "No",
                "Lo_SE":            round(lo_se,  4) if not np.isnan(lo_se)  else np.nan,
                "Lo_t":             round(lo_t,   4) if not np.isnan(lo_t)   else np.nan,
                "Lo_p":             round(lo_p,   4) if not np.isnan(lo_p)   else np.nan,
                "Significant_5pct": "Yes" if (not np.isnan(lo_p) and lo_p < 0.05) else "No",
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
#  🥈  SECONDARY B: Skip-Period Comparison  (look-ahead bias check)
# ══════════════════════════════════════════════════════════════════════════

def run_skip_period() -> pd.DataFrame:
    """
    Compare base-case vs skip-1 on CAGR, Sharpe, and max drawdown.
    """
    rows = []
    for freq_name, unit, freq in FREQS:
        ppy_ = _ppy(unit, freq)

        w_base, _ = run_backtest_with_costs(rebalancing_unit=unit,
                                            rebalancing_frequency=freq,
                                            signal_lag=0)
        w_skip, _ = run_backtest_with_costs(rebalancing_unit=unit,
                                            rebalancing_frequency=freq,
                                            signal_lag=1)

        def _metrics(w, strat):
            if strat not in w.columns:
                return {k: np.nan for k in ("cagr", "exc", "sr", "mdd")}
            col = w[strat]
            bh  = w.get("BuyHold", col)
            n   = len(col)
            cagr = ((col.iloc[-1] / col.iloc[0]) ** (ppy_ / (n - 1)) - 1) * 100
            exc  = cagr - ((bh.iloc[-1] / bh.iloc[0]) ** (ppy_ / (n - 1)) - 1) * 100
            r_   = col.pct_change().dropna().values
            sr   = _annualize_sharpe(r_, ppy_)
            roll = col.cummax()
            mdd  = ((col - roll) / roll).min() * 100
            return {"cagr": round(cagr, 4), "exc": round(exc, 4),
                    "sr":   round(sr,   4), "mdd": round(mdd, 4)}

        for strat in STRATS:
            b = _metrics(w_base, f"{strat}_Net")
            s = _metrics(w_skip, f"{strat}_Net")
            rows.append({
                "Frequency":         freq_name,
                "Strategy":          strat,
                "Base_CAGR":         b["cagr"],
                "Base_Excess_CAGR":  b["exc"],
                "Base_Sharpe":       b["sr"],
                "Base_MDD":          b["mdd"],
                "Skip1_CAGR":        s["cagr"],
                "Skip1_Excess_CAGR": s["exc"],
                "Skip1_Sharpe":      s["sr"],
                "Skip1_MDD":         s["mdd"],
                "CAGR_Delta":        round(s["cagr"] - b["cagr"], 4),
                "Sharpe_Delta":      round(s["sr"]   - b["sr"],   4),
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
#  🥉  SUPPORT A: Block Bootstrap Sharpe Difference
# ══════════════════════════════════════════════════════════════════════════

def _block_bootstrap(s_r: np.ndarray, b_r: np.ndarray,
                     n_boot: int, ann_f: float,
                     seed: int = 42) -> Tuple[float, float, float, float, float]:
    """
    Circular block bootstrap of annualized Sharpe(strategy) - Sharpe(benchmark).
    Block size ~ n^(1/3), consistent with Politis & Romano (1994).
    """
    rng = np.random.RandomState(seed)
    n   = len(s_r)
    bsz = max(2, int(np.ceil(n ** (1.0 / 3.0))))

    def _sharpe(r):
        s = r.std(ddof=1)
        return (r.mean() / s) * ann_f if s > 0 else np.nan

    obs   = _sharpe(s_r) - _sharpe(b_r)
    diffs = []

    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            start = rng.randint(0, n)
            idx.extend(range(start, start + bsz))
        idx = [i % n for i in idx[:n]]
        d   = _sharpe(s_r[idx]) - _sharpe(b_r[idx])
        if not np.isnan(d):
            diffs.append(d)

    diffs = np.array(diffs)
    return (
        float(obs),
        float(diffs.mean()),
        float((diffs > 0).mean() * 100),
        float(np.percentile(diffs,  2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def run_bootstrap_sharpe(n_boot: int = 10_000) -> pd.DataFrame:
    """Block bootstrap annualized Sharpe difference across all strategies and frequencies."""
    rows = []
    for freq_name, unit, freq in FREQS:
        ppy_  = _ppy(unit, freq)
        ann_f = np.sqrt(ppy_)

        wealth, _ = run_backtest_with_costs(rebalancing_unit=unit,
                                            rebalancing_frequency=freq)
        pret = _period_returns(wealth)

        for strat in STRATS:
            if strat not in pret.columns or "BuyHold" not in pret.columns:
                continue
            common = pret[[strat, "BuyHold"]].dropna()
            s_r    = common[strat].values
            b_r    = common["BuyHold"].values
            if len(s_r) < 5:
                continue

            obs, mean, pct, ci_lo, ci_hi = _block_bootstrap(s_r, b_r, n_boot, ann_f)
            sig = "Yes" if (ci_lo > 0 or ci_hi < 0) else "No"

            rows.append({
                "Frequency":        freq_name,
                "Strategy":         strat,
                "Sharpe_Strategy":  round(_annualize_sharpe(s_r, ppy_), 4),
                "Sharpe_BM":        round(_annualize_sharpe(b_r, ppy_), 4),
                "Sharpe_Diff":      round(obs,    4),
                "Boot_Mean":        round(mean,   4),
                "CI_Lower":         round(ci_lo,  4),
                "CI_Upper":         round(ci_hi,  4),
                "Pct_Positive":     round(pct,    1),
                "Significant_5pct": sig,
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
#  🥉  SUPPORT B: Lookback Sensitivity
# ══════════════════════════════════════════════════════════════════════════

def run_lookback_sensitivity() -> pd.DataFrame:
    """
    Compare CAGR, Sharpe, and MDD across different rebalancing frequencies,
    treating each frequency as a proxy for a different effective lookback window.
    """
    rows = []
    for freq_name, unit, freq in FREQS:
        ppy_ = _ppy(unit, freq)

        w, _ = run_backtest_with_costs(rebalancing_unit=unit,
                                       rebalancing_frequency=freq)

        for strat in STRATS:
            col  = f"{strat}_Net"
            bh   = w.get("BuyHold", w[col])
            if col not in w.columns:
                continue
            vals = w[col]
            n    = len(vals)
            cagr = ((vals.iloc[-1] / vals.iloc[0]) ** (ppy_ / (n - 1)) - 1) * 100
            exc  = cagr - ((bh.iloc[-1] / bh.iloc[0]) ** (ppy_ / (n - 1)) - 1) * 100
            r_   = vals.pct_change().dropna().values
            sr   = _annualize_sharpe(r_, ppy_)
            roll = vals.cummax()
            mdd  = ((vals - roll) / roll).min() * 100

            rows.append({
                "Frequency":   freq_name,
                "Strategy":    strat,
                "CAGR_pct":    round(cagr, 4),
                "Excess_CAGR": round(exc,  4),
                "Sharpe":      round(sr,   4),
                "MDD_pct":     round(mdd,  4),
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
#  Design constants for charts
# ══════════════════════════════════════════════════════════════════════════

_BG         = "#ffffff"
_GRID_ALPHA = 0.12
_GRID_COLOR = "#888888"
_EDGE_COLOR = "#dddddd"
_TITLE_SZ   = 13
_LABEL_SZ   = 10
_TICK_SZ    = 9
_ANNOT_SZ   = 9
_FREQS_LIST = ["Annual", "Semi-Annual", "Quarterly"]


def _apply_style(ax, grid_axis="both"):
    ax.set_facecolor(_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#555555", labelsize=_TICK_SZ)
    ax.grid(axis=grid_axis, alpha=_GRID_ALPHA, color=_GRID_COLOR, linestyle="-")


def _save(fig, fname):
    fig.tight_layout()
    fig.savefig(os.path.join(ST_VIZ, fname),
                dpi=300, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  Charts
# ══════════════════════════════════════════════════════════════════════════

def chart_nw_forest(nw: pd.DataFrame):
    """
    Minimal forest plot — CI bars + point estimates only.
    No numbers table (the CSV has all that). Just the visual story:
    which strategies have CIs that exclude zero?
    """
    if nw.empty or "Strategy" not in nw.columns:
        raise RuntimeError("Newey-West DataFrame is empty.")

    freq_order = _FREQS_LIST

    # Collect display rows
    display = []
    for freq in freq_order:
        for strat in STRATS:
            row = nw[(nw["Frequency"] == freq) & (nw["Strategy"] == strat)]
            if row.empty:
                continue
            r = row.iloc[0]
            display.append({
                "freq": freq, "strat": strat,
                "mean": r["Mean_Excess_pct"],
                "lo": r["CI_Lower_pct"], "hi": r["CI_Upper_pct"],
                "sig": r["Significant_5pct"] == "Yes",
            })

    n = len(display)
    fig, ax = plt.subplots(figsize=(10, 0.6 * n + 1.5), facecolor=_BG)
    _apply_style(ax, "x")
    ax.spines["left"].set_visible(False)

    # Alternating frequency group shading
    prev_freq, grp_start, shade = None, 0, False
    for i, r in enumerate(display):
        if r["freq"] != prev_freq:
            if prev_freq is not None and shade:
                ax.axhspan(grp_start - 0.5, i - 0.5, color="#f0f0f0", zorder=0)
            shade = not shade
            grp_start = i
            prev_freq = r["freq"]
    if shade:
        ax.axhspan(grp_start - 0.5, n - 0.5, color="#f0f0f0", zorder=0)

    # Plot
    for i, r in enumerate(display):
        color = _COLORS[r["strat"]]
        ax.plot([r["lo"], r["hi"]], [i, i], color=color, lw=4.5, alpha=0.35,
                solid_capstyle="round", zorder=2)
        marker = "D" if r["sig"] else "o"
        ax.plot(r["mean"], i, marker, color=color, ms=8,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4)

    ax.axvline(0, color="#cc3333", lw=1.2, ls="--", alpha=0.5, zorder=1)

    # Y-axis: strategy labels, colored
    ax.set_yticks(range(n))
    ylabels = [r["strat"] for r in display]
    ax.set_yticklabels(ylabels, fontsize=9.5)
    for tick, r in zip(ax.get_yticklabels(), display):
        tick.set_color(_COLORS[r["strat"]])
        tick.set_fontweight("bold")

    # Frequency group labels in left margin
    i = 0
    for freq in freq_order:
        group = [j for j, r in enumerate(display) if r["freq"] == freq]
        if group:
            mid = np.mean(group)
            ax.text(-0.02, mid, freq, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=10, fontweight="bold",
                    color="#444444")
            i += len(group)

    ax.set_xlabel("Mean Annualized Excess Return (%)  —  NW HAC 95% CI",
                  fontsize=_LABEL_SZ)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()

    # Compact legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=_COLORS[s], label=s, alpha=0.8) for s in STRATS]
    ax.legend(handles=handles, fontsize=8, loc="lower right",
              framealpha=0.95, edgecolor=_EDGE_COLOR)

    engine = nw["HAC_Engine"].iloc[0] if "HAC_Engine" in nw.columns else "manual"
    fig.suptitle("Newey-West HAC Test: Excess Return vs Buy-and-Hold",
                 fontsize=_TITLE_SZ, fontweight="bold", y=0.99)
    fig.text(0.5, 0.94,
             f"net of costs  |  ◆ = p < 0.05  |  engine: {engine}",
             fontsize=9, color="#888888", ha="center")

    fig.subplots_adjust(left=0.25)
    _save(fig, "nw_forest.png")


def chart_lo_sharpe(lo: pd.DataFrame):
    """Two-panel: Sharpe bars + AR(1) autocorrelation."""
    show = ["Median", "BuyHold"]
    x = np.arange(len(_FREQS_LIST))
    w = 0.30
    offsets = [-w / 2, w / 2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=_BG)

    ax = axes[0]
    _apply_style(ax, "y")
    for si, strat in enumerate(show):
        srs, ses, sigs = [], [], []
        for f in _FREQS_LIST:
            row = lo[(lo["Strategy"] == strat) & (lo["Frequency"] == f)]
            srs.append(row["Sharpe_Ann"].iloc[0] if not row.empty else np.nan)
            ses.append(row["Lo_SE"].iloc[0] if not row.empty else np.nan)
            sigs.append(row["Significant_5pct"].iloc[0] == "Yes" if not row.empty else False)
        col = _COLORS.get(strat, _COLORS["BuyHold"])
        lbl = strat if strat != "BuyHold" else "Buy & Hold"
        ax.bar(x + offsets[si], srs, w, color=col, alpha=0.85, edgecolor="white", lw=0.8, label=lbl)
        ax.errorbar(x + offsets[si], srs, yerr=ses, fmt="none", ecolor="#444444", capsize=4, lw=1.5)
        for xi, sv, se, sig in zip(x + offsets[si], srs, ses, sigs):
            if sig and not np.isnan(sv):
                ax.text(xi, sv + se + 0.02, "★", ha="center", fontsize=12, color="#cc2222", fontweight="bold")
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(_FREQS_LIST)
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_title("Sharpe + Lo (2002) SE\n★ = p < 0.05", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.95, edgecolor=_EDGE_COLOR)

    ax = axes[1]
    _apply_style(ax, "y")
    for si, strat in enumerate(show):
        rhos, lb_sig = [], []
        for f in _FREQS_LIST:
            row = lo[(lo["Strategy"] == strat) & (lo["Frequency"] == f)]
            rhos.append(row["AR1_rho"].iloc[0] if not row.empty else np.nan)
            lb_sig.append(row["AC_Present"].iloc[0] == "Yes" if not row.empty else False)
        col = _COLORS.get(strat, _COLORS["BuyHold"])
        lbl = strat if strat != "BuyHold" else "Buy & Hold"
        ax.bar(x + offsets[si], rhos, w, color=col, alpha=0.85, edgecolor="white", lw=0.8, label=lbl)
        for xi, rv, sig in zip(x + offsets[si], rhos, lb_sig):
            if sig:
                ax.bar(xi, rv, w, fill=False, edgecolor="#DAA520", lw=2.5)
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(_FREQS_LIST)
    ax.set_ylabel("AR(1) ρ₁")
    ax.set_title("Return Autocorrelation\ngold border = Ljung-Box p < 0.10", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.95, edgecolor=_EDGE_COLOR)

    fig.suptitle("Lo (2002) Sharpe Significance & Return Autocorrelation",
                 fontsize=_TITLE_SZ, fontweight="bold", y=1.03)
    _save(fig, "lo_sharpe.png")


def chart_skip_period(skip: pd.DataFrame):
    """Median-only skip-period comparison."""
    med = (skip[skip["Strategy"] == "Median"].set_index("Frequency").reindex(_FREQS_LIST))
    x = np.arange(len(_FREQS_LIST))
    w = 0.30
    c = _COLORS["Median"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=_BG)
    for ax, base_col, skip_col, ylabel, fmt in zip(
        axes, ["Base_CAGR", "Base_Sharpe"], ["Skip1_CAGR", "Skip1_Sharpe"],
        ["CAGR (%)", "Annualized Sharpe"], [".2f", ".3f"],
    ):
        _apply_style(ax, "y")
        bv, sv = med[base_col].tolist(), med[skip_col].tolist()
        ax.bar(x - w/2, bv, w, label="Base", color=c, alpha=0.9, edgecolor="white", lw=0.8)
        ax.bar(x + w/2, sv, w, label="Skip-1", color=c, alpha=0.35, edgecolor=c, lw=0.8, hatch="///")
        for xi, b, s in zip(x, bv, sv):
            if not np.isnan(b):
                ax.text(xi - w/2, b + 0.012, f"{b:{fmt}}", ha="center", fontsize=8, fontweight="bold")
            if not np.isnan(s):
                ax.text(xi + w/2, s + 0.012, f"{s:{fmt}}", ha="center", fontsize=8, color="#777")
        ax.set_xticks(x); ax.set_xticklabels(_FREQS_LIST)
        ax.set_ylabel(ylabel); ax.legend(fontsize=8, framealpha=0.95, edgecolor=_EDGE_COLOR)
    fig.suptitle("Skip-Period Robustness: Median Strategy\nBase vs 1-Period Execution Lag  (net of costs)",
                 fontsize=_TITLE_SZ, fontweight="bold", y=1.04)
    _save(fig, "skip_period.png")


def chart_bootstrap(boot: pd.DataFrame):
    """Bootstrap Sharpe difference for Median."""
    rows_m = [boot[(boot["Strategy"] == "Median") & (boot["Frequency"] == f)] for f in _FREQS_LIST]
    diffs = [r["Sharpe_Diff"].iloc[0] if not r.empty else np.nan for r in rows_m]
    ci_lo = [r["CI_Lower"].iloc[0] if not r.empty else np.nan for r in rows_m]
    ci_hi = [r["CI_Upper"].iloc[0] if not r.empty else np.nan for r in rows_m]
    pcts  = [r["Pct_Positive"].iloc[0] if not r.empty else np.nan for r in rows_m]
    err_lo = [d - lo for d, lo in zip(diffs, ci_lo)]
    err_hi = [hi - d for hi, d in zip(ci_hi, diffs)]
    x = np.arange(len(_FREQS_LIST))
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_BG)
    _apply_style(ax, "y")
    ax.bar(x, diffs, 0.5, color=_COLORS["Median"], alpha=0.85, edgecolor="white", lw=0.8)
    ax.errorbar(x, diffs, yerr=[err_lo, err_hi], fmt="none", ecolor="#444444", capsize=6, lw=1.8)
    for xi, dv, pct in zip(x, diffs, pcts):
        if not np.isnan(pct):
            off = 0.008 if dv >= 0 else -0.025
            ax.text(xi, dv + off, f"{pct:.0f}% positive", ha="center", fontsize=9, fontweight="bold", color="#333")
    ax.axhline(0, color="#cc3333", lw=1.2, ls="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(_FREQS_LIST)
    ax.set_ylabel("Annualized Sharpe Difference  (Median − Benchmark)")
    ax.set_title("Bootstrap Sharpe Difference: Median vs Buy-and-Hold\n"
                 "10K circular block bootstrap  |  error bars = 95% CI",
                 fontsize=_TITLE_SZ, fontweight="bold")
    _save(fig, "bootstrap_sharpe.png")


def chart_lookback(lb: pd.DataFrame):
    """Heatmaps — Strategy × Frequency."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), facecolor=_BG)
    for ax, metric, title, fmt, cmap in zip(
        axes, ["Sharpe", "CAGR_pct"],
        ["Annualized Sharpe (net)", "CAGR % (net)"],
        [".3f", ".2f"], ["YlOrBr", "YlGn"],
    ):
        pivot = pd.DataFrame(index=STRATS, columns=_FREQS_LIST, dtype=float)
        for strat in STRATS:
            for freq in _FREQS_LIST:
                row = lb[(lb["Strategy"] == strat) & (lb["Frequency"] == freq)]
                if not row.empty:
                    pivot.loc[strat, freq] = row[metric].iloc[0]
        pivot = pivot.astype(float)
        sns.heatmap(pivot, ax=ax, annot=True, fmt=fmt, cmap=cmap,
                    linewidths=2, linecolor="white",
                    cbar_kws={"shrink": 0.75},
                    annot_kws={"size": 13, "fontweight": "bold"}, square=False)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(""); ax.set_xlabel("")
        ax.tick_params(axis="y", labelsize=10, rotation=0)
        ax.tick_params(axis="x", labelsize=9.5)
        for i, label in enumerate(ax.get_yticklabels()):
            label.set_color(_COLORS[STRATS[i]])
            label.set_fontweight("bold")
    fig.suptitle("Lookback Sensitivity: Strategy × Frequency",
                 fontsize=_TITLE_SZ, fontweight="bold", y=1.04)
    _save(fig, "lookback_sharpe.png")


# ══════════════════════════════════════════════════════════════════════════
#  Master runner
# ══════════════════════════════════════════════════════════════════════════

def run_all(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Run all four statistical tests in priority order and save outputs."""
    sep = "=" * 72
    print(sep)
    print("  STATISTICAL VALIDATION — U.S. Treasury ETF Rotation Strategy")
    print(f"  Strategies : {', '.join(STRATS)}")
    print( "  Frequencies: Annual / Semi-Annual / Quarterly")
    print(f"  Costs      : {BASE_ROUND_TRIP_BPS} bps round-trip  |  Rf: {RF_ANNUAL}")
    print(f"  Sample     : {START_YEAR}–{END_YEAR}  |  Initial ${INITIAL_CASH:,}")
    print(f"  HAC Engine : {'statsmodels (OLS + HAC)' if HAS_STATSMODELS else 'manual (Bartlett kernel)'}")
    print(sep)

    print("\n[1/4]  Newey-West HAC excess return test…")
    nw = run_newey_west()
    nw.to_csv(os.path.join(ST_TABLES, "nw_excess_return.csv"), index=False)
    if verbose:
        print(nw.to_string(index=False))

    print("\n[2/4]  Lo (2002) Sharpe SE + Ljung-Box autocorrelation…")
    lo = run_lo_sharpe()
    lo.to_csv(os.path.join(ST_TABLES, "lo_sharpe.csv"), index=False)
    if verbose:
        cols = ["Frequency", "Strategy", "Sharpe_Ann", "AR1_rho",
                "LjungBox_p", "Lo_t", "Lo_p", "Significant_5pct"]
        print(lo[cols].to_string(index=False))

    print("\n[3/4]  Skip-period robustness (1-period execution lag)…")
    skip = run_skip_period()
    skip.to_csv(os.path.join(ST_TABLES, "skip_period.csv"), index=False)
    if verbose:
        print(skip[skip["Strategy"] == "Median"].to_string(index=False))

    print("\n[4/4]  Bootstrap Sharpe difference + Lookback sensitivity…")
    boot = run_bootstrap_sharpe(n_boot=10_000)
    boot.to_csv(os.path.join(ST_TABLES, "bootstrap_sharpe.csv"), index=False)
    lb = run_lookback_sensitivity()
    lb.to_csv(os.path.join(ST_TABLES, "lookback_sensitivity.csv"), index=False)
    if verbose:
        print(boot.to_string(index=False))
        print(lb[lb["Strategy"] == "Median"].to_string(index=False))

    print("\n  Generating charts…")
    # chart_nw_forest skipped — NW results presented as a table in the paper
    chart_lo_sharpe(lo);     print("    ✓  lo_sharpe.png")
    chart_skip_period(skip); print("    ✓  skip_period.png")
    chart_bootstrap(boot);   print("    ✓  bootstrap_sharpe.png")
    chart_lookback(lb);      print("    ✓  lookback_sharpe.png")

    print(f"\n{sep}")
    print(f"  Done.  Tables → {ST_TABLES}   Charts → {ST_VIZ}")
    print(sep)

    return {
        "newey_west":  nw,
        "lo_sharpe":   lo,
        "skip_period": skip,
        "bootstrap":   boot,
        "lookback":    lb,
    }
