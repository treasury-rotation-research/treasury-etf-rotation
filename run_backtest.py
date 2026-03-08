#!/usr/bin/env python3
"""
run_backtest.py
===============
Entry point for the U.S. Treasury ETF Rotation Backtest.

Usage
-----
    python run_backtest.py            # run all rebalancing frequencies
    python run_backtest.py --quick    # run only Annual + Semi-Annual
    python run_backtest.py --no-viz   # skip visualization generation
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    ETFS, START_YEAR, END_YEAR, INITIAL_CASH, RF_ANNUAL,
    LUATTRUU_FILENAME, REBALANCING_SETTINGS,
)
from src.backtest import run_etf_rotation
from src.selection_table import semiannual_selection_table
from src.visualizations import (
    build_ref_buyhold,
    plot_final_values,
    plot_growth_curve,
    plot_all_growth_curves,
    plot_metric_bars,
)


def main(quick: bool = False, no_viz: bool = False) -> None:
    settings = REBALANCING_SETTINGS
    if quick:
        settings = [s for s in settings if s[0] in ("Annual", "Semi-Annual")]

    print("=" * 80)
    print("U.S. TREASURY ETF ROTATION BACKTEST")
    print(f"Period : {START_YEAR}–{END_YEAR}  |  Universe: {', '.join(ETFS)}")
    print(f"Rf     : {RF_ANNUAL:.1%}  |  Initial cash: ${INITIAL_CASH}")
    print("=" * 80)

    runs: dict = {}

    for name, unit, freq in settings:
        print(f"\n▶  {name} ({freq} {unit})")
        annual_df, _groups = run_etf_rotation(
            etfs=ETFS,
            start_year=START_YEAR,
            end_year=END_YEAR,
            initial_cash=INITIAL_CASH,
            rebalancing_unit=unit,
            rebalancing_frequency=freq,
            luattruu_filename=LUATTRUU_FILENAME,
            rf_annual=RF_ANNUAL,
            verbose=True,
            save_csv=True,
        )
        if annual_df is not None and not annual_df.empty:
            runs[(unit, freq)] = (name, annual_df)

    if not runs:
        print("\nERROR: No backtest results produced. "
              "Check that data/ contains the required ETF CSVs and LUATTRUU file.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("SEMI-ANNUAL ETF SELECTION MEMBERSHIP TABLE")
    print("=" * 80)
    membership = semiannual_selection_table(
        etfs=ETFS,
        start_year=START_YEAR,
        end_year=END_YEAR,
        drop_years=[START_YEAR],
    )
    print(membership.to_string())

    if no_viz:
        print("\n[--no-viz] Skipping visualization generation.")
    else:
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)

        ref = build_ref_buyhold(runs, prefer=("months", 12))
        plot_final_values(runs, ref)
        plot_growth_curve(runs, ref, "Semi-Annual")
        plot_growth_curve(runs, ref, "Quarterly")
        plot_all_growth_curves(runs, ref)
        plot_metric_bars(runs, ref)

    print("\n✅  Done!  Outputs written to:")
    print("    results/tables/          — annual summary CSVs")
    if not no_viz:
        print("    results/visualizations/  — PNG / PDF charts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="U.S. Treasury ETF Rotation Backtest",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only Annual and Semi-Annual rebalancing frequencies.",
    )
    parser.add_argument(
        "--no-viz", dest="no_viz", action="store_true",
        help="Skip chart generation (useful for headless / CI runs).",
    )
    args = parser.parse_args()
    main(quick=args.quick, no_viz=args.no_viz)
