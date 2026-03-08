#!/usr/bin/env python3
"""
run_transaction_costs.py
========================
Run the complete transaction cost analysis for the semi-annual strategy.

Outputs
-------
results/transaction_costs/tables/          (7 CSV files)
results/transaction_costs/visualizations/  (7 charts)


Usage:
    python run_transaction_costs.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.transaction_costs import run_full_transaction_cost_analysis


if __name__ == "__main__":
    results = run_full_transaction_cost_analysis(verbose=True)

    print("\n" + "=" * 80)
    print("KEY HIGHLIGHTS")
    print("=" * 80)

    ts = results["turnover_summary"]
    print(f"\nTurnover (avg per period):")
    for s in ["Winners", "Median", "Losers"]:
        print(f"  {s:10s}: {ts.loc[s, 'Avg_Turnover_Per_Period']:.1f}%  "
              f"(trades at {ts.loc[s, 'Trade_Frequency_pct']:.0f}% of rebalancings)")

    rm = results["risk_metrics"]
    print(f"\nRisk Metrics (Net of Costs):")
    print(rm[["CAGR_pct", "Sharpe_Ratio", "Max_Drawdown_pct"]].to_string())

    ca      = results["cost_alpha"]
    med_base = ca[(ca["Strategy"] == "Median") & (ca["Scenario"] == "Base (1×)")]
    if not med_base.empty:
        r = med_base.iloc[0]
        print(f"\nMedian (Base cost):")
        print(f"  Gross alpha vs B&H:  {r['Gross_Alpha_pct']:+.3f}%")
        print(f"  Cost drag on CAGR:   {r['Cost_Drag_CAGR_pct']:.4f}%")
        print(f"  Cost consumes:       {r['Cost_as_pct_of_Alpha']}% of alpha")
        print(f"  Net alpha vs B&H:    {r['Net_Alpha_pct']:+.3f}%")

    print(f"\nBreak-Even Costs:")
    for s in ["Winners", "Median", "Losers"]:
        be  = results["breakevens"][s]["breakeven_bps"]
        bel = f"{be:.0f} bps" if be == be else ">200 bps"
        print(f"  {s:10s}: {bel}")

    cf = results["cross_freq"]
    print(f"\nCross-Frequency (Median, base cost):")
    print(cf[["Cost_Drag_CAGR_bps", "Net_vs_BuyHold_pct", "Beats_BuyHold"]].to_string())
