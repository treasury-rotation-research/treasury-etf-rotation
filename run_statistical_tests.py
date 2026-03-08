#!/usr/bin/env python3
"""
run_statistical_tests.py
========================
Run the four-tier statistical validation suite for the U.S. Treasury
ETF rotation strategy.

Test hierarchy:
    🥇  Primary   — Newey-West HAC Excess Return Test
    🥈  Secondary — Lo (2002) Sharpe SE  |  Skip-Period Comparison
    🥉  Support   — Bootstrap Sharpe Difference  |  Lookback Sensitivity

Outputs
-------
results/statistical_tests/tables/          (5 CSV files)
results/statistical_tests/visualizations/  (5 charts)

Usage:
    python run_statistical_tests.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.statistical_tests import run_all

if __name__ == "__main__":
    run_all(verbose=True)
