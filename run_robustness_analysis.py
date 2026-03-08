"""
Run the full robustness analysis for the US Treasury rotation strategy.
"""

from src.robustness import run_full_robustness_analysis

if __name__ == "__main__":
    run_full_robustness_analysis(verbose=True)
