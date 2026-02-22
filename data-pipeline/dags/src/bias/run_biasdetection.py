"""
Bias detection orchestrator.

Runs all three Phase 15 bias detectors in sequence:
1) financial_bias.py
2) product_bias.py
3) review_bias.py

Terminal output from each detector is printed in this single run.
"""

import argparse
import os
import sys


# Allow local imports when script is run directly.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from financial_bias import run_phase15_financial_bias
from product_bias import run_phase15_product_bias
from review_bias import run_phase15_review_bias


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all Phase 15 bias detectors.")
    parser.add_argument("--skip-financial", action="store_true", help="Skip financial bias detection.")
    parser.add_argument("--skip-product", action="store_true", help="Skip product bias detection.")
    parser.add_argument("--skip-review", action="store_true", help="Skip review bias detection.")
    args = parser.parse_args()

    overall_exit_code = 0

    if not args.skip_financial:
        print("\n==============================")
        print("Running Financial Bias Detection")
        print("==============================")
        try:
            run_phase15_financial_bias()
        except Exception as exc:
            overall_exit_code = 1
            print(f"[ERROR] financial_bias failed: {exc}")

    if not args.skip_product:
        print("\n============================")
        print("Running Product Bias Detection")
        print("============================")
        try:
            run_phase15_product_bias()
        except Exception as exc:
            overall_exit_code = 1
            print(f"[ERROR] product_bias failed: {exc}")

    if not args.skip_review:
        print("\n===========================")
        print("Running Review Bias Detection")
        print("===========================")
        try:
            run_phase15_review_bias()
        except Exception as exc:
            overall_exit_code = 1
            print(f"[ERROR] review_bias failed: {exc}")

    return overall_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
