"""
Training Data Orchestrator — Dev Entrypoint.

Flow:
    1. Load financial_profiles, products, and reviews from PostgreSQL
    2. generate_scenarios() → feature computation + Layer 1 (DecisionEngine)
       + Layer 2 (DowngradeEngine) → labeled scenarios
    3. Save labeled training data to CSV

Usage (from model_pipeline/):
    python src/generate_data.py
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data.db_loader import load_financial_profiles, load_products, load_reviews
from features.training_data_generator import generate_scenarios

logger = logging.getLogger(__name__)

LABEL_COL = Config.LABEL_COL


def main():
    parser = argparse.ArgumentParser(
        description="Generate labeled training data for the SavVio model pipeline.",
    )
    parser.add_argument(
        "--n-scenarios", type=int, default=Config.N_SCENARIOS,
        help=f"Number of scenarios to generate (default: {Config.N_SCENARIOS})",
    )
    parser.add_argument(
        "--output", type=str, default=Config.SCENARIO_OUTPUT_PATH,
        help=f"Output CSV path (default: {Config.SCENARIO_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--random-state", type=int, default=Config.RANDOM_STATE,
        help=f"Random seed (default: {Config.RANDOM_STATE})",
    )
    parser.add_argument(
        "--no-graduated", action="store_true",
        help="Use stratified sampling instead of graduated tiers",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SavVio — Training Data Generator")
    print("=" * 60)

    # 1. Load data from PostgreSQL.
    print("\n[1/3] Loading data from PostgreSQL...")
    t0 = time.time()
    financial_df = load_financial_profiles()
    products_df = load_products()
    reviews_df = load_reviews()
    load_time = time.time() - t0
    print(f"      Loaded in {load_time:.1f}s")
    print(f"      Financial profiles: {len(financial_df):,}")
    print(f"      Products:           {len(products_df):,}")
    print(f"      Reviews:            {len(reviews_df):,}")

    # 2. Generate scenarios (features + deterministic engine).
    graduated = not args.no_graduated
    mode = "graduated" if graduated else "stratified"
    print(f"\n[2/3] Generating {args.n_scenarios:,} scenarios ({mode} mode)...")
    t0 = time.time()
    scenarios = generate_scenarios(
        financial_df,
        products_df,
        reviews_df=reviews_df,
        n_scenarios=args.n_scenarios,
        random_state=args.random_state,
        graduated=graduated,
    )
    gen_time = time.time() - t0
    print(f"      Generated {len(scenarios):,} scenarios in {gen_time:.1f}s")

    # 3. Save to CSV.
    print(f"\n[3/3] Saving training data...")
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    scenarios.to_csv(output_path, index=False)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"      Saved to: {output_path}")
    print(f"      File size: {file_size_mb:.1f} MB")

    # Summary.
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Scenarios:     {len(scenarios):,}")
    print(f"  Columns:       {len(scenarios.columns)}")
    print(f"  Label column:  {LABEL_COL}")
    print(f"  Label distribution:")
    for label, count in scenarios[LABEL_COL].value_counts().items():
        pct = count / len(scenarios) * 100
        print(f"    {label:8s}  {count:>6,}  ({pct:.1f}%)")

    if "downgraded" in scenarios.columns:
        n_down = int(scenarios["downgraded"].sum())
        pct_down = n_down / len(scenarios) * 100
        print(f"  Downgraded:    {n_down:,} ({pct_down:.1f}%)")

    if "session_id" in scenarios.columns:
        n_sessions = scenarios["session_id"].nunique()
        print(f"  Sessions:      {n_sessions:,}")

    print(f"  Output path:   {output_path}")
    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
