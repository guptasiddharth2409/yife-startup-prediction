"""
run_pipeline.py — End-to-end YIFE pipeline runner.

Runs all steps in order:
    1. Feature Engineering
    2. Model Training
    3. Evaluation (metrics + figures)
    4. SHAP Analysis

Usage:
    python src/run_pipeline.py
"""

import time
from utils import print_banner


def main():
    print_banner("YIFE End-to-End Pipeline")
    start = time.time()

    print("\n[1/4] Feature Engineering")
    import feature_engineering
    feature_engineering.run()

    print("\n[2/4] Training Models")
    import train_models
    train_models.run()

    print("\n[3/4] Evaluation")
    import evaluate
    evaluate.run()

    print("\n[4/4] SHAP Analysis")
    import shap_analysis
    shap_analysis.run()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  ✓ Pipeline complete in {elapsed:.1f}s")
    print(f"  Figures saved to: figures/")
    print(f"  Models saved to:  models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
