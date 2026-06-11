"""Integration smoke test: generate -> train -> figures pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_trainer_runs_and_produces_artifacts():
    """Full pipeline: data -> train -> check model + metrics files exist."""
    from src.data.generate_synthetic import generate
    from src.train.trainer import fit_and_eval
    from src.config import MODEL_DIR, LOG_DIR

    generate(seed=42)
    fit_and_eval()

    # At least one model pkl should exist
    pkls = list(MODEL_DIR.glob("*.pkl"))
    assert len(pkls) >= 1, f"No .pkl files found in {MODEL_DIR}"

    metrics_file = LOG_DIR / "metrics_test.json"
    assert metrics_file.exists(), "metrics_test.json not created"

    import json
    with open(metrics_file) as f:
        metrics = json.load(f)
    # XGBoost should reach reasonable AUROC on synthetic data
    assert metrics["xgb"]["auroc"] >= 0.70, \
        f"XGBoost AUROC too low: {metrics['xgb']['auroc']:.4f}"
