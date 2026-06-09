"""Smoke tests for the synthetic data generator and trainer pipeline."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def test_generate_synthetic():
    from src.data.generate_synthetic import generate
    df = generate()
    assert len(df) == 4323, "Expected 4323 rows"
    assert "success" in df.columns
    assert 0.40 <= df["success"].mean() <= 0.55, "Success rate should be ~45%"
    assert df["ai_flag"].max() == 1
    assert df["team_size"].min() >= 1


def test_all_yife_features_present():
    from src.data.generate_synthetic import generate
    df = generate()
    required = [
        "total_funding_usd", "num_funding_rounds", "seed_round_size",
        "team_size", "faang_experience", "elite_edu",
        "github_repo_count", "github_commit_freq",
        "batch_year_encoded", "batch_size",
        "industry_category", "ai_flag",
        "geo_cluster", "tier1_vc_investor",
    ]
    for feat in required:
        assert feat in df.columns, f"Missing YIFE feature: {feat}"


def test_temporal_split():
    from src.data.generate_synthetic import generate
    df = generate()
    train = df[df["batch_year_encoded"] < 2021]
    test  = df[df["batch_year_encoded"] >= 2021]
    assert len(train) > len(test), "Training set should be larger"
    assert len(test) > 0, "Test set should not be empty"
