"""Smoke tests for the YIFE synthetic data generator and pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest


def test_generate_synthetic_shape():
    from src.data.generate_synthetic import generate
    df = generate()
    assert len(df) == 4323, f"Expected 4323 rows, got {len(df)}"


def test_generate_synthetic_success_rate():
    from src.data.generate_synthetic import generate
    df = generate()
    rate = df["success"].mean()
    assert 0.40 <= rate <= 0.55, f"Success rate {rate:.3f} outside expected range [0.40, 0.55]"


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


def test_temporal_split_sizes():
    from src.data.generate_synthetic import generate
    df = generate()
    train = df[df["batch_year_encoded"] < 2021]
    test  = df[df["batch_year_encoded"] >= 2021]
    assert len(train) > len(test), "Training set should be larger than test set"
    assert len(test) > 0, "Test set must not be empty"


def test_no_all_nan_columns():
    from src.data.generate_synthetic import generate
    df = generate()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        assert not df[col].isna().all(), f"Column '{col}' is all NaN"


def test_ai_flag_binary():
    from src.data.generate_synthetic import generate
    df = generate()
    assert set(df["ai_flag"].unique()).issubset({0, 1}), "ai_flag must be binary"


def test_team_size_range():
    from src.data.generate_synthetic import generate
    df = generate()
    assert df["team_size"].min() >= 1
    assert df["team_size"].max() <= 5
