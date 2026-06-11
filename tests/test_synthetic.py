"""Smoke tests for the YIFE synthetic data generator.

All tests share a single generated DataFrame (session-scoped fixture)
so generate() is called exactly once — fast and deterministic.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest


# ── shared fixture: generate once, reuse across all tests ──────────────────
@pytest.fixture(scope="session")
def df():
    from src.data.generate_synthetic import generate
    return generate(seed=42)


# ── tests ───────────────────────────────────────────────────────────────────
def test_shape(df):
    assert len(df) == 4323, f"Expected 4323 rows, got {len(df)}"


def test_success_rate(df):
    rate = df["success"].mean()
    assert 0.40 <= rate <= 0.55, \
        f"Success rate {rate:.3f} outside expected range [0.40, 0.55]"


def test_all_yife_features_present(df):
    required = [
        "total_funding_usd", "num_funding_rounds", "seed_round_size",
        "team_size", "faang_experience", "elite_edu",
        "github_repo_count", "github_commit_freq",
        "batch_year_encoded", "batch_size",
        "industry_category", "ai_flag",
        "geo_cluster", "tier1_vc_investor",
    ]
    missing = [f for f in required if f not in df.columns]
    assert not missing, f"Missing YIFE features: {missing}"


def test_temporal_split(df):
    train = df[df["batch_year_encoded"] < 2021]
    test  = df[df["batch_year_encoded"] >= 2021]
    assert len(train) > len(test), "Train set must be larger than test set"
    assert len(test) > 0, "Test set must not be empty"


def test_no_all_nan_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    all_nan = [c for c in numeric_cols if df[c].isna().all()]
    assert not all_nan, f"Columns are all-NaN: {all_nan}"


def test_ai_flag_binary(df):
    vals = set(df["ai_flag"].unique())
    assert vals.issubset({0, 1}), f"ai_flag has unexpected values: {vals}"


def test_team_size_range(df):
    assert df["team_size"].min() >= 1, "team_size below 1"
    assert df["team_size"].max() <= 5, "team_size above 5"


def test_no_negative_funding(df):
    assert (df["total_funding_usd"] > 0).all(), "Found non-positive funding values"


def test_batch_year_range(df):
    assert df["batch_year_encoded"].min() >= 2005
    assert df["batch_year_encoded"].max() <= 2024
