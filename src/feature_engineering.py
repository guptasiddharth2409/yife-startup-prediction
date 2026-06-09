"""
feature_engineering.py — YIFE 14-feature construction.

Loads raw CSVs from data/raw/, engineers all 14 YIFE features,
and writes the processed dataset to data/processed/yife_features.csv.

Usage:
    python src/feature_engineering.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from utils import load_config, set_seed, ensure_dir, print_banner


def load_raw_data(config: dict) -> pd.DataFrame:
    """Load and merge raw YC + Crunchbase CSVs."""
    yc_path = "data/raw/yc_companies.csv"
    cb_path = "data/raw/crunchbase_export.csv"

    if not os.path.exists(yc_path):
        raise FileNotFoundError(
            f"Missing: {yc_path}\n"
            "Please download the YC dataset from Kaggle and place it at data/raw/yc_companies.csv\n"
            "See data/README.md for instructions."
        )

    yc = pd.read_csv(yc_path)
    print(f"  Loaded YC data: {len(yc):,} rows")

    if os.path.exists(cb_path):
        cb = pd.read_csv(cb_path)
        df = yc.merge(cb, on="company_name", how="left")
        print(f"  Merged Crunchbase data: {len(df):,} rows")
    else:
        print("  Crunchbase file not found — funding features will be NaN-filled.")
        df = yc.copy()

    return df


def engineer_funding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features 1–3: Funding signals."""
    if "total_funding_usd" not in df.columns:
        df["total_funding_usd"] = np.nan
    if "funding_round_count" not in df.columns:
        df["funding_round_count"] = 0
    if "seed_amount_usd" not in df.columns:
        df["seed_amount_usd"] = np.nan

    df["total_funding_usd"] = pd.to_numeric(df["total_funding_usd"], errors="coerce")
    df["total_funding_usd"].fillna(df["total_funding_usd"].median(), inplace=True)

    df["funding_round_count"] = pd.to_numeric(df["funding_round_count"], errors="coerce").fillna(0).astype(int)
    df["seed_round_size"] = pd.to_numeric(df["seed_amount_usd"], errors="coerce")
    df["seed_round_size"].fillna(df["seed_round_size"].median(), inplace=True)

    return df


def engineer_team_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Features 4–6: Team quality signals."""
    elite_unis = set(config["feature_engineering"]["elite_universities"])
    faang = set(config["feature_engineering"]["faang_companies"])

    # Team size
    if "founder_count" in df.columns:
        df["team_size"] = pd.to_numeric(df["founder_count"], errors="coerce").fillna(1).astype(int)
    else:
        df["team_size"] = 2  # YC median is 2 founders

    # FAANG experience flag
    if "founder_previous_companies" in df.columns:
        df["has_faang_experience"] = (
            df["founder_previous_companies"]
            .fillna("")
            .apply(lambda x: int(any(f in x for f in faang)))
        )
    else:
        df["has_faang_experience"] = 0

    # Elite education score (0–3 based on how many founders attended elite schools)
    if "founder_education" in df.columns:
        df["elite_education_score"] = (
            df["founder_education"]
            .fillna("")
            .apply(lambda x: min(3, sum(u in x for u in elite_unis)))
        )
    else:
        df["elite_education_score"] = 0

    return df


def engineer_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features 7–8: Technical depth from GitHub signals."""
    if "github_repo_count" not in df.columns:
        df["github_repo_count"] = 0
    if "github_commit_frequency" not in df.columns:
        df["commit_frequency"] = 0.0
    else:
        df["commit_frequency"] = pd.to_numeric(df["github_commit_frequency"], errors="coerce").fillna(0)

    df["github_repo_count"] = pd.to_numeric(df["github_repo_count"], errors="coerce").fillna(0).astype(int)

    return df


def engineer_batch_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Features 9–10, 14: YC batch context."""
    season_map = config["feature_engineering"]["batch_season_map"]

    if "batch" in df.columns:
        # Parse e.g. 'W21', 'S23'
        df["batch_year"] = df["batch"].str[1:].apply(
            lambda x: int("20" + x) if len(x) == 2 else np.nan
        )
        df["yc_batch_encoded"] = df["batch"].str[0].map(season_map).fillna(0).astype(int)
    else:
        df["batch_year"] = 2020
        df["yc_batch_encoded"] = 0

    df["batch_year"] = pd.to_numeric(df["batch_year"], errors="coerce").fillna(2020).astype(int)

    if "batch_size" not in df.columns:
        # Approximate batch sizes per year
        batch_sizes = {2005: 8, 2010: 40, 2015: 100, 2019: 200, 2020: 200, 2021: 350, 2022: 400, 2023: 250, 2024: 200}
        df["batch_size"] = df["batch_year"].map(batch_sizes).fillna(200).astype(int)

    return df


def engineer_industry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features 11–12: Industry category and AI flag."""
    ai_keywords = ["AI", "ML", "machine learning", "artificial intelligence",
                   "LLM", "NLP", "computer vision", "deep learning", "generative"]

    cat_col = next((c for c in ["category", "vertical", "industry"] if c in df.columns), None)
    if cat_col:
        le = LabelEncoder()
        df["industry_category"] = le.fit_transform(df[cat_col].fillna("Other"))
        df["is_ai_ml"] = df[cat_col].fillna("").apply(
            lambda x: int(any(kw.lower() in x.lower() for kw in ai_keywords))
        )
    else:
        df["industry_category"] = 0
        df["is_ai_ml"] = 0

    return df


def engineer_geo_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Feature 13: Geography cluster."""
    geo_map = config["feature_engineering"]["geo_clusters"]
    sf_cities = set(geo_map["SF_Bay_Area"])
    nyc_cities = set(geo_map["NYC"])

    loc_col = next((c for c in ["city", "location", "hq_location"] if c in df.columns), None)

    def assign_cluster(loc):
        if pd.isna(loc) or loc == "":
            return 2  # Other US
        if any(c in loc for c in sf_cities):
            return 0  # SF Bay Area
        if any(c in loc for c in nyc_cities):
            return 1  # NYC
        us_indicators = ["CA", "NY", "TX", "WA", "MA", "IL", "USA", "US"]
        if any(ind in loc for ind in us_indicators):
            return 2  # Other US
        return 3  # International

    if loc_col:
        df["geo_cluster"] = df[loc_col].apply(assign_cluster)
    else:
        df["geo_cluster"] = 2

    return df


def build_success_label(df: pd.DataFrame) -> pd.DataFrame:
    """Build binary success label from status/outcome columns."""
    success_keywords = ["acquired", "ipo", "public", "series b", "series c",
                        "series d", "unicorn", "active"]
    fail_keywords = ["dead", "inactive", "failed", "closed", "defunct"]

    status_col = next((c for c in ["status", "outcome", "company_status"] if c in df.columns), None)

    if status_col:
        def label(s):
            if pd.isna(s):
                return np.nan
            sl = s.lower()
            if any(k in sl for k in success_keywords):
                return 1
            if any(k in sl for k in fail_keywords):
                return 0
            return np.nan

        df["success_label"] = df[status_col].apply(label)
        df.dropna(subset=["success_label"], inplace=True)
        df["success_label"] = df["success_label"].astype(int)
        print(f"  Success rate: {df['success_label'].mean():.1%} ({df['success_label'].sum():,} / {len(df):,})")
    else:
        raise ValueError("No status column found in dataset. Check data/README.md for column names.")

    return df


YIFE_FEATURES = [
    "total_funding_usd", "funding_round_count", "seed_round_size",
    "team_size", "has_faang_experience", "elite_education_score",
    "github_repo_count", "commit_frequency",
    "batch_year", "batch_size", "industry_category", "is_ai_ml",
    "geo_cluster", "yc_batch_encoded",
]


def run(config_path: str = "configs/model_config.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    set_seed(cfg["global"]["seed"])
    print_banner("YIFE Feature Engineering")

    df = load_raw_data(cfg)
    df = engineer_funding_features(df)
    df = engineer_team_features(df, cfg)
    df = engineer_technical_features(df)
    df = engineer_batch_features(df, cfg)
    df = engineer_industry_features(df)
    df = engineer_geo_features(df, cfg)
    df = build_success_label(df)

    output_cols = YIFE_FEATURES + ["success_label"]
    output = df[output_cols].copy()

    ensure_dir("data/processed")
    out_path = "data/processed/yife_features.csv"
    output.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved {len(output):,} samples → {out_path}")
    print(f"  Features: {YIFE_FEATURES}")
    return output


if __name__ == "__main__":
    run()
