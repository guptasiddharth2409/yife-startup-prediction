"""Build the raw YIFE feature table from research input sources.

This stage performs source integration and cohort-scoped missing-value handling only.
Model-specific imputation, one-hot encoding, and scaling are fitted inside the
training pipeline on training data only to prevent temporal information leakage.

The published YIFE framework contains 14 features. The original research target
label must be supplied by the research data; this script intentionally does not
invent a replacement success definition.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR


CONT_FEATS = [
    "total_funding_usd", "num_funding_rounds", "seed_round_size",
    "team_size", "github_repo_count", "github_commit_freq", "batch_size",
]


def batch_to_year(batch_str):
    try:
        s = str(batch_str).strip()
        if len(s) >= 3:
            yy = int(s[1:])
            return 2000 + yy if yy <= 30 else 1900 + yy
    except (ValueError, TypeError):
        pass
    return np.nan


def build_features():
    yc_path = RAW_DIR / "yc_companies.parquet"
    cb_path = RAW_DIR / "crunchbase_yc.parquet"
    founder_path = RAW_DIR / "founders_github.parquet"

    if not yc_path.exists() or not cb_path.exists():
        raise FileNotFoundError(
            "Research input files not found. Provide yc_companies.parquet and "
            "crunchbase_yc.parquet in data/raw/. See data/README.md."
        )

    yc = pd.read_parquet(yc_path)
    cb = pd.read_parquet(cb_path)
    founders = pd.read_parquet(founder_path) if founder_path.exists() else pd.DataFrame()

    df = yc.merge(cb, how="left", on="company", suffixes=("", "_cb"))

    df["total_funding_usd"] = df["total_funding_usd"].fillna(df.get("total_raised", np.nan))
    df["num_funding_rounds"] = (
        df["num_funding_rounds"].fillna(df.get("rounds", np.nan)).fillna(0).astype(int)
    )
    df["seed_round_size"] = df.get("seed_round_size", np.nan)
    df["team_size"] = df.get("team_size", pd.Series(1, index=df.index)).fillna(1).astype(int)
    df["faang_experience"] = df.get("faang_experience", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["elite_edu"] = df.get("elite_edu", pd.Series(0, index=df.index)).fillna(0).astype(int)

    if not founders.empty and {"company", "public_repos"}.issubset(founders.columns):
        g = founders.groupby("company")["public_repos"].sum().rename("github_repo_count")
        df = df.drop(columns=["github_repo_count"], errors="ignore").merge(g, how="left", on="company")
    else:
        df["github_repo_count"] = np.nan

    df["github_commit_freq"] = df.get("github_commit_freq", pd.Series(np.nan, index=df.index))
    df["batch_year_encoded"] = df["batch"].apply(batch_to_year)
    df["batch_size"] = df.get("batch_size", pd.Series(np.nan, index=df.index))
    df["industry_category"] = df.get("category", pd.Series("Other", index=df.index)).fillna("Other")
    df["ai_flag"] = df.get("ai_flag", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["geo_cluster"] = df.get("location", pd.Series("Other", index=df.index)).fillna("Other")
    df["tier1_vc_investor"] = df.get("tier1_vc_investor", pd.Series(0, index=df.index)).fillna(0).astype(int)

    if "success" not in df.columns:
        raise ValueError(
            "The research 'success' target is missing. Refusing to synthesize a "
            "different target definition; see data/README.md for the published label."
        )

    required = [
        "company", "batch", "success", "total_funding_usd", "num_funding_rounds",
        "seed_round_size", "team_size", "faang_experience", "elite_edu",
        "github_repo_count", "github_commit_freq", "batch_year_encoded", "batch_size",
        "industry_category", "ai_flag", "geo_cluster", "tier1_vc_investor",
    ]
    out = df[required].copy()

    # Keep missing values here. The trainer fits all imputers using training rows only.
    out_path = PROCESSED_DIR / "yife_features.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Saved raw YIFE feature table -> {out_path} | shape={out.shape}")
    return out


if __name__ == "__main__":
    build_features()
