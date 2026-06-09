"""YIFE Feature Engineering Pipeline

Loads raw tables and produces data/processed/yife_features.parquet.

Preprocessing steps:
  1. Merge Kaggle YC + Crunchbase + GitHub founder data
  2. Extract 14 YIFE features across 6 categories (Table 2 in paper)
  3. Median imputation within batch cohorts
  4. One-hot encoding for categorical features
  5. Z-score standardization for continuous features

If raw data is unavailable, use src/data/generate_synthetic.py.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import RAW_DIR, PROCESSED_DIR

CONT_FEATS = [
    "total_funding_usd", "num_funding_rounds", "seed_round_size",
    "team_size", "github_repo_count", "github_commit_freq", "batch_size",
]
CONT_TO_SCALE = [
    "total_funding_usd", "seed_round_size",
    "github_repo_count", "github_commit_freq", "batch_year_encoded",
]


def batch_to_year(batch_str):
    try:
        if isinstance(batch_str, str) and len(batch_str) >= 3:
            yy = int(batch_str[1:])
            return 2000 + yy if yy <= 30 else 1900 + yy
    except (ValueError, TypeError):
        pass
    return np.nan


def build_features():
    yc      = pd.read_parquet(RAW_DIR / "yc_companies.parquet")
    cb      = pd.read_parquet(RAW_DIR / "crunchbase_yc.parquet")
    fpath   = RAW_DIR / "founders_github.parquet"
    founders = pd.read_parquet(fpath) if fpath.exists() else pd.DataFrame()

    df = yc.merge(cb, how="left", on="company", suffixes=("", "_cb"))

    df["total_funding_usd"]  = df["total_funding_usd"].fillna(df.get("total_raised", np.nan))
    df["num_funding_rounds"] = (
        df["num_funding_rounds"].fillna(df.get("rounds", np.nan)).fillna(0).astype(int)
    )
    df["seed_round_size"]    = df.get("seed_round_size", np.nan)
    df["team_size"]          = df.get("team_size", 1).fillna(1).astype(int)
    df["faang_experience"]   = df.get("faang_experience", 0).fillna(0).astype(int)
    df["elite_edu"]          = df.get("elite_edu", 0).fillna(0).astype(int)

    if not founders.empty:
        g  = founders.groupby("company")["public_repos"].sum().rename("github_repo_count")
        df = df.merge(g, how="left", on="company")
    else:
        df["github_repo_count"] = np.nan
    df["github_commit_freq"]  = df.get("github_commit_freq", np.nan)
    df["batch_year_encoded"]  = df["batch"].apply(batch_to_year)
    df["batch_size"]          = df.get("batch_size", np.nan)
    df["industry_category"]   = df.get("category", "Other")
    df["ai_flag"]             = df.get("ai_flag", 0).fillna(0).astype(int)
    df["geo_cluster"]         = df.get("location", "Other")
    df["tier1_vc_investor"]   = df.get("tier1_vc_investor", 0).fillna(0).astype(int)

    if "success" not in df.columns:
        df["success"] = ((df["num_funding_rounds"] >= 1) | (df["total_funding_usd"] > 1e6)).astype(int)

    df["batch_year_encoded"] = df["batch_year_encoded"].fillna(df["batch_year_encoded"].median())
    for feat in CONT_FEATS:
        df[feat] = df.groupby("batch_year_encoded")[feat].transform(lambda x: x.fillna(x.median()))
        df[feat] = df[feat].fillna(df[feat].median())

    df = pd.get_dummies(df, columns=["industry_category", "geo_cluster"],
                        prefix=["industry", "geo"], dummy_na=False)

    scaler = StandardScaler()
    df[CONT_TO_SCALE] = scaler.fit_transform(df[CONT_TO_SCALE])

    base_cols   = ["total_funding_usd","num_funding_rounds","seed_round_size",
                   "team_size","faang_experience","elite_edu","github_repo_count",
                   "github_commit_freq","batch_year_encoded","batch_size",
                   "ai_flag","tier1_vc_investor"]
    onehot_cols = [c for c in df.columns if c.startswith(("industry_","geo_"))]
    out = df[["company"] + base_cols + onehot_cols + ["success", "batch"]]
    out_path = PROCESSED_DIR / "yife_features.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Saved processed features to {out_path} | shape={out.shape}")
    return out


if __name__ == "__main__":
    build_features()
