"""
src/features/build_features.py

YIFE Feature Engineering Pipeline.
Merges YC company directory + Crunchbase + GitHub data, constructs the
14 YIFE features, applies batch-cohort median imputation, one-hot encoding,
and z-score standardization.

Outputs: data/processed/yife_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.config import RAW_DIR, PROCESSED_DIR, YIFE_FEATURES


def batch_to_year(b: str) -> int:
    """Convert batch code (e.g. 'W20', 'S24') to full year integer."""
    try:
        yr = int(str(b).strip()[1:])  # drop W/S prefix
        return 2000 + yr if yr <= 30 else 1900 + yr
    except (ValueError, IndexError):
        return np.nan


def load_raw():
    yc_path = RAW_DIR / 'yc_companies.csv'
    cb_path = RAW_DIR / 'crunchbase_yc.csv'
    gh_path = RAW_DIR / 'founders_github.csv'

    if not yc_path.exists():
        raise FileNotFoundError(
            f"Place YC company directory CSV at: {yc_path}\n"
            "Download from: https://www.kaggle.com/datasets/"
            "(search: Y Combinator companies)"
        )
    yc = pd.read_csv(yc_path)
    cb = pd.read_csv(cb_path) if cb_path.exists() else pd.DataFrame()
    gh = pd.read_csv(gh_path) if gh_path.exists() else pd.DataFrame()
    return yc, cb, gh


def merge_sources(yc, cb, gh):
    """Merge YC, Crunchbase, and GitHub data on company identifier."""
    df = yc.copy()
    if not cb.empty:
        merge_col = 'company' if 'company' in cb.columns else cb.columns[0]
        df = df.merge(cb, on=merge_col, how='left', suffixes=('', '_cb'))
    if not gh.empty:
        gh_agg = gh.groupby('company').agg(
            github_repo_count=('public_repos', 'sum'),
            github_commit_freq=('weekly_commits', 'mean')
        ).reset_index()
        df = df.merge(gh_agg, on='company', how='left')
    return df


def engineer_features(df):
    """Construct all 14 YIFE features."""
    # --- Funding features ---
    df['total_funding_usd']   = pd.to_numeric(df.get('total_funding_usd',
        df.get('total_raised', np.nan)), errors='coerce')
    df['num_funding_rounds']  = pd.to_numeric(df.get('num_funding_rounds',
        df.get('funding_rounds', 0)), errors='coerce').fillna(0).astype(int)
    df['seed_round_size']     = pd.to_numeric(df.get('seed_round_size', np.nan), errors='coerce')

    # --- Team features ---
    df['team_size']           = pd.to_numeric(df.get('team_size', 1), errors='coerce').fillna(1).astype(int)
    df['faang_experience']    = pd.to_numeric(df.get('faang_experience', 0), errors='coerce').fillna(0).astype(int)
    df['elite_edu']           = pd.to_numeric(df.get('elite_edu', 0), errors='coerce').fillna(0).astype(int)

    # --- GitHub (already merged or NaN) ---
    if 'github_repo_count' not in df.columns:
        df['github_repo_count'] = np.nan
    if 'github_commit_freq' not in df.columns:
        df['github_commit_freq'] = np.nan

    # --- Batch context ---
    df['batch_year_encoded'] = df['batch'].apply(batch_to_year).astype(float)
    df['batch_size'] = pd.to_numeric(df.get('batch_size', np.nan), errors='coerce')

    # --- Industry & AI ---
    df['industry_category'] = df.get('category', df.get('industry', 'Other')).fillna('Other')
    df['ai_flag'] = (df['industry_category'].str.lower().str.contains('ai|machine learning')
                     ).astype(int)
    if 'ai_flag' in df.columns and df['ai_flag'].dtype != int:
        df['ai_flag'] = pd.to_numeric(df['ai_flag'], errors='coerce').fillna(0).astype(int)

    # --- Geography ---
    def map_geo(loc):
        if pd.isna(loc):
            return 'Other'
        loc = str(loc).lower()
        if any(x in loc for x in ['san francisco', 'bay area', 'mountain view', 'palo alto', 'sf', 'sunnyvale']):
            return 'SF Bay'
        if any(x in loc for x in ['new york', 'nyc', 'brooklyn']):
            return 'NY'
        if 'united states' in loc or 'usa' in loc:
            return 'Other'
        return 'International'
    df['geo_cluster'] = df.get('location', pd.Series(['Other'] * len(df))).apply(map_geo)

    # --- Network ---
    df['tier1_vc_investor'] = pd.to_numeric(df.get('tier1_vc_investor', 0), errors='coerce').fillna(0).astype(int)

    # --- Success label ---
    if 'success' not in df.columns:
        df['success'] = (
            (df['num_funding_rounds'] >= 1) |
            (df['total_funding_usd'] > 1_000_000)
        ).astype(int)

    return df


def impute_and_scale(df):
    """Batch-cohort median imputation + z-score standardization."""
    continuous = ['total_funding_usd', 'seed_round_size', 'github_repo_count',
                  'github_commit_freq', 'batch_size', 'total_funding_usd']
    for col in continuous:
        if col in df.columns:
            df[col] = df.groupby('batch_year_encoded')[col].transform(
                lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())

    scale_cols = ['total_funding_usd', 'seed_round_size',
                  'github_repo_count', 'github_commit_freq', 'batch_year_encoded']
    scale_cols = [c for c in scale_cols if c in df.columns]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols].fillna(0))
    return df


def build_features():
    yc, cb, gh = load_raw()
    df = merge_sources(yc, cb, gh)
    df = engineer_features(df)
    df = impute_and_scale(df)

    keep = ['company', 'batch'] + YIFE_FEATURES + ['success']
    keep = [c for c in keep if c in df.columns]
    out = df[keep]
    out.to_csv(PROCESSED_DIR / 'yife_features.csv', index=False)
    print(f"Features built: {out.shape} -> {PROCESSED_DIR / 'yife_features.csv'}")
    print(f"Success rate: {out['success'].mean():.3f}")
    return out


if __name__ == '__main__':
    build_features()
