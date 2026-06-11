"""Synthetic YIFE Dataset Generator

Generates a synthetic dataset matching the paper's schema.

Outputs: data/processed/yife_features.parquet
  - n = 4,323 companies (W05-S24)
  - success rate ~45%
  - 14 YIFE features

Usage:
    python src/data/generate_synthetic.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from src.config import PROCESSED_DIR

N = 4323
SEED = 42


def generate(seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic YIFE dataset. Always deterministic given same seed."""
    # Fresh RNG every call — deterministic, no shared state
    rng = np.random.default_rng(seed)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    companies = [f"company_{i:04d}" for i in range(N)]

    years = rng.integers(2005, 2025, size=N)
    seasons = rng.choice(['W', 'S'], size=N)          # vectorised — no loop
    batches = [f"{s}{str(y)[-2:]}" for s, y in zip(seasons, years)]
    batch_year_encoded = years.astype(int)
    batch_size = np.where(
        batch_year_encoded > 2019,
        rng.integers(250, 400, size=N),
        rng.integers(50, 200, size=N)
    )

    num_funding_rounds = np.clip(rng.poisson(1.2, size=N), 0, 8)
    total_funding_usd  = rng.lognormal(12.0, 1.2, size=N) * (1 + num_funding_rounds * 0.4)
    seed_round_size    = np.where(num_funding_rounds > 0,
                                  rng.lognormal(6.5, 1.0, size=N), 0.0)

    team_size        = rng.choice([1, 2, 3, 4, 5], size=N, p=[0.15, 0.45, 0.20, 0.12, 0.08])
    faang_experience = rng.binomial(1, 0.18, size=N)   # scalar p — safe
    elite_edu        = rng.binomial(1, 0.22, size=N)   # scalar p — safe

    github_coverage    = rng.choice([1, 0], size=N, p=[0.71, 0.29])
    github_repo_count  = (rng.poisson(5, size=N) * github_coverage).astype(float)
    github_commit_freq = np.clip(
        github_repo_count * rng.uniform(0.2, 1.5, size=N)
        + rng.normal(0, 0.5, size=N),
        0, None
    )
    github_repo_count[github_repo_count == 0]   = np.nan
    github_commit_freq[github_commit_freq == 0] = np.nan

    industry_choices = ['B2B', 'AI', 'FinTech', 'Consumer',
                        'Healthcare', 'DevTools', 'Other']
    industry = rng.choice(industry_choices, size=N,
                          p=[0.30, 0.18, 0.08, 0.12, 0.06, 0.10, 0.16])

    # ai_prob is a float array — use rng.random() < ai_prob instead of binomial
    ai_prob = np.where((batch_year_encoded >= 2020) & (industry == 'AI'), 0.90,
              np.where(batch_year_encoded >= 2020, 0.40, 0.10))
    ai_flag = (rng.random(size=N) < ai_prob).astype(int)   # FIX: was rng.binomial(1, array)

    geo_cluster = rng.choice(['SF Bay', 'NY', 'International', 'Other'],
                             size=N, p=[0.45, 0.15, 0.25, 0.15])
    tier1_vc_investor = rng.binomial(1, 0.12, size=N)      # scalar p — safe

    # --- success label via logistic model ---
    batch_norm  = (batch_year_encoded - 2005) / (2024 - 2005)
    rounds_norm = num_funding_rounds / 8.0
    fund_norm   = np.log1p(total_funding_usd) / np.log1p(total_funding_usd.max())
    team_norm   = team_size / 5.0

    # Safe commit-freq normalisation — guard against all-NaN
    commit_max = np.nanmax(github_commit_freq) if not np.all(np.isnan(github_commit_freq)) else 1.0
    commit_norm = np.nan_to_num(github_commit_freq / commit_max, nan=0.0)

    logit = (
        1.87 * rounds_norm
      + 1.63 * batch_norm
      + 1.41 * team_norm
      + 1.28 * fund_norm
      + 1.12 * ai_flag
      + 0.94 * (industry == 'B2B').astype(float)
      + 0.87 * commit_norm
      + 0.71 * (geo_cluster == 'SF Bay').astype(float)
      + 0.52 * elite_edu
      + 0.31 * faang_experience
    )
    prob      = 1 / (1 + np.exp(-(logit - logit.mean())))
    threshold = np.quantile(prob, 1 - 0.45)
    success   = (prob >= threshold).astype(int)

    df = pd.DataFrame({
        "company":            companies,
        "batch":              batches,
        "batch_year_encoded": batch_year_encoded,
        "batch_size":         batch_size,
        "total_funding_usd":  total_funding_usd,
        "num_funding_rounds": num_funding_rounds,
        "seed_round_size":    seed_round_size,
        "team_size":          team_size,
        "faang_experience":   faang_experience,
        "elite_edu":          elite_edu,
        "github_repo_count":  github_repo_count,
        "github_commit_freq": github_commit_freq,
        "industry_category":  industry,
        "ai_flag":            ai_flag,
        "geo_cluster":        geo_cluster,
        "tier1_vc_investor":  tier1_vc_investor,
        "success":            success,
    })

    out_path = PROCESSED_DIR / "yife_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Synthetic dataset written  ->  {out_path}")
    print(f"  n={len(df):,}  |  success_rate={df['success'].mean():.3f}  "
          f"|  ai_flag_rate={df['ai_flag'].mean():.3f}")
    return df


if __name__ == "__main__":
    generate()
