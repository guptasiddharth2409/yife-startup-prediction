"""
01_generate_synthetic_data.py

Generates a synthetic YIFE dataset (n=4,323) matching the paper's schema,
feature distributions, temporal drift, and feature-target correlations.

This allows full pipeline reproducibility without proprietary Crunchbase/Kaggle keys.
Outputs: data/processed/yife_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N = 4323
rng = np.random.default_rng(42)

# --- Companies & Batches (W05-S24) ---
company_ids = [f"YC-{i:04d}" for i in range(1, N + 1)]
years = rng.choice(np.arange(2005, 2025), size=N)
batches = [f"{'W' if rng.random() < 0.5 else 'S'}{str(y)[-2:]}" for y in years]
batch_year_encoded = np.array(
    [int("20" + b[1:]) if int(b[1:]) <= 30 else int("19" + b[1:]) for b in batches]
)
batch_size = np.where(batch_year_encoded > 2019,
    rng.integers(250, 400, N), rng.integers(50, 200, N))

# --- Funding Features ---
num_funding_rounds = rng.choice(np.arange(0, 9), size=N,
    p=[0.20, 0.25, 0.22, 0.15, 0.08, 0.05, 0.03, 0.01, 0.01])
total_funding_usd = np.exp(rng.normal(13.5, 1.5, N)) * (1 + num_funding_rounds * 0.35)
seed_round_size = total_funding_usd * rng.uniform(0.05, 0.35, N)
seed_round_size = np.where(num_funding_rounds == 0, 0.0, seed_round_size)

# --- Team Features ---
team_size = rng.choice([1, 2, 3, 4, 5], size=N, p=[0.15, 0.43, 0.22, 0.12, 0.08])
faang_experience = rng.binomial(1, 0.18, N)  # ~18% have FAANG
elite_edu = rng.binomial(1, 0.22, N)          # ~22% elite university

# --- GitHub Features (71% coverage) ---
github_mask = rng.random(N) < 0.71
github_repo_count = np.where(github_mask, rng.poisson(18, N), np.nan)
github_commit_freq = np.where(
    github_mask,
    np.abs(rng.normal(12.5, 6.0, N)),
    np.nan
)

# --- Industry & AI ---
industry_choices = ['B2B', 'AI', 'FinTech', 'Consumer', 'Healthcare', 'DevTools', 'Other']
industry_probs = [0.30, 0.18, 0.08, 0.12, 0.06, 0.10, 0.16]
industry_category = rng.choice(industry_choices, N, p=industry_probs)
ai_flag = np.where(
    (industry_category == 'AI') | ((batch_year_encoded >= 2020) & (rng.random(N) < 0.35)),
    1, 0
)

# --- Geography ---
geo_choices = ['SF Bay', 'NY', 'International', 'Other']
geo_cluster = rng.choice(geo_choices, N, p=[0.45, 0.15, 0.25, 0.15])

# --- Tier-1 VC ---
tier1_vc_investor = rng.binomial(1, 0.12, N)

# --- Success Label (logistic function matching paper's SHAP ranking) ---
def normalize(x):
    mn, mx = np.nanmin(x), np.nanmax(x)
    return (x - mn) / (mx - mn + 1e-9)

github_repo_fill = np.where(np.isnan(github_repo_count), np.nanmedian(github_repo_count), github_repo_count)
github_commit_fill = np.where(np.isnan(github_commit_freq), np.nanmedian(github_commit_freq), github_commit_freq)

logit = (
    1.80 * normalize(num_funding_rounds.astype(float))  # rank 1: funding rounds
  + 1.20 * normalize(batch_year_encoded.astype(float))  # rank 2: batch year
  + 0.90 * normalize(team_size.astype(float))           # rank 3: team size
  + 0.80 * normalize(np.log1p(total_funding_usd))       # rank 4: total funding
  + 0.60 * ai_flag.astype(float)                        # rank 5: ai flag
  + 0.45 * normalize(industry_category == 'B2B')        # rank 6: B2B category
  + 0.35 * normalize(github_commit_fill)                # rank 7: github commits
  + 0.25 * normalize(geo_cluster == 'SF Bay')           # rank 8: SF Bay
  + 0.10 * elite_edu.astype(float)                      # rank 9: elite edu (weak)
  + 0.05 * faang_experience.astype(float)               # rank 10: FAANG (near-zero)
  + rng.normal(0, 0.3, N)                               # noise
)

# Calibrate to ~45% success rate
threshold = np.quantile(logit, 0.55)
success = (logit >= threshold).astype(int)
assert 0.43 <= success.mean() <= 0.47, f"Success rate out of bounds: {success.mean():.3f}"
print(f"Success rate: {success.mean():.3f} (target: ~0.45)")

# --- Assemble DataFrame ---
df = pd.DataFrame({
    'company': company_ids,
    'batch': batches,
    'batch_year_encoded': batch_year_encoded,
    'batch_size': batch_size,
    'total_funding_usd': total_funding_usd.round(2),
    'num_funding_rounds': num_funding_rounds,
    'seed_round_size': seed_round_size.round(2),
    'team_size': team_size,
    'faang_experience': faang_experience,
    'elite_edu': elite_edu,
    'github_repo_count': github_repo_count,
    'github_commit_freq': np.where(np.isnan(github_commit_freq),
                                    np.nan, github_commit_freq.round(2)),
    'industry_category': industry_category,
    'ai_flag': ai_flag,
    'geo_cluster': geo_cluster,
    'tier1_vc_investor': tier1_vc_investor,
    'success': success,
})

out_path = OUTPUT_DIR / 'yife_features.csv'
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df.describe())
