"""
src/data/collect_crunchbase.py

Load Crunchbase funding data for YC companies.
Requires a manual export (CSV) placed at data/raw/crunchbase_yc.csv

Free-tier academic access: https://data.crunchbase.com/docs
Alternative: Crunchbase Open Data Map (subset)

Expected columns: company, total_funding_usd, num_funding_rounds,
                  seed_round_size, tier1_vc_investor, lead_investor
"""

import pandas as pd
from pathlib import Path
from src.config import RAW_DIR


def load_crunchbase(path=None):
    path = path or RAW_DIR / 'crunchbase_yc.csv'
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Crunchbase CSV not found at {path}.\n"
            "Please export from Crunchbase and place at data/raw/crunchbase_yc.csv"
        )
    df = pd.read_csv(path)
    print(f"Loaded Crunchbase data: {df.shape}")
    return df


if __name__ == '__main__':
    df = load_crunchbase()
    print(df.describe())
