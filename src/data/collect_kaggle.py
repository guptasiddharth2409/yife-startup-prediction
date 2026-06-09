"""
src/data/collect_kaggle.py

Download or load the Kaggle YC company directory.
If you have kaggle CLI configured, uncomment the API call below.
Otherwise, download manually and place at data/raw/yc_companies.csv

Kaggle dataset: https://www.kaggle.com/datasets/search?q=y+combinator+companies
"""

import pandas as pd
from pathlib import Path
from src.config import RAW_DIR


def load_yc_companies(path=None):
    path = path or RAW_DIR / 'yc_companies.csv'
    path = Path(path)

    # Uncomment if kaggle CLI is configured:
    # import subprocess
    # subprocess.run(['kaggle', 'datasets', 'download', '-d',
    #                 'dylanknasir/y-combinator-companies', '--unzip',
    #                 '-p', str(RAW_DIR)], check=True)

    if not path.exists():
        raise FileNotFoundError(
            f"YC company CSV not found at {path}.\n"
            "Please download from Kaggle and place at data/raw/yc_companies.csv"
        )
    df = pd.read_csv(path)
    print(f"Loaded YC companies: {df.shape}")
    return df


if __name__ == '__main__':
    df = load_yc_companies()
    print(df.head())
    print(df.columns.tolist())
