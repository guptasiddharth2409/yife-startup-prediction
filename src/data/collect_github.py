"""
src/data/collect_github.py

Fetch GitHub activity metrics for YC founders.
Requires:
  - data/raw/founders.csv with columns: company, founder_name, github_username
  - GITHUB_TOKEN in .env (for higher rate limits; optional for public endpoints)

Coverage: ~71% of YC founders have public GitHub profiles.
Remaining 29% are imputed using batch-cohort medians in build_features.py.
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from src.config import RAW_DIR

load_dotenv()
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN else {}
BASE_URL = 'https://api.github.com'


def get_user_repos(username: str) -> int:
    """Return total public repo count for a GitHub user."""
    url = f'{BASE_URL}/users/{username}'
    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.status_code == 200:
        return resp.json().get('public_repos', 0)
    return None


def get_weekly_commits(username: str, weeks: int = 26) -> float:
    """
    Approximate weekly commit frequency from the user's first repo.
    Returns average weekly commits over the last `weeks` weeks.
    """
    repos_url = f'{BASE_URL}/users/{username}/repos?sort=updated&per_page=5'
    resp = requests.get(repos_url, headers=HEADERS, timeout=10)
    if resp.status_code != 200 or not resp.json():
        return None
    first_repo = resp.json()[0]['full_name']
    stats_url = f'{BASE_URL}/repos/{first_repo}/stats/participation'
    resp2 = requests.get(stats_url, headers=HEADERS, timeout=10)
    if resp2.status_code == 200:
        all_commits = resp2.json().get('all', [])
        recent = all_commits[-weeks:]
        return sum(recent) / len(recent) if recent else 0.0
    return None


def collect_github_stats():
    founders_path = RAW_DIR / 'founders.csv'
    if not founders_path.exists():
        raise FileNotFoundError(
            f"Place founders.csv at {founders_path}\n"
            "Columns: company, founder_name, github_username"
        )
    founders = pd.read_csv(founders_path)
    records = []
    for _, row in founders.iterrows():
        username = row.get('github_username')
        if pd.isna(username) or not str(username).strip():
            records.append({**row.to_dict(), 'public_repos': None, 'weekly_commits': None})
            continue
        repos = get_user_repos(str(username).strip())
        commits = get_weekly_commits(str(username).strip())
        records.append({**row.to_dict(), 'public_repos': repos, 'weekly_commits': commits})
        time.sleep(0.5)  # respect API rate limits

    out_df = pd.DataFrame(records)
    out_path = RAW_DIR / 'founders_github.csv'
    out_df.to_csv(out_path, index=False)
    print(f"GitHub stats saved to {out_path}")
    coverage = out_df['public_repos'].notna().mean()
    print(f"GitHub coverage: {coverage:.1%}")
    return out_df


if __name__ == '__main__':
    collect_github_stats()
