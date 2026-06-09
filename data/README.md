# Data Sources

Raw data is **not included** in this repository due to licensing restrictions.
Follow the steps below to reproduce the dataset used in the paper.

---

## Required Files

Place downloaded files at these exact paths:

```
data/
└── raw/
    ├── yc_companies.csv        ← from Kaggle (primary)
    └── crunchbase_export.csv   ← from Crunchbase ODM (optional, enriches funding features)
```

---

## Step 1 — YC Companies Dataset (Primary)

**Source:** Kaggle  
**Search:** `Y Combinator companies` → download the most recent CSV  
**URL:** https://www.kaggle.com/datasets (search "Y Combinator companies")  
**Size:** ~4,500 rows, ~20 columns  
**License:** Public / CC0

Expected columns used by `feature_engineering.py`:

| Column | Description |
|--------|-------------|
| `company_name` | Company identifier (used for merging) |
| `batch` | YC cohort, e.g. `W21`, `S23` |
| `status` | Current company status (success label source) |
| `category` | Primary industry vertical |
| `city` | HQ city |
| `founder_count` | Number of co-founders |
| `founder_education` | Comma-separated university names |
| `founder_previous_companies` | Prior employer names |
| `github_repo_count` | Public repo count at batch entry |

---

## Step 2 — Crunchbase Export (Optional)

**Source:** Crunchbase Open Data Map  
**URL:** https://data.crunchbase.com  
**Access:** Free academic tier or Open Data Map CSV download  
**License:** Crunchbase ODM license

Expected columns:

| Column | Description |
|--------|-------------|
| `company_name` | Join key |
| `total_funding_usd` | Total capital raised |
| `funding_round_count` | Number of rounds |
| `seed_amount_usd` | Seed round size |

> If this file is absent, funding features default to median-imputed values.

---

## Step 3 — GitHub API (Optional)

Commit frequency data was collected via the GitHub REST API:

```bash
curl https://api.github.com/repos/{owner}/{repo}/stats/commit_activity
```

This is optional — `commit_frequency` defaults to 0 if absent.

---

## Success Label Definition

The binary `success_label` is constructed from the `status` column:

| Status value | Label |
|---|---|
| acquired, ipo, public, series b+, unicorn, active | **1 (Success)** |
| dead, inactive, failed, closed, defunct | **0 (Failure)** |
| Ambiguous / unknown | Dropped |

Final dataset: **4,323 labeled companies** after cleaning.
