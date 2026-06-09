# YIFE: Predicting Early-Stage Startup Success Using ML

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-Champion%20F1%3D0.85-orange)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)](https://shap.readthedocs.io)
[![Status](https://img.shields.io/badge/Status-Under%20Review-yellow)]()

> **Paper:** *Predicting Early-Stage Startup Success Using ML: A YC-Inspired Feature Engineering Approach*  
> **Author:** Siddharth Gupta · Greater Noida Institute of Technology · [guptasiddharth0767@gmail.com](mailto:guptasiddharth0767@gmail.com)  
> **Status:** Under peer review

---

## Overview

This repository contains the complete, reproducible pipeline for the YIFE (**YC-Inspired Feature Engineering**) framework — a machine learning system for predicting early-stage startup success using domain-specific signals drawn from Y Combinator's portfolio.

Most existing ML approaches use generic Crunchbase/PitchBook features and miss critical accelerator-specific signals. **YIFE** incorporates:
- Batch cohort timing (encodes market cycle conditions)
- Founder technical depth (proxied via GitHub activity)
- Team composition dynamics
- Industry category & AI integration flags
- Geographic clustering

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score | AUROC |
|---|---|---|---|---|---|
| Logistic Regression (B1 Baseline) | 0.71 | 0.68 | 0.64 | 0.66 | 0.74 |
| Logistic Regression (YIFE) | 0.75 | 0.73 | 0.70 | 0.71 | 0.79 |
| Random Forest (B2 Baseline) | 0.80 | 0.79 | 0.76 | 0.77 | 0.85 |
| Random Forest (YIFE) | 0.83 | 0.82 | 0.79 | 0.80 | 0.88 |
| **XGBoost (YIFE) ★** | **0.86** | **0.85** | **0.85** | **0.85** | **0.91** |
| SVM (YIFE) | 0.79 | 0.78 | 0.74 | 0.76 | 0.83 |
| MLP Neural Network (YIFE) | 0.81 | 0.80 | 0.78 | 0.79 | 0.86 |

★ Best Model · Evaluated on held-out temporal test set (W21–S24 batches, n=863)

---

## YIFE Feature Set (14 Features)

| Category | Feature | Description |
|---|---|---|
| Funding | `total_funding_usd` | Total capital raised |
| Funding | `num_funding_rounds` | Count of funding events |
| Funding | `seed_round_size` | Seed round amount |
| Team | `team_size` | Co-founder count |
| Team | `faang_experience` | Prior FAANG employment (binary) |
| Team | `elite_edu` | Top-20 university attendance (binary) |
| Tech Depth | `github_repo_count` | Total public repos of founding team |
| Tech Depth | `github_commit_freq` | Avg weekly commits (6 mo pre-YC) |
| Batch Context | `batch_year_encoded` | Batch year (encodes market cycles) |
| Batch Context | `batch_size` | Companies in same cohort |
| Industry | `industry_category` | YC-assigned category (B2B, AI, etc.) |
| Industry | `ai_flag` | AI as core product component (binary) |
| Geography | `geo_cluster` | Location cluster (SF Bay, NY, etc.) |
| Network | `tier1_vc_investor` | Backed by Tier-1 VC post-YC (binary) |

---

## Top SHAP Findings

| Rank | Feature | Mean \|SHAP\| | Effect |
|---|---|---|---|
| 1 | `num_funding_rounds` | 0.187 | Higher rounds → higher success |
| 2 | `batch_year_encoded` | 0.163 | Recent batches (AI era 2020+) → positive |
| 3 | `team_size` | 0.141 | 2–4 co-founders = optimal |
| 4 | `total_funding_usd` | 0.128 | Higher funding → positive (diminishing) |
| 5 | `ai_flag` | 0.112 | AI-core products strongly positive post-2020 |
| 10 | `faang_experience` | 0.031 | **Near-zero effect** — challenges VC heuristic |

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/guptasiddharth2409/yife-startup-prediction.git
cd yife-startup-prediction
pip install -r requirements.txt

# 2. Generate synthetic dataset (matches paper distributions)
python scripts/01_generate_synthetic_data.py

# 3. Train all 5 models
python scripts/02_train_model.py

# 4. Generate figures (ROC curves + SHAP)
python scripts/03_generate_figures.py

# 5. (Optional) Run Streamlit prediction UI
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
yife-startup-prediction/
├── scripts/
│   ├── 01_generate_synthetic_data.py   # Synthetic dataset matching paper distributions
│   ├── 02_train_model.py               # Train all 5 classifiers + save models
│   └── 03_generate_figures.py          # ROC curves + SHAP importance plots
├── src/
│   ├── config.py                       # Paths & constants
│   ├── data/
│   │   ├── collect_kaggle.py           # Load YC company directory
│   │   ├── collect_crunchbase.py       # Load Crunchbase funding data
│   │   └── collect_github.py          # Fetch founder GitHub activity
│   ├── features/
│   │   └── build_features.py          # YIFE 14-feature engineering pipeline
│   ├── models/
│   │   └── trainer.py                 # Model training & evaluation
│   └── visualization/
│       ├── roc_curves.py              # Figure 1: ROC curves
│       └── shap_analysis.py           # Figure 2: SHAP global importance
├── app/
│   └── streamlit_app.py               # Interactive prediction dashboard
├── notebooks/
│   └── YIFE_full_pipeline.ipynb       # End-to-end Jupyter notebook
├── data/
│   ├── raw/                           # Place Kaggle/Crunchbase CSVs here
│   └── processed/                     # Auto-generated processed features
├── models/                            # Saved model artifacts (.pkl)
├── figures/                           # Generated plots (ROC, SHAP)
├── paper/                             # Paper PDF and supplementary materials
├── requirements.txt
└── README.md
```

---

## Dataset

- **4,323 YC-funded companies** spanning Winter 2005 – Summer 2024
- **Success label**: raised Series A+, acquired, or remained active 3+ years post-batch
- **~45% success rate** (consistent with published YC Series A statistics)
- **Temporal split**: Training = W05–S20 (~3,460 companies), Test = W21–S24 (~863 companies)

> **Note on data:** Due to Crunchbase API terms of service, this repository ships with a high-fidelity synthetic dataset generated by `scripts/01_generate_synthetic_data.py`. The generator preserves the paper's feature distributions, temporal drift, and feature-target correlations. Real data can be substituted by placing `yc_companies.csv` and `crunchbase_yc.csv` into `data/raw/` and running `src/features/build_features.py`.

---

## Reproducibility

All experiments use `random_state=42`. The temporal train/test split prevents any look-ahead bias. Five-fold stratified cross-validation is performed on the training set only. McNemar's test confirms XGBoost significantly outperforms all baselines (vs. MLP: χ²=33.28, p<0.0001).

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{gupta2025yife,
  title   = {Predicting Early-Stage Startup Success Using ML: A YC-Inspired Feature Engineering Approach},
  author  = {Gupta, Siddharth},
  journal = {Under Review},
  year    = {2025},
  url     = {https://github.com/guptasiddharth2409/yife-startup-prediction}
}
```

---

## Key References

1. Li et al. (2025). *Founder Backgrounds and Startup Funding: Evidence from Y Combinator.* arXiv:2512.13755.
2. Razaghzadeh Bidgoli et al. (2024). *Predicting the success of startups using a ML approach.* J. Innovation & Entrepreneurship.
3. Park et al. (2024). *Predicting startup success using two bias-free machine learning.* J. Big Data, 11, 122.
4. Maarouf et al. (2025). *A fused large language model for predicting startup success.* EJOR, 322(1).
5. Lundberg & Lee (2017). *A unified approach to interpreting model predictions.* NeurIPS 30.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
