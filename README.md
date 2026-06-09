<div align="center">

<img src="https://img.shields.io/badge/Status-Under%20Review-orange?style=for-the-badge" alt="Status"/>
<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/XGBoost-F1%3A0.85-brightgreen?style=for-the-badge" alt="XGBoost"/>
<img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License"/>
<img src="https://img.shields.io/badge/Conference-PTEMS--2026-red?style=for-the-badge" alt="Conference"/>

# 🚀 YIFE: YC-Inspired Feature Engineering
### for Early-Stage Startup Success Prediction

*A reproducible ML research framework trained on 4,323 Y Combinator companies (2005–2024)*

[📄 Paper](#citation) · [🔬 Methodology](#-yife-feature-framework) · [📊 Results](#-results) · [🛠 Setup](#-quick-start) · [📁 Structure](#-repository-structure)

</div>

---

## 📌 Overview

**YIFE** (YC-Inspired Feature Engineering) is a domain-specific ML framework for predicting whether an early-stage startup will achieve a successful outcome (IPO, acquisition, or active series-B+ growth). Unlike generic tabular ML benchmarks, YIFE constructs **14 structured features** grounded in Y Combinator's publicly known evaluation criteria — team quality, technical depth, funding momentum, and market context.

> **Paper:** *Predicting Early-Stage Startup Success Using ML: A YC-Inspired Feature Engineering Approach*  
> **Authors:** Siddharth Gupta · Pratham Namdev · Shubham Nagar · Sunny Kumar · Anjali Deshwal  
> **Institution:** Greater Noida Institute of Technology (GNIOT), GGSIPU, New Delhi  
> **Venue:** International Conference on Progressive Trends in Engineering, Management & Science (**PTEMS-2026**)

---

## 🧠 YIFE Feature Framework

All 14 features are engineered from publicly available signals:

| # | Feature | Category | Description |
|---|---------|----------|-------------|
| 1 | `total_funding_usd` | 💰 Funding | Total capital raised across all rounds |
| 2 | `funding_round_count` | 💰 Funding | Number of distinct funding events |
| 3 | `seed_round_size` | 💰 Funding | Size of the initial seed round |
| 4 | `team_size` | 👥 Team | Number of co-founders at YC application |
| 5 | `has_faang_experience` | 👥 Team | Any founder with FAANG/tier-1 tech background |
| 6 | `elite_education_score` | 👥 Team | Composite of founder alma mater rankings |
| 7 | `github_repo_count` | 💻 Technical | Public repositories at batch entry |
| 8 | `commit_frequency` | 💻 Technical | Avg commits/week in 6 months pre-batch |
| 9 | `batch_year` | 📅 Batch | YC cohort year (encoded) |
| 10 | `batch_size` | 📅 Batch | Number of companies in the same cohort |
| 11 | `industry_category` | 🏭 Industry | Primary vertical (label-encoded) |
| 12 | `is_ai_ml` | 🏭 Industry | Binary flag: AI/ML company |
| 13 | `geo_cluster` | 🌍 Geography | Founder location cluster (SF/NYC/Other/Intl) |
| 14 | `yc_batch_encoded` | 📅 Batch | Ordinal encoding of batch season (S/W) |

---

## 📊 Results

All models evaluated on an 80/20 stratified train-test split with 5-fold cross-validation.

| Model | Precision | Recall | **F1-Score** | **AUROC** | Training Time |
|-------|-----------|--------|-------------|----------|---------------|
| 🏆 **XGBoost + YIFE** | 0.87 | 0.83 | **0.85** | **0.91** | ~12s |
| Random Forest + YIFE | 0.82 | 0.78 | 0.80 | 0.88 | ~8s |
| MLP Neural Network | 0.80 | 0.78 | 0.79 | 0.86 | ~45s |
| SVM (RBF Kernel) | 0.77 | 0.75 | 0.76 | 0.83 | ~6s |
| Logistic Regression | 0.68 | 0.64 | 0.66 | 0.74 | ~1s |

> ★ XGBoost with YIFE features is the recommended model. Full confusion matrices and ROC curves are in [`figures/`](./figures/).

### Key Findings

- **FAANG experience** and **elite education score** are the top-2 predictors by SHAP value
- **Funding round count** outperforms raw funding amount as a signal of sustained investor confidence
- **AI/ML flag** adds +3 F1 points post-2019 cohorts due to sector tailwinds
- YIFE features outperform raw Crunchbase features by **+11 F1 points** (XGBoost ablation)

---

## 🛠 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/guptasiddharth2409/yife-startup-prediction.git
cd yife-startup-prediction
pip install -r requirements.txt
```

### 2. Prepare Data

Raw data is not included due to licensing. Download from the sources listed in [`data/README.md`](./data/README.md) and place files as:

```
data/
├── raw/
│   ├── yc_companies.csv        # from Kaggle
│   └── crunchbase_export.csv   # from Crunchbase ODM
```

### 3. Run the Pipeline

```bash
# Step 1: Engineer features
python src/feature_engineering.py

# Step 2: Train all models
python src/train_models.py

# Step 3: Evaluate and generate figures
python src/evaluate.py

# Step 4: SHAP explainability analysis
python src/shap_analysis.py
```

Or run the full pipeline in one command:

```bash
python src/run_pipeline.py
```

### 4. Jupyter Notebook

```bash
jupyter lab notebooks/
```

Open `01_eda.ipynb` → `02_feature_engineering.ipynb` → `03_model_training_evaluation.ipynb` in order.

---

## 📁 Repository Structure

```
yife-startup-prediction/
│
├── 📄 README.md                          ← You are here
├── 📄 LICENSE                            ← MIT
├── 📄 requirements.txt                   ← Pinned dependencies
├── 📄 setup.py                           ← Installable package
├── 📄 .gitignore
│
├── 📂 src/                               ← Core Python source
│   ├── feature_engineering.py            ← 14-feature YIFE construction
│   ├── train_models.py                   ← Train all 5 classifiers
│   ├── evaluate.py                       ← Metrics, ROC, confusion matrix
│   ├── shap_analysis.py                  ← SHAP bar + beeswarm plots
│   ├── run_pipeline.py                   ← End-to-end runner
│   └── utils.py                          ← Shared helpers
│
├── 📂 notebooks/
│   ├── 01_eda.ipynb                      ← Exploratory data analysis
│   ├── 02_feature_engineering.ipynb      ← YIFE feature walkthrough
│   └── 03_model_training_evaluation.ipynb← Training, evaluation, SHAP
│
├── 📂 data/
│   └── README.md                         ← Data sources + download guide
│
├── 📂 figures/                           ← Output plots (auto-generated)
│   └── README.md
│
├── 📂 configs/
│   └── model_config.yaml                 ← Hyperparameters (reproducible)
│
└── 📂 paper/
    └── README.md                         ← Paper link (post-acceptance)
```

---

## 🔬 Reproducibility

All experiments use a fixed random seed (`SEED=42`). Hyperparameters are stored in [`configs/model_config.yaml`](./configs/model_config.yaml) — no hardcoded values in source files.

```python
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
```

Expected runtime on a standard laptop (Intel i7, 16GB RAM): **< 5 minutes** end-to-end.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|----------|
| `scikit-learn` | ≥1.4 | ML models, preprocessing, metrics |
| `xgboost` | ≥2.0 | Gradient boosting classifier |
| `shap` | ≥0.44 | Feature importance explainability |
| `torch` | ≥2.1 | MLP Neural Network |
| `pandas` | ≥2.0 | Data manipulation |
| `numpy` | ≥1.26 | Numerical computing |
| `matplotlib` | ≥3.8 | Plotting |
| `seaborn` | ≥0.13 | Statistical visualizations |
| `pyyaml` | ≥6.0 | Config loading |
| `jupyter` | ≥7.0 | Notebook interface |

---

## 📝 Citation

If you use YIFE in your research, please cite:

```bibtex
@inproceedings{gupta2026yife,
  title     = {Predicting Early-Stage Startup Success Using Machine Learning:
               A YC-Inspired Feature Engineering Approach},
  author    = {Gupta, Siddharth and Namdev, Pratham and Nagar, Shubham
               and Kumar, Sunny and Deshwal, Anjali},
  booktitle = {Proceedings of the International Conference on Progressive
               Trends in Engineering, Management and Science (PTEMS-2026)},
  year      = {2026},
  institution = {Greater Noida Institute of Technology (GNIOT), GGSIPU}
}
```

---

## 👥 Authors

| Name | Role | Profile |
|------|------|---------|
| **Siddharth Gupta** | Lead Author & ML Engineer | [GitHub](https://github.com/guptasiddharth2409) |
| Pratham Namdev | Co-Author | GNIOT, GGSIPU |
| Shubham Nagar | Co-Author | GNIOT, GGSIPU |
| Sunny Kumar | Co-Author | GNIOT, GGSIPU |
| Anjali Deshwal | Co-Author | GNIOT, GGSIPU |

---

<div align="center">

**Made with ❤️ at GNIOT, GGSIPU · Licensed under MIT**

</div>
