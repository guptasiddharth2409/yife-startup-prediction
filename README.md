# YIFE: YC-Inspired Feature Engineering for Startup Outcome Classification

[![CI](https://github.com/guptasiddharth2409/yife-startup-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/guptasiddharth2409/yife-startup-prediction/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Published](https://img.shields.io/badge/Published-Cureus%20Journal%20of%20Computer%20Science-red.svg)](https://doi.org/10.7759/s44389-026-00254-0)
[![XGBoost](https://img.shields.io/badge/Champion-XGBoost-F1%3D0.85-brightgreen.svg)](#published-results)

> **Predicting Startup Outcomes Using Explainable Machine Learning and Y Combinator-Inspired Feature Engineering**  
> Siddharth Gupta · Pratham Namdev · Shubham Nagar · Sunny Singh · Anjali Deshwal  
> Department of Computer Science & Engineering, Greater Noida Institute of Technology  
> **Published in Cureus Journal of Computer Science (Part of Springer Nature), September 1, 2026**

## 📌 Publication

The final paper is published in the **Cureus Journal of Computer Science**.

- **DOI:** https://doi.org/10.7759/s44389-026-00254-0
- **Published:** September 1, 2026
- **Dataset scope:** 4,323 YC-funded companies, W05–S24 (2005–2024)
- **Champion model:** XGBoost with YIFE
- **Held-out evaluation cohort:** W21–S24, n = 863

The project was previously presented at PTESM 2026 (April 10–11, 2026). The repository now reflects the published-paper terminology and framing.

## 🔍 Overview

YIFE (YC-Inspired Feature Engineering) is a domain-specific feature engineering framework for analyzing startup outcomes within the Y Combinator ecosystem. The study evaluates five supervised-learning classifiers using accelerator-specific structural signals alongside conventional funding and company features.

The final curated dataset contains **4,323 YC-funded companies from 2005 through 2024**. The published evaluation uses a temporal split: W05–S20 for training and W21–S24 as an untouched held-out cohort. The paper reports XGBoost with YIFE as the strongest configuration, achieving **F1 = 0.85** and **AUROC = 0.91** on the held-out cohort.

## ⚠️ Important Interpretation Note

This project should **not** be interpreted as a strict ex-ante startup screening or investment-decision system.

Several predictors, especially cumulative funding and Tier-1 investor participation, may become available after the initial YC stage and can overlap conceptually with the success definition. The published work therefore frames YIFE as a **domain-contextualized retrospective classification approach for the YC ecosystem**.

Newer cohorts also have shorter outcome-observation windows, creating potential right-censoring. Results are specific to the YC ecosystem and should not be assumed to generalize directly to other accelerators or the broader startup population.

## 🏆 Published Results

Performance on the temporally held-out **W21–S24 test cohort (n = 863)**:

| Model | Accuracy | Precision | Recall | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (B1 baseline) | 0.71 | 0.68 | 0.64 | 0.66 | 0.74 |
| Logistic Regression (YIFE) | 0.75 | 0.73 | 0.70 | 0.71 | 0.79 |
| Random Forest (B2 baseline) | 0.80 | 0.79 | 0.76 | 0.77 | 0.85 |
| Random Forest (YIFE) | 0.83 | 0.82 | 0.79 | 0.80 | 0.88 |
| **XGBoost (YIFE) ★** | **0.86** | **0.85** | **0.85** | **0.85** | **0.91** |
| Support Vector Machine (YIFE) | 0.79 | 0.78 | 0.74 | 0.76 | 0.83 |
| MLP Neural Network (YIFE) | 0.81 | 0.80 | 0.78 | 0.79 | 0.86 |

The XGBoost YIFE configuration improves F1 by **19 percentage points** over the generic Crunchbase baseline (0.85 vs. 0.66) and by **8 percentage points** over the replicated B2 configuration (0.85 vs. 0.77).

## 🧩 YIFE Feature Set

The published feature framework contains 14 features spanning funding, team, technical depth, batch context, industry, geography, and investor-network signals.

| Category | Feature | Type | Description |
|---|---|---|---|
| Funding | `total_funding_usd` | Continuous | Total capital raised across rounds |
| Funding | `num_funding_rounds` | Integer | Number of distinct funding events |
| Funding | `seed_round_size` | Continuous | Seed round amount |
| Team | `team_size` | Integer | Number of co-founders at application |
| Team | `faang_experience` | Binary | Any founder had prior FAANG experience |
| Team | `elite_edu` | Binary | Any founder attended a top-20 university |
| Technical Depth | `github_repo_count` | Integer | Public repositories of the founding team |
| Technical Depth | `github_commit_freq` | Continuous | Average weekly commits in the six months before YC |
| Batch Context | `batch_year_encoded` | Ordinal | Batch-year/cohort context |
| Batch Context | `batch_size` | Integer | Number of companies in the same batch |
| Industry | `industry_category` | Categorical | YC-assigned industry category |
| Industry | `ai_flag` | Binary | AI is a core product component |
| Geography | `geo_cluster` | Categorical | Geographic cluster |
| Network | `tier1_vc_investor` | Binary | Tier-1 VC backing after YC |

> **Note:** The published paper describes the framework as a 14-feature YIFE set. Its table contains seven named signal group labels (Funding, Team, Technical Depth, Batch Context, Industry, Geography, Network); this README avoids calling those groups “six categories” to prevent ambiguity.

## 📊 SHAP Interpretation

SHAP analysis was performed on the best-performing XGBoost model using the 863 held-out test instances.

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | `num_funding_rounds` | 0.187 |
| 2 | `batch_year_encoded` | 0.163 |
| 3 | `team_size` | 0.141 |
| 4 | `total_funding_usd` | 0.128 |
| 5 | `ai_flag` | 0.112 |
| 6 | `industry_category (B2B)` | 0.094 |
| 7 | `github_commit_freq` | 0.087 |
| 8 | `geo_cluster (San Francisco Bay)` | 0.071 |
| 9 | `elite_edu` | 0.052 |
| 10 | `faang_experience` | 0.031 |

**Interpretation:** Mean absolute SHAP values describe relative attribution magnitude and ranking. They do **not**, by themselves, establish causal direction, monotonicity, or an optimal feature value. The published analysis deliberately avoids making directional claims without signed SHAP dependence analysis.

## 🧪 Success Label

A company is labeled successful if it satisfies at least one of the following published criteria:

1. Raised a Series A round or beyond after YC.
2. Was acquired at a disclosed valuation.
3. Remained in active operation with measurable growth beyond three years after its batch.

Approximately 45% of the curated dataset is labeled successful. The latest cohorts have incomplete outcome maturation, so cohort success rates should be treated as descriptive rather than directly comparable estimates of underlying success probability.

## 🔬 Experimental Design

The published study uses:

- **Training cohorts:** W05–S20, approximately 3,460 companies
- **Held-out test cohorts:** W21–S24, n = 863
- **Hyperparameter tuning:** 3-fold cross-validated grid search within training data
- **Model selection:** 5-fold stratified cross-validation within training data
- **Final evaluation:** one evaluation on the untouched W21–S24 cohort
- **Primary metrics:** F1-score and AUROC
- **Secondary metrics:** Accuracy, Precision, Recall
- **Explainability:** global mean absolute SHAP attribution on the held-out test set

The temporal split reduces cohort-level look-ahead bias, but it does not eliminate leakage from predictors that themselves contain post-incubation information.

## 🗃️ Data Sources & Availability

The research dataset was assembled from complementary public and access-controlled sources:

| Source | Research use |
|---|---|
| YC Company Directory / Kaggle | Batch, category, status, team size, location |
| Crunchbase | Funding rounds, total funding, lead-investor information |
| GitHub Public API | Founder public repositories and commit activity |
| LinkedIn manual sample | Founder education and prior-employer information for 500 profiles |

The original research data are **not redistributed in this repository**. Access to third-party datasets is subject to their respective terms. The repository includes a deterministic synthetic-data generator so the software pipeline can be exercised without redistributing the underlying research records.

> The synthetic dataset is a **pipeline/reproducibility aid**, not a byte-for-byte replacement for the original research dataset and should not be used to claim reproduction of the published numerical results.

See [`data/README.md`](data/README.md) for the data setup details.

## 📁 Repository Structure

```text
yife-startup-prediction/
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── requirements-optional.txt
├── configs/
│   └── model_config.yaml
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── collect_kaggle.py
│   │   ├── collect_crunchbase.py
│   │   ├── collect_github.py
│   │   └── generate_synthetic.py
│   ├── preprocess/
│   │   └── prepare_features.py
│   ├── train/
│   │   └── trainer.py
│   └── visualization/
│       └── roc_shap.py
├── scripts/
│   ├── 01_generate_synthetic_data.py
│   ├── 02_train_model.py
│   └── 03_generate_figures.py
├── notebooks/
├── app/
├── data/
├── models/
├── figures/
├── logs/
└── paper/
```

## ⚡ Quickstart

### 1. Install

```bash
git clone https://github.com/guptasiddharth2409/yife-startup-prediction.git
cd yife-startup-prediction
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate the synthetic pipeline dataset

```bash
python src/data/generate_synthetic.py
```

This creates `data/processed/yife_features.parquet` with 4,323 synthetic records and the published 14-feature schema.

### 3. Run the reference training pipeline

```bash
python src/train/trainer.py
```

Outputs are written to `models/` and `logs/`.

### 4. Generate figures

```bash
python src/visualization/roc_shap.py
```

> **Reproducibility warning:** Running the synthetic pipeline reproduces the repository's workflow and schema, but it does not recreate the original research dataset. Therefore, the published F1/AUROC values should be treated as the reported paper results, not as expected outputs from synthetic data.

## 🖥️ Interactive Demo

An optional Streamlit interface is included:

```bash
streamlit run app/streamlit_app.py
```

The interface is intended for **demonstration and educational exploration**, not investment advice. Its predictions use the trained XGBoost artifact and the same feature definitions documented in this repository.

## 📦 Environment

The project targets Python 3.10+ and uses NumPy, pandas, scikit-learn, XGBoost, SHAP, PyTorch, and related tooling. See `requirements.txt` for the current dependency ranges.

## 📚 Citation

```bibtex
@article{gupta2026yife,
  title   = {Predicting Startup Outcomes Using Explainable Machine Learning and Y Combinator-Inspired Feature Engineering},
  author  = {Gupta, Siddharth and Namdev, Pratham and Nagar, Shubham and Singh, Sunny and Deshwal, Anjali},
  journal = {Cureus Journal of Computer Science},
  volume  = {3},
  year    = {2026},
  doi     = {10.7759/s44389-026-00254-0}
}
```

## 📄 License

MIT License. See [`LICENSE`](LICENSE).

## 🙏 Acknowledgements

The authors thank the Department of Computer Science & Engineering at Greater Noida Institute of Technology for institutional support and the open-source community behind scikit-learn, XGBoost, SHAP, PyTorch, and related tools.

## 🔭 Future Work

The published paper identifies several directions for stronger forecasting validity:

- fixed prediction timestamps and follow-up horizons
- explicit leakage-controlled feature availability
- survival-aware analysis for right-censored cohorts
- signed SHAP dependence and local explanations
- validation beyond the YC ecosystem

---

**Status:** Published  
**Last repository alignment:** September 2026