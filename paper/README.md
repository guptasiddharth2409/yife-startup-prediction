# Published Paper

This repository accompanies the published article:

**Predicting Startup Outcomes Using Explainable Machine Learning and Y Combinator-Inspired Feature Engineering**

**Authors:** Siddharth Gupta, Pratham Namdev, Shubham Nagar, Sunny Singh, Anjali Deshwal

Published in the **Cureus Journal of Computer Science**, September 1, 2026.

- **DOI:** https://doi.org/10.7759/s44389-026-00254-0
- **Volume:** 3
- **Article identifier:** es44389-026-00254-0
- **Publisher:** Cureus / Springer Nature

The work was previously presented at PTESM 2026, April 10–11, 2026, Greater Noida, India.

> The repository does not redistribute the original research dataset. See [`../data/README.md`](../data/README.md) for data availability and reproducibility guidance.

---

## Citation

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

---

## Abstract-level Summary

The study introduces **YIFE (YC-Inspired Feature Engineering)**, a domain-specific framework combining accelerator cohort context, founder/team characteristics, technical activity, industry, geography, funding, and investor-network signals. Five classifiers were evaluated on a curated dataset of **4,323 YC-funded companies spanning 2005–2024** using a temporal train/test design. XGBoost with YIFE achieved **F1 = 0.85** and **AUROC = 0.91** on the held-out W21–S24 cohort (n = 863). SHAP analysis was used to quantify relative feature-attribution magnitude.

The published paper frames YIFE as a **retrospective, domain-contextualized classification approach** rather than a strict ex-ante forecasting tool because some funding and investor-network predictors can be observed after the initial YC stage and may overlap conceptually with the outcome definition.

---

## Published Results

| Metric | Value |
|---|---:|
| Best model | XGBoost + YIFE |
| Accuracy | 0.86 |
| Precision | 0.85 |
| Recall | 0.85 |
| F1-score | 0.85 |
| AUROC | 0.91 |
| Held-out cohort | W21–S24 |
| Held-out sample | 863 companies |
| F1 improvement vs. generic B1 baseline | +0.19 |

See the main repository README for the complete model comparison and reproducibility instructions.
