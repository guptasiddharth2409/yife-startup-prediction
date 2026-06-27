# Paper

This paper is published in the **Cureus Journal of Computer Science** (Part of Springer Nature).

> *Previously presented at PTESM 2026, April 10–11, 2026, Greater Noida, India.*

The final accepted manuscript PDF will be uploaded here once Cureus assigns the DOI and bibliographic metadata.

---

## Citation

```bibtex
@article{gupta2026yife,
  title     = {Predicting Early-Stage Startup Success Using ML:
               A YC-Inspired Feature Engineering Approach},
  author    = {Gupta, Siddharth and Namdev, Pratham and Nagar, Shubham
               and Singh, Sunny and Deshwal, Anjali},
  journal   = {Cureus Journal of Computer Science},
  publisher = {Springer Nature},
  year      = {2026}
}
```

> **Note:** DOI, volume, issue, and article number will be added once officially assigned by Cureus.

---

## Abstract

Predicting startup success remains a challenging problem due to high uncertainty and limited early-stage signals. This paper introduces YIFE (YC-Inspired Feature Engineering), a domain-specific feature construction framework trained on 4,323 Y Combinator companies from 2005–2024. YIFE constructs 14 structured features across six signal categories: funding momentum, team quality, technical depth, batch context, industry, and geography. Evaluated across five classifiers, YIFE-enhanced XGBoost achieves an F1-score of 0.85 and AUROC of 0.91, outperforming raw Crunchbase features by +19 F1 points. SHAP analysis reveals funding round count, batch year market conditions, and team size as the dominant predictors — while prior FAANG experience turns out not to be a reliable signal.

---

## Research Impact

| Metric | Value |
|---|---|
| Best Accuracy | 86% (XGBoost + YIFE) |
| Best F1-Score | 0.85 |
| Best AUROC | 0.91 |
| Dataset Size | 4,323 YC-funded startups (2005–2024) |
| Test Cohort | W21–S24 (n=863) |
| Improvement over B1 baseline | +19 F1 points |
