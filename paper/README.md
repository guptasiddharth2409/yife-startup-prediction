# Paper

The full paper will be uploaded here after formal publication at **PTEMS-2026**.

---

## Citation

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

## Abstract

Predicting startup success remains a challenging problem due to high uncertainty and limited early-stage signals. This paper introduces YIFE (YC-Inspired Feature Engineering), a domain-specific feature construction framework trained on 4,323 Y Combinator companies from 2005–2024. YIFE constructs 14 structured features across five signal categories: funding momentum, team quality, technical depth, batch context, and industry/geography. Evaluated across five classifiers, YIFE-enhanced XGBoost achieves an F1-score of 0.85 and AUROC of 0.91, outperforming raw Crunchbase features by +11 F1 points. SHAP analysis reveals founder quality signals (FAANG experience, elite education) as the dominant predictors, followed by funding round count over raw funding amount.
