# Trained Models

This directory stores serialized model artifacts produced by `src/train/trainer.py`.

## Files (generated — not committed to git)

| File | Algorithm | F1-Score | AUROC | Notes |
|---|---|---|---|---|
| `xgb.pkl` | XGBoost (Champion) | **0.85** | **0.91** | Primary model; used for SHAP analysis |
| `random_forest.pkl` | Random Forest | 0.80 | 0.88 | Strong ensemble baseline |
| `mlp.pkl` | MLP Neural Network | 0.79 | 0.86 | 3-layer: 128→64→32 |
| `svm.pkl` | SVM (RBF kernel) | 0.76 | 0.83 | C=10, gamma=scale |
| `logistic.pkl` | Logistic Regression | 0.66 | 0.74 | B1 Baseline |

## Why Models Are Not Committed

Model `.pkl` files are binary artifacts that can be large (50–200 MB for XGBoost
with 300 estimators). They are excluded via `.gitignore` to keep the repository
lean and reproducible.

**To generate models locally:**

```bash
python src/data/generate_synthetic.py   # step 1: create dataset
python src/train/trainer.py             # step 2: train & save all 5 models
```

Training takes approximately **2–5 minutes** on a standard laptop (CPU).

## Loading a Saved Model

```python
import joblib
import pandas as pd

# Load champion model
model = joblib.load("models/xgb.pkl")

# Run prediction on new data
X_new = pd.read_csv("data/raw/sample_yc_data.csv")
X_feat = X_new[["total_funding_usd", "num_funding_rounds", "team_size",
                 "faang_experience", "elite_edu", "ai_flag", "tier1_vc_investor"]]
preds = model.predict(X_feat)
print(preds)  # 0 = not successful, 1 = successful
```
