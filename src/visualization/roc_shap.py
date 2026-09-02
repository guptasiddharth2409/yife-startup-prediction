"""Generate ROC and SHAP figures from the current training pipeline.

Figures are generated from the repository's current artifacts. When synthetic
data are used, these figures are demonstrations and should not be presented as
the published paper's original plots.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from src.config import FIG_DIR, LOG_DIR, MODEL_DIR, PROCESSED_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "mlp": "MLP Neural Network",
    "svm": "SVM",
    "logistic": "Logistic Regression",
    "xgb": "XGBoost",
}


def plot_roc_curves():
    pred_files = sorted(LOG_DIR.glob("preds_*.parquet"))
    if not pred_files:
        print("No prediction files found. Run trainer.py first.")
        return

    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=300)
    records = []
    for p in pred_files:
        name = p.stem.replace("preds_", "")
        df = pd.read_parquet(p)
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
        records.append((auc(fpr, tpr), name, fpr, tpr))
    records.sort(key=lambda x: -x[0])

    for auc_val, name, fpr, tpr in records:
        ax.plot(
            fpr, tpr,
            lw=2.8 if name in {"xgb", "xgboost"} else 1.8,
            ls="-" if name in {"xgb", "xgboost"} else "--",
            label=f"{MODEL_LABELS.get(name, name)} (AUC = {auc_val:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k:", lw=1.2, label="Random Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Held-Out YC Test Cohort (W21–S24)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    out = FIG_DIR / "roc_curves.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


def _load_heldout_raw():
    df = pd.read_parquet(PROCESSED_DIR / "yife_features.parquet").copy()
    if "batch_year_encoded" not in df.columns:
        raise ValueError("batch_year_encoded is required for temporal SHAP analysis.")
    years = pd.to_numeric(df["batch_year_encoded"], errors="coerce")
    return df[years >= 2021].copy()


def plot_shap_importance():
    try:
        import shap
    except ImportError:
        print("SHAP is not installed. Skipping SHAP plot.")
        return

    model_path = MODEL_DIR / "xgboost.pkl"
    if not model_path.exists():
        # Backward-compatible artifact name.
        model_path = MODEL_DIR / "xgb.pkl"
    if not model_path.exists():
        print("XGBoost model not found. Run trainer.py first.")
        return

    df = _load_heldout_raw()
    skip = {"company", "success", "batch"}
    X = df[[c for c in df.columns if c not in skip]]
    model = joblib.load(model_path)

    # Current trainer saves a sklearn Pipeline with preprocessing + model.
    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps["preprocess"]
        estimator = model.named_steps["model"]
        X_transformed = preprocessor.transform(X)
        feature_names = preprocessor.get_feature_names_out()
    else:
        # Backward-compatible path for legacy bare XGBoost artifacts.
        estimator = model
        X_transformed = X.astype(float).fillna(0).values
        feature_names = np.asarray(X.columns)

    try:
        booster = estimator.get_booster()
        explainer = shap.TreeExplainer(booster)
        shap_vals = explainer.shap_values(X_transformed)
    except Exception as exc:
        print(f"SHAP explainer failed: {exc}. Skipping SHAP plot.")
        return

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    mean_abs = np.abs(shap_vals).mean(axis=0)

    feat_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    labels = feat_df["feature"].tolist()[::-1]
    values = feat_df["mean_abs_shap"].tolist()[::-1]
    y_pos = np.arange(len(feat_df))

    fig, ax = plt.subplots(figsize=(5.8, 4.2), dpi=300)
    ax.barh(y_pos, values, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean |SHAP| Value")
    ax.set_title("SHAP Global Feature Importance — XGBoost (YIFE)", fontweight="bold")
    for i, v in enumerate(values):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    ax.xaxis.grid(True, linewidth=0.7)
    plt.tight_layout()
    out = FIG_DIR / "shap_importance.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    plot_roc_curves()
    plot_shap_importance()
