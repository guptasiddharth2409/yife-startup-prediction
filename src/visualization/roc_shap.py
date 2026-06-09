"""YIFE Visualization: ROC Curves + SHAP Feature Importance

Generates two publication-quality figures (300 DPI):
  figures/roc_curves.png       - Figure 1 of the paper
  figures/shap_importance.png  - Figure 2 of the paper

Requires:
  - logs/preds_*.parquet        (output of trainer.py)
  - models/xgb.pkl              (saved XGBoost model)
  - data/processed/yife_features.parquet
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import FIG_DIR, LOG_DIR, MODEL_DIR, PROCESSED_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "xgb":           "#2563eb",
    "random_forest": "#16a34a",
    "mlp":           "#7c3aed",
    "svm":           "#ea580c",
    "logistic":      "#6b7280",
}
MODEL_LABELS = {
    "xgb":           "XGBoost",
    "random_forest": "Random Forest",
    "mlp":           "MLP Neural Network",
    "svm":           "SVM",
    "logistic":      "Logistic Regression",
}


def plot_roc_curves():
    from sklearn.metrics import roc_curve, auc
    pred_files = sorted(LOG_DIR.glob("preds_*.parquet"))
    if not pred_files:
        print("No prediction files found in logs/. Run trainer.py first.")
        return

    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=300)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.xaxis.grid(True, color="#e5e7eb", linewidth=0.7)

    records = []
    for p in pred_files:
        name = p.stem.replace("preds_", "")
        df   = pd.read_parquet(p)
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
        records.append((auc(fpr, tpr), name, fpr, tpr))
    records.sort(key=lambda x: -x[0])

    for auc_val, name, fpr, tpr in records:
        color = MODEL_COLORS.get(name, "#374151")
        label = MODEL_LABELS.get(name, name.capitalize())
        ax.plot(fpr, tpr, color=color,
                lw=2.8 if name=="xgb" else 1.8,
                ls="-" if name=="xgb" else "--",
                label=f"{label} (AUC = {auc_val:.3f})")

    ax.plot([0,1],[0,1], "k:", lw=1.2, label="Random Chance")
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curves — Held-Out YC Test Set (W21–S24)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    out = FIG_DIR / "roc_curves.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC figure -> {out}")


def plot_shap_importance():
    try:
        import shap, joblib
    except ImportError:
        print("shap/joblib not installed.")
        return
    xgb_path = MODEL_DIR / "xgb.pkl"
    if not xgb_path.exists():
        print("XGBoost model not found. Run trainer.py first.")
        return
    df = pd.read_parquet(PROCESSED_DIR / "yife_features.parquet")
    skip = {"company","success","batch"}
    X_cols = [c for c in df.columns if c not in skip]
    X = df[X_cols].astype(float).values
    model     = joblib.load(xgb_path)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    feat_df   = (pd.DataFrame({"feature":X_cols,"mean_abs_shap":mean_abs})
                   .sort_values("mean_abs_shap",ascending=False).head(10).reset_index(drop=True))

    def _color(v):
        if v >= 0.12: return "#1d4ed8"
        if v >= 0.07: return "#3b82f6"
        return "#93c5fd"

    labels = feat_df["feature"].tolist()[::-1]
    values = feat_df["mean_abs_shap"].tolist()[::-1]
    colors = [_color(v) for v in values]
    y_pos  = np.arange(len(feat_df))

    fig, ax = plt.subplots(figsize=(5.8, 4.2), dpi=300)
    ax.barh(y_pos, values, color=colors, height=0.65, edgecolor="white")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean |SHAP| Value", fontsize=10)
    ax.set_title("SHAP Global Feature Importance — XGBoost (YIFE)", fontsize=11, fontweight="bold")
    for i,v in enumerate(values):
        ax.text(v+0.003, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e5e7eb", linewidth=0.7)
    plt.tight_layout()
    out = FIG_DIR / "shap_importance.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP figure -> {out}")


if __name__ == "__main__":
    plot_roc_curves()
    plot_shap_importance()
