"""
evaluate.py — Compute metrics and generate figures for all YIFE models.

Outputs:
    figures/roc_curves.png
    figures/confusion_matrices.png
    figures/metrics_table.csv

Usage:
    python src/evaluate.py
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    classification_report, f1_score, precision_score, recall_score
)
from utils import load_config, set_seed, ensure_dir, print_banner


# ── Palette ──────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "XGBoost":             "#01696f",
    "Random Forest":       "#437a22",
    "MLP":                 "#006494",
    "SVM":                 "#da7101",
    "Logistic Regression": "#a12c7b",
}


def load_artifacts():
    """Load saved models and test data."""
    models = {}
    model_names = {
        "logistic_regression": "Logistic Regression",
        "svm": "SVM",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "mlp": "MLP",
    }
    for fname, name in model_names.items():
        path = f"models/{fname}.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)

    X_test = np.load("models/X_test.npy")
    y_test = np.load("models/y_test.npy")
    with open("models/feature_cols.json") as f:
        feature_cols = json.load(f)

    return models, X_test, y_test, feature_cols


def plot_roc_curves(models, X_test, y_test, out_path: str) -> dict:
    """Plot overlaid ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#f7f6f2")
    ax.set_facecolor("#f9f8f5")

    aurocs = {}
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        score = auc(fpr, tpr)
        aurocs[name] = score
        lw = 2.5 if name == "XGBoost" else 1.5
        ax.plot(fpr, tpr, lw=lw, color=MODEL_COLORS.get(name, "gray"),
                label=f"{name}  (AUC = {score:.3f})",
                zorder=5 if name == "XGBoost" else 3)

    ax.plot([0, 1], [0, 1], "--", color="#bab9b4", lw=1.2, label="Random chance")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — YIFE Feature Set", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out_path}")
    return aurocs


def plot_confusion_matrices(models, X_test, y_test, out_path: str):
    """Plot confusion matrix grid for all models."""
    n = len(models)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4.5))
    fig.patch.set_facecolor("#f7f6f2")
    axes = axes.flatten()

    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        color = MODEL_COLORS.get(name, "#01696f")

        # Build custom colormap from white → model color
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("yife", ["#f9f8f5", color])

        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap,
            ax=axes[i], linewidths=0.5, linecolor="#dcd9d5",
            cbar=False, annot_kws={"size": 13, "weight": "bold"}
        )
        axes[i].set_title(name, fontsize=12, fontweight="bold", pad=8)
        axes[i].set_xlabel("Predicted", fontsize=9)
        axes[i].set_ylabel("Actual", fontsize=9)
        axes[i].set_xticklabels(["Fail", "Success"])
        axes[i].set_yticklabels(["Fail", "Success"], rotation=0)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices — YIFE Feature Set", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out_path}")


def compute_metrics_table(models, X_test, y_test, aurocs: dict, out_path: str) -> pd.DataFrame:
    """Build and save a metrics comparison table."""
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "AUROC": round(aurocs.get(name, float("nan")), 4),
        })

    df = pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved: {out_path}")
    print("\n" + df.to_string(index=False))
    return df


def run(config_path: str = "configs/model_config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["global"]["seed"])
    print_banner("Evaluating YIFE Models")
    ensure_dir(cfg["global"]["figures_dir"])

    models, X_test, y_test, _ = load_artifacts()

    aurocs = plot_roc_curves(
        models, X_test, y_test,
        out_path=os.path.join(cfg["global"]["figures_dir"], "roc_curves.png")
    )
    plot_confusion_matrices(
        models, X_test, y_test,
        out_path=os.path.join(cfg["global"]["figures_dir"], "confusion_matrices.png")
    )
    compute_metrics_table(
        models, X_test, y_test, aurocs,
        out_path=os.path.join(cfg["global"]["figures_dir"], "metrics_table.csv")
    )


if __name__ == "__main__":
    run()
