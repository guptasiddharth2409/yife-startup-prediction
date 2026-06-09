"""
shap_analysis.py — SHAP feature importance for the best YIFE model (XGBoost).

Outputs:
    figures/shap_bar.png        — Mean |SHAP| bar chart
    figures/shap_beeswarm.png   — Beeswarm summary plot

Usage:
    python src/shap_analysis.py
"""

import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import shap
from utils import load_config, set_seed, ensure_dir, print_banner


def run(config_path: str = "configs/model_config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["global"]["seed"])
    print_banner("SHAP Explainability — XGBoost")
    ensure_dir(cfg["global"]["figures_dir"])

    with open("models/xgboost.pkl", "rb") as f:
        model = pickle.load(f)

    X_test = np.load("models/X_test.npy")
    with open("models/feature_cols.json") as f:
        feature_cols = json.load(f)

    print("  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # ── Bar chart — mean |SHAP| ───────────────────────────────────────────────
    mean_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_shap)[::-1]
    feat_names = [feature_cols[i] for i in order]
    vals = mean_shap[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#f7f6f2")
    ax.set_facecolor("#f9f8f5")

    bars = ax.barh(feat_names[::-1], vals[::-1], color="#01696f", alpha=0.85, edgecolor="none")
    bars[0].set_color("#0c4e54")  # highlight top feature

    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Feature Importance (XGBoost + YIFE)", fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    bar_path = os.path.join(cfg["global"]["figures_dir"], "shap_bar.png")
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {bar_path}")

    # ── Beeswarm ──────────────────────────────────────────────────────────────
    plt.figure(figsize=(9, 6))
    plt.gcf().patch.set_facecolor("#f7f6f2")
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_cols,
        show=False,
        color_bar_label="Feature value",
        plot_size=None,
    )
    plt.title("SHAP Beeswarm — XGBoost + YIFE", fontsize=13, fontweight="bold")
    plt.tight_layout()
    beeswarm_path = os.path.join(cfg["global"]["figures_dir"], "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {beeswarm_path}")


if __name__ == "__main__":
    run()
