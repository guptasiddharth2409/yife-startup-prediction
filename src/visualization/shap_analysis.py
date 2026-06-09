"""
src/visualization/shap_analysis.py
Generate Figure 2: SHAP global feature importance for XGBoost (YIFE).
Can run in two modes:
  1. With trained model (computes real SHAP values)
  2. From paper Table 6 values (fallback for reproducibility)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from src.config import FIG_DIR, MODEL_DIR, PROCESSED_DIR, YIFE_FEATURES


# Paper Table 6: SHAP values for reproducibility
PAPER_SHAP = [
    ('num_funding_rounds',   0.187),
    ('batch_year_encoded',   0.163),
    ('team_size',            0.141),
    ('total_funding_usd',    0.128),
    ('ai_flag',              0.112),
    ('industry_category (B2B)', 0.094),
    ('github_commit_freq',   0.087),
    ('geo_cluster (SF Bay)', 0.071),
    ('elite_edu',            0.052),
    ('faang_experience',     0.031),
]


def plot_from_paper_values():
    """Reproduce Figure 2 from Table 6 values (no model required)."""
    features, values = zip(*PAPER_SHAP)
    features = list(features)[::-1]
    values   = list(values)[::-1]
    colors = ['#D65F5F' if v >= 0.11 else '#4878CF' if v >= 0.07
              else '#6ACC65' for v in values]

    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=300)
    bars = ax.barh(np.arange(len(features)), values,
                   color=colors, edgecolor='white', linewidth=0.4)
    for bar, v in zip(bars, values):
        ax.text(v + 0.004, bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=8.5)
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel('Mean |SHAP| Value', fontsize=10)
    ax.set_title('Figure 2: SHAP Global Feature Importance \u2014 XGBoost (YIFE)',
                 fontsize=10, fontweight='bold')
    ax.set_xlim([0, 0.22])
    ax.grid(axis='x', alpha=0.25, linewidth=0.6)
    ax.spines[['top','right']].set_visible(False)
    legend_handles = [
        mpatches.Patch(color='#D65F5F', label='High importance (\u2265 0.11)'),
        mpatches.Patch(color='#4878CF', label='Moderate importance'),
        mpatches.Patch(color='#6ACC65', label='Low importance (< 0.06)'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc='lower right')
    plt.tight_layout()
    out = FIG_DIR / 'shap_importance.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_from_model():
    """Compute real SHAP values from saved XGBoost model."""
    try:
        import shap, joblib
        model  = joblib.load(MODEL_DIR / 'xgboost.pkl')
        df     = pd.read_csv(PROCESSED_DIR / 'yife_features.csv')
        feature_cols = [f for f in YIFE_FEATURES if f in df.columns
                        and df[f].dtype != object]
        X = df[feature_cols].fillna(0).values
        explainer = shap.Explainer(model)
        shap_vals = explainer(X)
        mean_abs  = np.abs(shap_vals.values).mean(axis=0)
        imp = (pd.DataFrame({'feature': feature_cols, 'shap': mean_abs})
               .sort_values('shap', ascending=False).head(10))
        features = imp['feature'].tolist()[::-1]
        values   = imp['shap'].tolist()[::-1]
        colors   = ['#D65F5F' if v >= 0.10 else '#4878CF' if v >= 0.06
                    else '#6ACC65' for v in values]
        fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=300)
        bars = ax.barh(np.arange(len(features)), values, color=colors)
        for bar, v in zip(bars, values):
            ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{v:.3f}', va='center', fontsize=8.5)
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Mean |SHAP| Value', fontsize=10)
        ax.set_title('Figure 2: SHAP Global Feature Importance \u2014 XGBoost (YIFE)',
                     fontsize=10, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        out = FIG_DIR / 'shap_importance_computed.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out}')
    except Exception as e:
        print(f'Model-based SHAP failed ({e}), using paper values instead.')
        plot_from_paper_values()


if __name__ == '__main__':
    import sys
    if '--from-model' in sys.argv:
        plot_from_model()
    else:
        plot_from_paper_values()
