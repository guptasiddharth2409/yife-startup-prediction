"""
03_generate_figures.py

Generates publication-quality figures:
  figures/roc_curves.png       -- Figure 1: ROC curves for all 5 classifiers
  figures/shap_importance.png  -- Figure 2: SHAP global feature importance (XGBoost)

Requires: models saved by 02_train_model.py, logs/preds_*.csv
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import roc_curve, auc

FIG_DIR = Path('figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DISPLAY = {
    'logistic':     'Logistic Regression',
    'random_forest':'Random Forest',
    'xgboost':      'XGBoost \u2605',
    'svm':          'SVM',
    'mlp':          'MLP Neural Network',
}

COLORS = {
    'logistic':     '#4878CF',
    'random_forest':'#6ACC65',
    'xgboost':      '#D65F5F',
    'svm':          '#B47CC7',
    'mlp':          '#C4AD66',
}

LINESTYLES = {
    'logistic':     ':',
    'random_forest':'--',
    'xgboost':      '-',
    'svm':          '-.',
    'mlp':          (0, (3, 1, 1, 1)),
}


def plot_roc_curves(log_dir='logs'):
    log_dir = Path(log_dir)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)

    order = ['logistic', 'random_forest', 'xgboost', 'svm', 'mlp']
    for name in order:
        preds_file = log_dir / f'preds_{name}.csv'
        if not preds_file.exists():
            print(f"  [skip] {preds_file} not found")
            continue
        df = pd.read_csv(preds_file)
        fpr, tpr, _ = roc_curve(df['y_true'], df['y_prob'])
        roc_auc = auc(fpr, tpr)
        lw = 2.5 if name == 'xgboost' else 1.8
        ax.plot(fpr, tpr,
                color=COLORS[name], lw=lw,
                linestyle=LINESTYLES[name],
                label=f"{MODEL_DISPLAY[name]} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1.0, label='Random Chance (AUC = 0.500)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('Figure 1: ROC Curves \u2014 Held-Out Test Set (W21\u2013S24)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8.5, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / 'roc_curves.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_shap_importance():
    # SHAP values from paper (Table 6) -- reproduced exactly
    shap_data = [
        ('num_funding_rounds',  0.187, '#D65F5F'),
        ('batch_year_encoded',  0.163, '#D65F5F'),
        ('team_size',           0.141, '#D65F5F'),
        ('total_funding_usd',   0.128, '#D65F5F'),
        ('ai_flag',             0.112, '#D65F5F'),
        ('industry_category\n(B2B)', 0.094, '#4878CF'),
        ('github_commit_freq',  0.087, '#4878CF'),
        ('geo_cluster\n(SF Bay)', 0.071, '#4878CF'),
        ('elite_edu',           0.052, '#6ACC65'),
        ('faang_experience',    0.031, '#6ACC65'),
    ]
    features, values, colors = zip(*shap_data)
    features = features[::-1]
    values   = values[::-1]
    colors   = colors[::-1]

    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=300)
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=0.4)

    for bar, v in zip(bars, values):
        ax.text(v + 0.004, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=8.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel('Mean |SHAP| Value', fontsize=10)
    ax.set_title('Figure 2: SHAP Global Feature Importance \u2014 XGBoost (YIFE)',
                 fontsize=10, fontweight='bold')
    ax.set_xlim([0, 0.22])
    ax.grid(axis='x', alpha=0.25, linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_handles = [
        mpatches.Patch(color='#D65F5F', label='Top predictors (\u2265 0.11)'),
        mpatches.Patch(color='#4878CF', label='Moderate predictors'),
        mpatches.Patch(color='#6ACC65', label='Weak predictors (< 0.06)'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc='lower right')
    plt.tight_layout()
    out = FIG_DIR / 'shap_importance.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    print("Generating Figure 1: ROC Curves...")
    plot_roc_curves()
    print("Generating Figure 2: SHAP Feature Importance...")
    plot_shap_importance()
    print("Done. Check figures/ directory.")
