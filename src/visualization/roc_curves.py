"""
src/visualization/roc_curves.py
Generate Figure 1: ROC curves for all 5 classifiers.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from src.config import LOG_DIR, FIG_DIR


def plot_roc_curves():
    MODEL_META = {
        'logistic':      ('Logistic Regression', '#4878CF', ':'),
        'random_forest': ('Random Forest',        '#6ACC65', '--'),
        'xgboost':       ('XGBoost \u2605',       '#D65F5F', '-'),
        'svm':           ('SVM',                   '#B47CC7', '-.'),
        'mlp':           ('MLP Neural Network',    '#C4AD66', (0, (3, 1, 1, 1))),
    }
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    for name, (label, color, ls) in MODEL_META.items():
        p = LOG_DIR / f'preds_{name}.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p)
        fpr, tpr, _ = roc_curve(df['y_true'], df['y_prob'])
        roc_auc = auc(fpr, tpr)
        lw = 2.5 if name == 'xgboost' else 1.8
        ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
                label=f'{label} (AUC = {roc_auc:.3f})')
    ax.plot([0,1],[0,1],'k--',lw=1,label='Random Chance')
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
           title='Figure 1: ROC Curves \u2014 Held-Out Test Set (W21\u2013S24)',
           xlim=[0,1], ylim=[0,1.02])
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / 'roc_curves.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    plot_roc_curves()
