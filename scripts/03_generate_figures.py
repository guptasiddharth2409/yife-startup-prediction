"""Compatibility entry point for figure generation.

The canonical implementation is ``src/visualization/roc_shap.py``. This wrapper
prevents the older script from generating figures from stale CSV/model formats
or hard-coded SHAP values.
"""

from src.visualization.roc_shap import plot_roc_curves, plot_shap_importance


if __name__ == "__main__":
    print("Generating ROC curves...")
    plot_roc_curves()
    print("Generating SHAP feature-importance plot...")
    plot_shap_importance()
    print("Done. Check figures/ directory.")
