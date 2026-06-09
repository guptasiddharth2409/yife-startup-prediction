"""YIFE Model Training Pipeline

Applies a strict temporal train-test split (W05-S20 train | W21-S24 test),
trains five classifiers, and writes:
  models/<name>.pkl          : serialized model artifacts
  logs/metrics_test.json     : per-model evaluation metrics
  logs/preds_<name>.parquet  : predictions used by roc_shap.py
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from src.config import PROCESSED_DIR, MODEL_DIR, LOG_DIR


def load_data():
    df = pd.read_parquet(PROCESSED_DIR / "yife_features.parquet")

    def _year(b):
        try:
            yy = int(str(b)[1:]) if isinstance(b, str) else int(b)
            return 2000 + yy if yy <= 30 else (1900 + yy if yy < 100 else yy)
        except Exception:
            return 2010

    df["_by"] = df["batch"].apply(_year) if "batch" in df.columns else df["batch_year_encoded"].astype(int)

    train = df[df["_by"] < 2021]
    test  = df[df["_by"] >= 2021]

    skip   = {"company", "success", "batch", "_by"}
    X_cols = [c for c in df.columns if c not in skip]

    X_train = train[X_cols].astype(float).values
    y_train = train["success"].values
    X_test  = test[X_cols].astype(float).values
    y_test  = test["success"].values

    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}  |  Test success rate: {y_test.mean():.3f}")
    return X_train, y_train, X_test, y_test, X_cols


def fit_and_eval():
    X_train, y_train, X_test, y_test, feature_names = load_data()

    models = {
        "logistic":      LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=15,
                             class_weight="balanced", n_jobs=-1, random_state=42),
        "xgb":           XGBClassifier(n_estimators=300, learning_rate=0.05,
                             eval_metric="logloss", n_jobs=-1, random_state=42),
        "svm":           SVC(kernel="rbf", C=10, gamma="scale",
                             class_weight="balanced", probability=True, random_state=42),
        "mlp":           MLPClassifier(hidden_layer_sizes=(128,64,32),
                             learning_rate_init=0.001, max_iter=300,
                             early_stopping=True, random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...", flush=True)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / f"{name}.pkl")

        y_pred = model.predict(X_test)
        y_prob = (model.predict_proba(X_test)[:, 1]
                  if hasattr(model, "predict_proba")
                  else model.decision_function(X_test))

        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            "accuracy":        float(accuracy_score(y_test, y_pred)),
            "precision":       float(precision_score(y_test, y_pred, zero_division=0)),
            "recall":          float(recall_score(y_test, y_pred, zero_division=0)),
            "f1":              float(f1_score(y_test, y_pred, zero_division=0)),
            "auroc":           float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix": cm.tolist(),
        }
        results[name] = metrics
        pd.DataFrame({"y_true": y_test, "y_prob": y_prob}).to_parquet(
            LOG_DIR / f"preds_{name}.parquet", index=False
        )
        print(f"  {name:<15s}  F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}")

    with open(LOG_DIR / "metrics_test.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print(f"{'Model':<20} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'AUROC':>7}")
    print("-"*70)
    for name, m in results.items():
        print(f"{name:<20} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>7.4f} {m['auroc']:>7.4f}")
    print("="*70)


if __name__ == "__main__":
    fit_and_eval()
