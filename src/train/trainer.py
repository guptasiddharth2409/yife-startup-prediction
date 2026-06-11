"""YIFE Model Training Pipeline"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object/string columns in-place."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def load_data():
    df = pd.read_parquet(PROCESSED_DIR / "yife_features.parquet")

    # Encode categoricals BEFORE numeric conversion
    df = encode_categoricals(df)

    def _year(b):
        try:
            return int(b) if int(b) > 100 else 2000 + int(b)
        except Exception:
            return 2010

    # batch_year_encoded is already int after label encoding of batch
    if "batch_year_encoded" in df.columns:
        df["_by"] = df["batch_year_encoded"].astype(int)
    else:
        df["_by"] = df["batch"].apply(_year)

    train = df[df["_by"] < 2021]
    test  = df[df["_by"] >= 2021]

    skip   = {"company", "success", "batch", "_by"}
    X_cols = [c for c in df.columns if c not in skip]

    X_train = train[X_cols].astype(float).fillna(0).values
    y_train = train["success"].values
    X_test  = test[X_cols].astype(float).fillna(0).values
    y_test  = test["success"].values

    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}  |  "
          f"Test success rate: {y_test.mean():.3f}")
    return X_train, y_train, X_test, y_test, X_cols


def fit_and_eval():
    X_train, y_train, X_test, y_test, feature_names = load_data()

    models = {
        "logistic":      LogisticRegression(
                            C=1.0, max_iter=1000,
                            class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
                            n_estimators=200, max_depth=15,
                            class_weight="balanced", n_jobs=-1, random_state=42),
        "xgb":           XGBClassifier(
                            n_estimators=300, learning_rate=0.05,
                            eval_metric="logloss", n_jobs=-1, random_state=42,
                            verbosity=0),
        "svm":           SVC(
                            kernel="rbf", C=10, gamma="scale",
                            class_weight="balanced", probability=True,
                            random_state=42),
        "mlp":           MLPClassifier(
                            hidden_layer_sizes=(128, 64, 32),
                            learning_rate_init=0.001, max_iter=300,
                            early_stopping=True, random_state=42),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} ...", flush=True)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / f"{name}.pkl")

        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )

        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            "accuracy":         float(accuracy_score(y_test, y_pred)),
            "precision":        float(precision_score(y_test, y_pred, zero_division=0)),
            "recall":           float(recall_score(y_test, y_pred, zero_division=0)),
            "f1":               float(f1_score(y_test, y_pred, zero_division=0)),
            "auroc":            float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix": cm.tolist(),
        }
        results[name] = metrics
        pd.DataFrame({"y_true": y_test, "y_prob": y_prob}).to_parquet(
            LOG_DIR / f"preds_{name}.parquet", index=False
        )
        print(f"  {name:<15s}  F1={metrics['f1']:.4f}  AUROC={metrics['auroc']:.4f}")

    with open(LOG_DIR / "metrics_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll metrics saved -> {LOG_DIR / 'metrics_test.json'}")

    print("\n" + "="*70)
    print(f"{'Model':<20} {'Accuracy':>9} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>7} {'AUROC':>7}")
    print("-"*70)
    for name, m in results.items():
        print(f"{name:<20} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>7.4f} {m['auroc']:>7.4f}")
    print("="*70)


if __name__ == "__main__":
    fit_and_eval()
