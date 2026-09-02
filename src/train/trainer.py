"""YIFE reference training and evaluation pipeline.

The published protocol uses:
  1. Temporal split: W05-S20 train, W21-S24 untouched test.
  2. 3-fold CV grid search for hyperparameter tuning on training data only.
  3. 5-fold stratified CV on the training data for model selection/comparison.
  4. One final evaluation on the untouched W21-S24 cohort.

The synthetic dataset bundled with this repository is for pipeline demonstration
only; it is not the original research dataset and will not reproduce the paper's
reported metrics exactly.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import LOG_DIR, MODEL_DIR, PROCESSED_DIR

SEED = 42
TARGET = "success"
DROP = {"company", "success", "batch"}
CATEGORICAL = ["industry_category", "geo_cluster"]
NUMERIC = [
    "total_funding_usd", "num_funding_rounds", "seed_round_size", "team_size",
    "faang_experience", "elite_edu", "github_repo_count", "github_commit_freq",
    "batch_year_encoded", "batch_size", "ai_flag", "tier1_vc_investor",
]


def load_data(path=None):
    path = path or (PROCESSED_DIR / "yife_features.parquet")
    df = pd.read_parquet(path).copy()

    if TARGET not in df.columns:
        raise ValueError(f"Required target column '{TARGET}' not found in {path}")
    if "batch" not in df.columns and "batch_year_encoded" not in df.columns:
        raise ValueError("Temporal split requires 'batch' or 'batch_year_encoded'.")

    # Use the actual batch year rather than a label-encoded categorical value.
    if "batch_year_encoded" in df.columns:
        years = pd.to_numeric(df["batch_year_encoded"], errors="coerce")
    else:
        years = df["batch"].astype(str).str.extract(r"(\d{2,4})")[0].astype(float)
        years = np.where(years < 100, years + 2000, years)
        years = pd.Series(years, index=df.index)

    df["_split_year"] = years
    if df["_split_year"].isna().any():
        raise ValueError("Some rows have an invalid batch year; refusing an ambiguous temporal split.")

    train = df[df["_split_year"] < 2021].copy()
    test = df[df["_split_year"] >= 2021].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty train or test set.")

    available_numeric = [c for c in NUMERIC if c in df.columns]
    available_categorical = [c for c in CATEGORICAL if c in df.columns]
    feature_cols = available_numeric + available_categorical

    X_train = train[feature_cols]
    y_train = train[TARGET].astype(int)
    X_test = test[feature_cols]
    y_test = test[TARGET].astype(int)

    print(f"Train: {len(train):,} | Test: {len(test):,}")
    print(f"Train success rate: {y_train.mean():.3f} | Test success rate: {y_test.mean():.3f}")
    return X_train, y_train, X_test, y_test, feature_cols


def make_pipeline(model, scale_numeric=False, numeric_features=None, categorical_features=None):
    numeric_features = numeric_features or NUMERIC
    categorical_features = categorical_features or CATEGORICAL
    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            *([("scaler", StandardScaler())] if scale_numeric else []),
        ]), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def model_searches(y_train):
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    return {
        "logistic": (
            make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED), True),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            make_pipeline(RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1), False),
            {"model__n_estimators": [200, 300], "model__max_depth": [8, 15], "model__min_samples_split": [2, 5]},
        ),
        "xgboost": (
            make_pipeline(XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1, verbosity=0), False),
            {"model__n_estimators": [300], "model__max_depth": [4, 6], "model__learning_rate": [0.05],
             "model__subsample": [0.8], "model__colsample_bytree": [0.8],
             "model__scale_pos_weight": [pos_weight]},
        ),
        "svm": (
            make_pipeline(SVC(class_weight="balanced", probability=True, random_state=SEED), True),
            {"model__C": [1.0, 10.0], "model__kernel": ["rbf"], "model__gamma": ["scale"]},
        ),
        "mlp": (
            make_pipeline(MLPClassifier(max_iter=300, early_stopping=True, random_state=SEED), True),
            {"model__hidden_layer_sizes": [(128, 64, 32)], "model__learning_rate_init": [0.001],
             "model__alpha": [0.0001], "model__max_iter": [300]},
        ),
    }


def evaluate(name, estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    y_prob = estimator.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "auroc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": cm.tolist(),
    }
    pd.DataFrame({"y_true": y_test, "y_prob": y_prob}).to_parquet(
        LOG_DIR / f"preds_{name}.parquet", index=False
    )
    return metrics


def main():
    X_train, y_train, X_test, y_test, feature_cols = load_data()
    tuning_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    selection_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results = {}
    searches = model_searches(y_train)
    for name, (pipeline, grid) in searches.items():
        print(f"\nTuning {name} with 3-fold CV ...", flush=True)
        search = GridSearchCV(pipeline, grid, scoring="f1", cv=tuning_cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        cv_f1 = float(cross_val_score(best, X_train, y_train, scoring="f1", cv=selection_cv, n_jobs=-1).mean())
        print(f"  best params: {search.best_params_}")
        print(f"  5-fold training CV F1: {cv_f1:.4f}")

        best.fit(X_train, y_train)
        joblib.dump(best, MODEL_DIR / f"{name}.pkl")
        results[name] = evaluate(name, best, X_test, y_test)
        results[name]["training_cv_f1"] = round(cv_f1, 4)
        results[name]["best_params"] = search.best_params_

    with open(LOG_DIR / "metrics_test.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nFinal held-out W21-S24 results:")
    print(pd.DataFrame({k: {m: v for m, v in r.items() if m in {"accuracy", "precision", "recall", "f1", "auroc"}} for k, r in results.items()}).T)


if __name__ == "__main__":
    main()
