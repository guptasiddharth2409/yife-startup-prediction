"""
02_train_model.py

Loads data/processed/yife_features.csv, applies temporal split (train: W05-S20,
test: W21-S24), trains all 5 YIFE classifiers, saves models and test metrics.

Models: Logistic Regression, Random Forest, XGBoost, SVM, MLP
Metrics: Accuracy, Precision, Recall, F1, AUROC
Outputs:
  models/   -> <model>.pkl files
  logs/     -> metrics_test.json, preds_<model>.csv
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier

MODEL_DIR = Path("models")
LOG_DIR = Path("logs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_and_split(csv_path="data/processed/yife_features.csv"):
    df = pd.read_csv(csv_path)

    # Encode categoricals
    for col in ['industry_category', 'geo_cluster']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    feature_cols = [
        'total_funding_usd', 'num_funding_rounds', 'seed_round_size',
        'team_size', 'faang_experience', 'elite_edu',
        'github_repo_count', 'github_commit_freq',
        'batch_year_encoded', 'batch_size',
        'industry_category', 'ai_flag', 'geo_cluster', 'tier1_vc_investor'
    ]

    # Temporal split: train = pre-2021, test = 2021+
    train_df = df[df['batch_year_encoded'] < 2021].copy()
    test_df  = df[df['batch_year_encoded'] >= 2021].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df['success'].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df['success'].values

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Train success rate: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

    # Impute missing (GitHub features are ~29% NaN)
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)
    joblib.dump(imputer, MODEL_DIR / 'imputer.pkl')

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')

    return X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, feature_cols


def evaluate(name, model, X_test, y_test, use_proba=True):
    y_pred = model.predict(X_test)
    y_prob = (model.predict_proba(X_test)[:, 1]
              if use_proba and hasattr(model, 'predict_proba')
              else model.decision_function(X_test))
    metrics = {
        'accuracy':  round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1':        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'auroc':     round(float(roc_auc_score(y_test, y_prob)), 4),
    }
    print(f"  {name:25s}  F1={metrics['f1']:.3f}  AUROC={metrics['auroc']:.3f}")
    preds_df = pd.DataFrame({'y_true': y_test, 'y_prob': y_prob})
    preds_df.to_csv(LOG_DIR / f'preds_{name}.csv', index=False)
    return metrics


def main():
    (X_train, y_train, X_test, y_test,
     X_train_s, X_test_s, feature_cols) = load_and_split()

    # --- Models (paper config) ---
    models = {
        'logistic': (
            LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42),
            True   # needs scaled data
        ),
        'random_forest': (
            RandomForestClassifier(n_estimators=200, max_depth=15,
                                   class_weight='balanced', random_state=42, n_jobs=-1),
            False
        ),
        'xgboost': (
            XGBClassifier(n_estimators=300, learning_rate=0.05,
                          max_depth=6, subsample=0.8, colsample_bytree=0.8,
                          eval_metric='logloss', random_state=42,
                          scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()),
            False
        ),
        'svm': (
            SVC(kernel='rbf', C=10, gamma='scale',
                class_weight='balanced', probability=True, random_state=42),
            True
        ),
        'mlp': (
            MLPClassifier(hidden_layer_sizes=(128, 64, 32), learning_rate_init=0.001,
                          max_iter=200, early_stopping=True, random_state=42),
            True
        ),
    }

    results = {}
    print("\nTraining models...")
    for name, (model, use_scaled) in models.items():
        X_tr = X_train_s if use_scaled else X_train
        X_te = X_test_s  if use_scaled else X_test
        model.fit(X_tr, y_train)
        joblib.dump(model, MODEL_DIR / f'{name}.pkl')
        results[name] = evaluate(name, model, X_te, y_test)

    # Save metrics
    with open(LOG_DIR / 'metrics_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {LOG_DIR / 'metrics_test.json'}")
    print("\nResults Summary:")
    print(pd.DataFrame(results).T.to_string())


if __name__ == '__main__':
    main()
