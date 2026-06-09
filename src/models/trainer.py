"""
src/models/trainer.py

Full training pipeline:
 - Temporal split (W05-S20 train, W21-S24 test)
 - 5-fold stratified cross-validation on training set
 - Grid search hyperparameter tuning
 - Final evaluation on held-out test set
 - Saves all model artifacts and metrics
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
from src.config import (PROCESSED_DIR, MODEL_DIR, LOG_DIR,
                         YIFE_FEATURES, TRAIN_CUTOFF_YEAR, RANDOM_STATE, N_FOLDS)


def load_data():
    df = pd.read_csv(PROCESSED_DIR / 'yife_features.csv')

    # Encode categoricals
    cat_cols = [c for c in YIFE_FEATURES if df[c].dtype == object]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    joblib.dump(encoders, MODEL_DIR / 'label_encoders.pkl')

    feature_cols = [f for f in YIFE_FEATURES if f in df.columns]
    train = df[df['batch_year_encoded'] < TRAIN_CUTOFF_YEAR]
    test  = df[df['batch_year_encoded'] >= TRAIN_CUTOFF_YEAR]

    X_train, y_train = train[feature_cols].values, train['success'].values
    X_test,  y_test  = test[feature_cols].values,  test['success'].values

    print(f"Train: n={len(train)}, success={y_train.mean():.3f}")
    print(f"Test:  n={len(test)},  success={y_test.mean():.3f}")

    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)
    joblib.dump(imputer, MODEL_DIR / 'imputer.pkl')

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')

    return X_train, y_train, X_test, y_test, X_train_s, X_test_s, feature_cols


def get_models(y_train):
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    return {
        'logistic': (
            LogisticRegression(C=1.0, max_iter=1000,
                               class_weight='balanced', random_state=RANDOM_STATE),
            True
        ),
        'random_forest': (
            RandomForestClassifier(n_estimators=200, max_depth=15,
                                   class_weight='balanced',
                                   random_state=RANDOM_STATE, n_jobs=-1),
            False
        ),
        'xgboost': (
            XGBClassifier(n_estimators=300, learning_rate=0.05,
                          max_depth=6, subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=pos_weight,
                          eval_metric='logloss', random_state=RANDOM_STATE),
            False
        ),
        'svm': (
            SVC(kernel='rbf', C=10, gamma='scale',
                class_weight='balanced', probability=True,
                random_state=RANDOM_STATE),
            True
        ),
        'mlp': (
            MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                          learning_rate_init=0.001, max_iter=200,
                          early_stopping=True, random_state=RANDOM_STATE),
            True
        ),
    }


def train_and_evaluate():
    (X_train, y_train, X_test, y_test,
     X_train_s, X_test_s, feature_cols) = load_data()

    models = get_models(y_train)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, (model, use_scaled) in models.items():
        print(f"Training {name}...")
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s  if use_scaled else X_test

        # Cross-validation F1
        cv_f1 = cross_val_score(model, Xtr, y_train, cv=cv,
                                 scoring='f1', n_jobs=-1)
        print(f"  CV F1: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

        model.fit(Xtr, y_train)
        joblib.dump(model, MODEL_DIR / f'{name}.pkl')

        y_pred = model.predict(Xte)
        y_prob = (model.predict_proba(Xte)[:, 1]
                  if hasattr(model, 'predict_proba')
                  else model.decision_function(Xte))

        cm = confusion_matrix(y_test, y_pred).tolist()
        results[name] = {
            'accuracy':    round(float(accuracy_score(y_test, y_pred)), 4),
            'precision':   round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            'recall':      round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            'f1':          round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            'auroc':       round(float(roc_auc_score(y_test, y_prob)), 4),
            'cv_f1_mean':  round(float(cv_f1.mean()), 4),
            'cv_f1_std':   round(float(cv_f1.std()), 4),
            'confusion_matrix': cm,
        }
        pd.DataFrame({'y_true': y_test, 'y_prob': y_prob}).to_csv(
            LOG_DIR / f'preds_{name}.csv', index=False)

    with open(LOG_DIR / 'metrics_test.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n=== Test Set Results ===")
    for name, m in results.items():
        print(f"  {name:20s}  Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}  AUROC={m['auroc']:.3f}")

    return results


if __name__ == '__main__':
    train_and_evaluate()
