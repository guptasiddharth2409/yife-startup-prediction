"""
train_models.py — Train all 5 YIFE classifiers and save to models/.

Models: Logistic Regression, SVM, Random Forest, XGBoost, MLP.
All hyperparameters read from configs/model_config.yaml.

Usage:
    python src/train_models.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from utils import load_config, set_seed, ensure_dir, print_banner


class MLP:
    """Lightweight PyTorch MLP wrapper with sklearn-like API."""

    def __init__(self, input_dim, hidden_sizes, dropout, lr, epochs, batch_size, patience, weight_decay, seed):
        import torch
        import torch.nn as nn

        set_seed(seed)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self._torch = torch

    def fit(self, X, y):
        import torch
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        best_loss, patience_count = float("inf"), 0
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(X_t))
            for i in range(0, len(X_t), self.batch_size):
                idx = perm[i:i + self.batch_size]
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X_t[idx]), y_t[idx])
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = self.criterion(self.model(X_t), y_t).item()
            if val_loss < best_loss:
                best_loss = val_loss
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    print(f"    Early stopping at epoch {epoch + 1}")
                    break
        return self

    def predict_proba(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_t).cpu().numpy().flatten()
        proba = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def load_features(config: dict):
    """Load processed features and split into train/test."""
    path = "data/processed/yife_features.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed features not found at {path}.\n"
            "Run: python src/feature_engineering.py"
        )

    df = pd.read_csv(path)
    target = config["global"]["target_column"]
    numeric_feats = config["preprocessing"]["numeric_features"]

    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].values
    y = df[target].values

    scaler = StandardScaler()
    num_idx = [feature_cols.index(f) for f in numeric_feats if f in feature_cols]
    X[:, num_idx] = scaler.fit_transform(X[:, num_idx])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["global"]["test_size"],
        random_state=config["global"]["seed"],
        stratify=y
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,} | Positive rate: {y.mean():.1%}")
    return X_train, X_test, y_train, y_test, scaler, feature_cols


def build_models(config: dict, input_dim: int) -> dict:
    """Instantiate all models from config."""
    cfg = config["models"]
    seed = config["global"]["seed"]

    lr_cfg = cfg["logistic_regression"]
    svm_cfg = cfg["svm"]
    rf_cfg = cfg["random_forest"]
    xgb_cfg = cfg["xgboost"]
    mlp_cfg = cfg["mlp"]

    return {
        "Logistic Regression": LogisticRegression(
            C=lr_cfg["C"], max_iter=lr_cfg["max_iter"],
            solver=lr_cfg["solver"], class_weight=lr_cfg["class_weight"],
            random_state=seed
        ),
        "SVM": SVC(
            C=svm_cfg["C"], kernel=svm_cfg["kernel"], gamma=svm_cfg["gamma"],
            probability=svm_cfg["probability"], class_weight=svm_cfg["class_weight"],
            random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"], max_depth=rf_cfg["max_depth"],
            min_samples_split=rf_cfg["min_samples_split"],
            min_samples_leaf=rf_cfg["min_samples_leaf"],
            class_weight=rf_cfg["class_weight"], random_state=seed,
            n_jobs=rf_cfg["n_jobs"]
        ),
        "XGBoost": XGBClassifier(
            n_estimators=xgb_cfg["n_estimators"], max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"], subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            scale_pos_weight=xgb_cfg["scale_pos_weight"],
            eval_metric=xgb_cfg["eval_metric"], random_state=seed,
            n_jobs=xgb_cfg["n_jobs"], verbosity=0
        ),
        "MLP": MLP(
            input_dim=input_dim,
            hidden_sizes=mlp_cfg["hidden_sizes"],
            dropout=mlp_cfg["dropout"],
            lr=mlp_cfg["learning_rate"],
            epochs=mlp_cfg["epochs"],
            batch_size=mlp_cfg["batch_size"],
            patience=mlp_cfg["patience"],
            weight_decay=mlp_cfg["weight_decay"],
            seed=seed
        ),
    }


def run(config_path: str = "configs/model_config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["global"]["seed"])
    print_banner("Training YIFE Models")

    X_train, X_test, y_train, y_test, scaler, feature_cols = load_features(cfg)
    models = build_models(cfg, input_dim=X_train.shape[1])
    ensure_dir("models")

    trained = {}
    cv = StratifiedKFold(n_splits=cfg["global"]["cv_folds"], shuffle=True, random_state=cfg["global"]["seed"])

    for name, model in models.items():
        print(f"\n  [{name}]")
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
            print(f"    CV AUROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        trained[name] = model
        safe_name = name.lower().replace(" ", "_")
        with open(f"models/{safe_name}.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"    Saved → models/{safe_name}.pkl")

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("\n  ✓ All models saved to models/")

    # Save test split for evaluate.py
    np.save("models/X_test.npy", X_test)
    np.save("models/y_test.npy", y_test)
    import json
    with open("models/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    return trained, X_test, y_test


if __name__ == "__main__":
    run()
