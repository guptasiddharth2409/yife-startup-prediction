"""
src/config.py
Central path configuration for the YIFE project.
"""
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[1]
DATA_DIR     = ROOT / 'data'
RAW_DIR      = DATA_DIR / 'raw'
PROCESSED_DIR= DATA_DIR / 'processed'
MODEL_DIR    = ROOT / 'models'
FIG_DIR      = ROOT / 'figures'
LOG_DIR      = ROOT / 'logs'
PAPER_DIR    = ROOT / 'paper'

# Create all directories on import
for _p in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, FIG_DIR, LOG_DIR, PAPER_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# Model hyperparameters (paper Table 3)
MODEL_PARAMS = {
    'logistic': dict(C=1.0, max_iter=1000, class_weight='balanced', random_state=42),
    'random_forest': dict(n_estimators=200, max_depth=15,
                         class_weight='balanced', random_state=42, n_jobs=-1),
    'xgboost': dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric='logloss', random_state=42),
    'svm': dict(kernel='rbf', C=10, gamma='scale',
                class_weight='balanced', probability=True, random_state=42),
    'mlp': dict(hidden_layer_sizes=(128, 64, 32), learning_rate_init=0.001,
                max_iter=200, early_stopping=True, random_state=42),
}

# Temporal split (paper Section 4.2)
TRAIN_CUTOFF_YEAR = 2021  # batches W05-S20 train, W21-S24 test
RANDOM_STATE = 42
N_FOLDS = 5

# YIFE 14 core features (paper Table 2)
YIFE_FEATURES = [
    'total_funding_usd', 'num_funding_rounds', 'seed_round_size',
    'team_size', 'faang_experience', 'elite_edu',
    'github_repo_count', 'github_commit_freq',
    'batch_year_encoded', 'batch_size',
    'industry_category', 'ai_flag',
    'geo_cluster', 'tier1_vc_investor',
]
