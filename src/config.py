"""Global path configuration for the YIFE project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR      = ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR     = ROOT / "models"
FIG_DIR       = ROOT / "figures"
LOG_DIR       = ROOT / "logs"

for _p in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, FIG_DIR, LOG_DIR]:
    _p.mkdir(parents=True, exist_ok=True)
