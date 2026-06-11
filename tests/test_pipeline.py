"""Lightweight smoke test — no full model training in CI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_all_modules_importable():
    import src.config
    import src.data.generate_synthetic
    import src.train.trainer
    import src.visualization.roc_shap


def test_parquet_output_schema():
    from src.data.generate_synthetic import generate
    from src.config import PROCESSED_DIR
    import pandas as pd

    df = generate(seed=42)
    parquet_path = PROCESSED_DIR / "yife_features.parquet"
    assert parquet_path.exists()
    df_loaded = pd.read_parquet(parquet_path)
    assert len(df_loaded) == 4323
    assert "success" in df_loaded.columns
    assert df_loaded["success"].isin([0, 1]).all()
