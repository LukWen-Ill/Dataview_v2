"""Spara, ladda och prediktera med den tränade modellen (joblib)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import prepare_features, validate

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "churn_model.joblib"


def predict_proba(pipeline: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Sannolikhet för churn per rad. Tar rådata (samma kolumner som CSV:n, utan Churn)."""
    validate(df, require_target=False)
    X = prepare_features(df)
    proba = pipeline.predict_proba(X)[:, 1]
    return pd.Series(proba, index=df.index, name="churn_probability")


def save(pipeline: Pipeline, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load(path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ingen tränad modell på {path}. Kör: python -m src.train")
    return joblib.load(path)
