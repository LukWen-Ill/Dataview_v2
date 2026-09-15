"""Träning, utvärdering och persistens av churn-modellen."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS, validate, prepare_features
from src.features import build_preprocessor

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass(frozen=True)
class Metrics:
    """Utvärdering på testmängden."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    n_train: int
    n_test: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def build_pipeline(classifier=None) -> Pipeline:
    """Förbehandling + logistisk regression i en enda pipeline."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                    classifier or LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, Metrics]:
    """Träna på en stratifierad split och utvärdera på hållet testset."""
    if len(X) != len(y):
        raise ValueError(f"X och y har olika längd: {len(X)} vs {len(y)}")
    if y.nunique() < 2:
        raise ValueError("Targeten har bara en klass - modellen går inte att träna.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = Metrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, y_proba)),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
    )
    return pipeline, metrics


def predict_proba(pipeline: Pipeline, X: pd.DataFrame) -> pd.Series:
    """Sannolikhet för churn."""
    proba = pipeline.predict_proba(prepare_features(X))[:, 1]
    return pd.Series(proba, index=X.index, name="churn_probability")


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
