"""Delade fixtures. Syntetisk data så testerna inte beror på rå-CSV:n."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data import (
    DEFAULT_DATA_PATH,
    ID_COLUMN,
    RAW_CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    split_features_target,
)
from src.models import build_pipeline

CATEGORY_VALUES = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def make_raw_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Giltig rådata i samma form som CSV:n, båda klasserna, TotalCharges som text."""
    rng = np.random.default_rng(seed)
    data = {
        ID_COLUMN: [f"{i:04d}-TEST" for i in range(n)],
        "SeniorCitizen": rng.integers(0, 2, n),
        "tenure": rng.integers(0, 72, n),
        "MonthlyCharges": rng.uniform(18.0, 119.0, n).round(2),
    }
    for col in RAW_CATEGORICAL_COLUMNS:
        data[col] = rng.choice(CATEGORY_VALUES[col], n)
    data["TotalCharges"] = [f"{v:.2f}" for v in data["tenure"] * data["MonthlyCharges"]]
    # Balanserad target så stratifierad split fungerar på den lilla mängden.
    data[TARGET_COLUMN] = ["Yes" if i % 2 else "No" for i in range(n)]
    return pd.DataFrame(data)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_df()


@pytest.fixture
def fitted_pipeline(raw_df):
    """En snabbt tränad pipeline (utan GridSearch) för prediktions- och utvärderingstester."""
    X, y = split_features_target(raw_df)
    pipeline = build_pipeline(LogisticRegression(max_iter=2000, random_state=0))
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def tiny_candidates():
    """En modell med ett minimalt grid så att hela träningsflödet kan testas på sekunder."""
    pipeline = build_pipeline(LogisticRegression(max_iter=2000, random_state=0))
    return {"Logistisk regression": (pipeline, {"classifier__C": [0.1, 1.0]})}


@pytest.fixture
def db_path(tmp_path, raw_df):
    """En färdig databas i en tillfällig mapp, byggd från syntetisk rådata."""
    from src.db import init_db

    csv_path = tmp_path / "raw.csv"
    raw_df.to_csv(csv_path, index=False)
    return init_db(csv_path, tmp_path / "test.db")


@pytest.fixture
def real_data_path():
    """Sökväg till rå-CSV:n, hoppar inte över tyst utan failar om filen saknas."""
    assert DEFAULT_DATA_PATH.is_file(), f"Rådatan saknas: {DEFAULT_DATA_PATH}"
    return DEFAULT_DATA_PATH
