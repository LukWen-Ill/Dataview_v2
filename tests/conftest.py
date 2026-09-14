"""Delade fixtures. Syntetisk data så testerna inte beror på rå-CSV:n."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import CATEGORICAL_COLUMNS, DEFAULT_DATA_PATH, ID_COLUMN, TARGET_COLUMN

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


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Giltig rådata i samma form som CSV:n: 60 rader, båda klasserna, TotalCharges som text."""
    rng = np.random.default_rng(0)
    n = 60
    data = {
        ID_COLUMN: [f"{i:04d}-TEST" for i in range(n)],
        "SeniorCitizen": rng.integers(0, 2, n),
        "tenure": rng.integers(0, 72, n),
        "MonthlyCharges": rng.uniform(18.0, 119.0, n).round(2),
    }
    for col in CATEGORICAL_COLUMNS:
        data[col] = rng.choice(CATEGORY_VALUES[col], n)
    data["TotalCharges"] = [f"{v:.2f}" for v in data["tenure"] * data["MonthlyCharges"]]
    # Balanserad target så stratifierad split fungerar på den lilla mängden.
    data[TARGET_COLUMN] = ["Yes" if i % 2 else "No" for i in range(n)]
    return pd.DataFrame(data)


@pytest.fixture
def real_data_path():
    """Sökväg till rå-CSV:n, hoppar inte över tyst utan failar om filen saknas."""
    assert DEFAULT_DATA_PATH.is_file(), f"Rådatan saknas: {DEFAULT_DATA_PATH}"
    return DEFAULT_DATA_PATH
