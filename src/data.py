"""Inläsning, schemavalidering och rensning av churn-datasettet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
TARGET_MAP = {"No": 0, "Yes": 1}

NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
REQUIRED_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]


class SchemaError(ValueError):
    """Datat matchar inte det schema modellen är byggd för."""


def load_raw(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Läs rå-CSV:n. Kastar FileNotFoundError om filen saknas."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Datafilen saknas: {path}")
    return pd.read_csv(path)


def validate(df: pd.DataFrame, *, require_target: bool = True) -> None:
    """Kasta SchemaError om dataramen inte går att träna eller prediktera på."""
    if df.empty:
        raise SchemaError("Datasettet är tomt.")

    required = REQUIRED_COLUMNS if require_target else FEATURE_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SchemaError(f"Kolumner saknas: {sorted(missing)}")

    if require_target:
        unexpected = sorted(set(df[TARGET_COLUMN].dropna().unique()) - set(TARGET_MAP))
        if unexpected:
            raise SchemaError(
                f"Ogiltiga värden i {TARGET_COLUMN}: {unexpected}. Tillåtna: {sorted(TARGET_MAP)}"
            )
        if df[TARGET_COLUMN].isna().any():
            raise SchemaError(f"{TARGET_COLUMN} innehåller saknade värden.")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Gör kolumnerna numeriska där de ska vara det. Lämnar NaN till imputern."""
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enda vägen från rå dataram till modellindata."""
    validate(df, require_target=False)
    return clean(df)[FEATURE_COLUMNS]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COLUMN].map(TARGET_MAP).astype(int)
    return prepare_features(df), y

def load_dataset(path=DEFAULT_DATA_PATH):
    df = load_raw(path)
    validate(df)                              # targeten kollas här
    y = df[TARGET_COLUMN].map(TARGET_MAP).astype(int)
    return prepare_features(df), y
