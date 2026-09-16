"""Inläsning, schemavalidering, rensning och feature engineering av churn-datasettet.

Flödet för en rad kunddata är alltid detsamma, oavsett om den kommer från CSV:n,
databasen eller ett formulär i appen:

    rådata -> validate() -> clean() -> add_features() -> FEATURE_COLUMNS -> modell
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
TARGET_MAP = {"No": 0, "Yes": 1}

# --- Kolumner som finns i rådatan (CSV:n och tabellen customers i databasen) ---

RAW_NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
RAW_CATEGORICAL_COLUMNS = [
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
RAW_FEATURE_COLUMNS = RAW_NUMERIC_COLUMNS + RAW_CATEGORICAL_COLUMNS
REQUIRED_COLUMNS = [ID_COLUMN, *RAW_FEATURE_COLUMNS, TARGET_COLUMN]

# --- Feature engineering: kolumner som skapas i add_features() ---

# De sex tilläggstjänsterna. En kund med många tjänster är mer "inlåst" och
# borde vara mindre benägen att lämna - det vill vi låta modellen se direkt.
ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
TENURE_BINS = [-1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49+"]

ENGINEERED_NUMERIC_COLUMNS = ["num_addon_services", "avg_monthly_charge"]
ENGINEERED_CATEGORICAL_COLUMNS = ["tenure_group"]

# --- Det modellen faktiskt tränas på: rådata + härledda kolumner ---

NUMERIC_COLUMNS = RAW_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS
CATEGORICAL_COLUMNS = RAW_CATEGORICAL_COLUMNS + ENGINEERED_CATEGORICAL_COLUMNS
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


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

    required = REQUIRED_COLUMNS if require_target else RAW_FEATURE_COLUMNS
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
    """Gör de numeriska råkolumnerna numeriska. Lämnar NaN till imputern i pipelinen.

    I datasettet har 11 kunder en tom sträng i TotalCharges (alla med tenure = 0).
    """
    out = df.copy()
    for col in RAW_NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lägg till härledda kolumner. Förutsätter rensad data (clean()).

    - num_addon_services: antal tilläggstjänster kunden har (0-6).
    - avg_monthly_charge: TotalCharges / tenure, alltså vad kunden i snitt betalat
      per månad över hela kundtiden. Skiljer den sig från MonthlyCharges har priset
      ändrats. NaN för nya kunder (tenure = 0) - imputeras med medianen.
    - tenure_group: kundtiden i grupper. Gör det lätt att se i EDA:n att churn
      är kraftigt koncentrerad till första året.
    """
    out = df.copy()

    has_service = out[ADDON_SERVICES].eq("Yes")
    out["num_addon_services"] = has_service.sum(axis=1)

    tenure = out["tenure"]
    out["avg_monthly_charge"] = (out["TotalCharges"] / tenure).where(tenure > 0)

    out["tenure_group"] = pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS).astype(object)
    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hela vägen från validerad rådata till det som matas in i modellen."""
    return add_features(clean(df))[FEATURE_COLUMNS]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Dela upp validerad rådata i features (X) och binär target (y)."""
    y = df[TARGET_COLUMN].map(TARGET_MAP).astype(int)
    return prepare_features(df), y


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Hela vägen från CSV-fil till (X, y)."""
    df = load_raw(path)
    validate(df)
    return split_features_target(df)
