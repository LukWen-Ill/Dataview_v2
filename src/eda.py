"""Små hjälpfunktioner för EDA-sidan i appen. Bara aggregering, ingen modellering."""

from __future__ import annotations

import pandas as pd

from src.data import ID_COLUMN, NUMERIC_COLUMNS, TARGET_COLUMN, add_features, clean

# Kategoriska kolumner som är intressanta att titta på churn-andel för.
EDA_CATEGORIES = [
    "Contract",
    "tenure_group",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "OnlineSecurity",
    "PaperlessBilling",
    "SeniorCitizen",
    "Partner",
    "Dependents",
]

# Prisintervall för MonthlyCharges (18-119 i datan). Gränserna följer tjänstetyperna:
# enbart telefoni under 30, DSL 30-70, fiber 70-110, fiber med många tillägg över 110.
PRICE_BINS = [0, 30, 50, 70, 90, 110, 130]
PRICE_LABELS = ["18-30", "30-50", "50-70", "70-90", "90-110", "110-130"]


def with_churn_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Rensad data med härledda kolumner och en numerisk churn-kolumn (0/1)."""
    out = add_features(clean(df))
    out["churn"] = (df[TARGET_COLUMN] == "Yes").astype(int)
    return out


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn-andel och antal kunder per kategori i en kolumn."""
    data = with_churn_flag(df)  # härledda kolumner som tenure_group finns först efter detta
    if column not in data.columns:
        raise KeyError(f"Kolumnen {column!r} finns inte i datan")
    grouped = data.groupby(column, observed=True)["churn"].agg(["mean", "size"])
    grouped.columns = ["churn_andel", "antal_kunder"]
    return grouped.reset_index().sort_values("churn_andel", ascending=False)


def churn_correlations(df: pd.DataFrame) -> pd.Series:
    """Korrelation mellan varje numerisk kolumn och churn (0/1). Mäter linjärt samband."""
    data = with_churn_flag(df)
    return data[NUMERIC_COLUMNS].corrwith(data["churn"]).sort_values()


def churn_rate_by_price_band(df: pd.DataFrame) -> pd.DataFrame:
    """Churn-andel och antal kunder per prisintervall av MonthlyCharges.

    Visar att churn inte följer priset rakt av utan går upp och ner mellan intervallen.
    """
    if "MonthlyCharges" not in df.columns:
        raise KeyError("Kolumnen 'MonthlyCharges' finns inte i datan")
    data = with_churn_flag(df)
    band = pd.cut(data["MonthlyCharges"], bins=PRICE_BINS, labels=PRICE_LABELS)
    grouped = data.groupby(band, observed=True)["churn"].agg(["mean", "size"])
    grouped.columns = ["churn_andel", "antal_kunder"]
    grouped.index.name = "prisintervall"
    return grouped.reset_index()


def data_quality(df: pd.DataFrame) -> dict[str, int]:
    """Enkla datakontroller: dubbletter, unika kund-id och tomma TotalCharges."""
    for col in (ID_COLUMN, "TotalCharges"):
        if col not in df.columns:
            raise KeyError(f"Kolumnen {col!r} finns inte i datan")
    total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return {
        "kunder": len(df),
        "unika_id": int(df[ID_COLUMN].nunique()),
        "dubbletter": int(df.duplicated().sum()),
        "tomma_totalcharges": int(total_charges.isna().sum()),
    }
