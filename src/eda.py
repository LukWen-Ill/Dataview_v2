"""Små hjälpfunktioner för EDA-sidan i appen. Bara aggregering, ingen modellering."""

from __future__ import annotations

import pandas as pd

from src.data import NUMERIC_COLUMNS, TARGET_COLUMN, add_features, clean

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


def with_churn_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Rensad data med härledda kolumner och en numerisk churn-kolumn (0/1)."""
    out = add_features(clean(df))
    out["churn"] = (df[TARGET_COLUMN] == "Yes").astype(int)
    return out


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn-andel och antal kunder per kategori i en kolumn."""
    if column not in df.columns:
        raise KeyError(f"Kolumnen {column!r} finns inte i datan")
    data = with_churn_flag(df)
    grouped = data.groupby(column, observed=True)["churn"].agg(["mean", "size"])
    grouped.columns = ["churn_andel", "antal_kunder"]
    return grouped.reset_index().sort_values("churn_andel", ascending=False)


def churn_correlations(df: pd.DataFrame) -> pd.Series:
    """Korrelation mellan varje numerisk kolumn och churn (0/1). Mäter linjärt samband."""
    data = with_churn_flag(df)
    return data[NUMERIC_COLUMNS].corrwith(data["churn"]).sort_values()
