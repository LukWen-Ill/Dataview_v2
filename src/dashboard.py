"""Hjälpfunktioner för Ledning-sidan i appen. Bara filtrering och aggregering, ingen modellering.

Modulen har samma roll för Ledning-sidan som src/eda.py har för Data-sidan: sidan skickar in
kundtabellen, får tillbaka färdiga tal och tabeller att visa. Ingen Streamlit-import här –
det gör att funktionerna kan testas utan appen.
"""

from __future__ import annotations

import pandas as pd

from src.data import TARGET_COLUMN


def filter_customers(df: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    """Filtrerar kundtabellen på kategorikolumner, t.ex. {"Contract": ["Two year"]}.

    Tom dict eller tom lista betyder inget filter (som en tom multiselect i appen).
    Flera kolumner kombineras med OCH, flera värden i samma kolumn med ELLER.
    Okänd kolumn ger KeyError, ett filter som inte träffar någon kund ger ValueError.
    Returnerar en ny dataram så att senare kolumntillägg inte varnar.
    """
    mask = pd.Series(True, index=df.index)
    for column, values in filters.items():
        if column not in df.columns:
            raise KeyError(f"Kolumnen {column!r} finns inte i datan")
        if not values:
            continue
        mask &= df[column].isin(values)  # isin fungerar för både text och heltal
    if not mask.any():
        raise ValueError("Filtret matchar inga kunder")
    return df.loc[mask].copy()


def active_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Kunder som är kvar (Churn = No). De som redan lämnat ska inte rullas fram i prognosen."""
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Kolumnen {TARGET_COLUMN!r} finns inte i datan")
    return df.loc[df[TARGET_COLUMN] == "No"].copy()
