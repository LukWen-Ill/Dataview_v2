"""Hjälpfunktioner för Ledning-sidan i appen. Bara filtrering och aggregering, ingen modellering.

Modulen har samma roll för Ledning-sidan som src/eda.py har för Data-sidan: sidan skickar in
kundtabellen, får tillbaka färdiga tal och tabeller att visa. Ingen Streamlit-import här –
det gör att funktionerna kan testas utan appen.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.data import TARGET_COLUMN

# Hårdkodade svenska månadsnamn. strftime("%b") beror på systemets locale och ger "Oct" på CI.
MONTH_NAMES_SV = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "Maj",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Okt",
    "Nov",
    "Dec",
]


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


def revenue_forecast(df: pd.DataFrame, months: int = 24, start: date | None = None) -> pd.DataFrame:
    """Förväntad kundstock och månadsintäkt (MRR) m månader framåt, m = 0..months.

    Varje aktiv kund antas vara kvar med sannolikheten (1 - p)^m där p = churn_probability.
    Kunder som redan lämnat (Churn = Yes) räknas inte med. start=None betyder innevarande
    månad; start normaliseras alltid till första dagen i månaden.

    Kolumner ut: months_ahead, month, month_label, expected_customers, expected_mrr,
    expected_loss (tappet från föregående månad, 0 vid m = 0) och cumulative_loss
    (tappet sedan m = 0).
    """
    for column in ("churn_probability", "MonthlyCharges"):
        if column not in df.columns:
            raise KeyError(f"Kolumnen {column!r} finns inte i datan")
    if months < 1:
        raise ValueError(f"months måste vara minst 1, fick {months}")

    active = active_customers(df)
    if active.empty:
        raise ValueError("Inga aktiva kunder att göra prognos på")
    p = active["churn_probability"].to_numpy(dtype=float)
    if ((p < 0) | (p > 1)).any():
        raise ValueError("churn_probability måste ligga i [0, 1]")
    charges = active["MonthlyCharges"].to_numpy(dtype=float)

    if start is None:
        start = date.today()
    first_month = pd.Timestamp(year=start.year, month=start.month, day=1)

    rows = []
    for m in range(months + 1):
        survival = (1 - p) ** m  # andel av varje kund som förväntas vara kvar efter m månader
        month = first_month + pd.DateOffset(months=m)
        rows.append(
            {
                "months_ahead": m,
                "month": month,
                "month_label": f"{MONTH_NAMES_SV[month.month - 1]} {month.year}",
                "expected_customers": survival.sum(),
                "expected_mrr": (charges * survival).sum(),
            }
        )
    out = pd.DataFrame(rows)
    # Tappet en månad är skillnaden i MRR mot månaden före. Första raden har ingen föregående.
    out["expected_loss"] = (out["expected_mrr"].shift(1) - out["expected_mrr"]).fillna(0.0)
    out["cumulative_loss"] = out["expected_mrr"].iloc[0] - out["expected_mrr"]
    return out
