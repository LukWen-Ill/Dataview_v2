"""Arbetslista (extra) för Säljverktyg-sidans flik Befintlig kund.

Alla kunder ur båda registren med risknivå, pris och några nyckelfält, sorterade så att
högst risk ligger överst, plus en enkel sökning på kund-id. Sannolikheterna kommer från
predict_proba i src/model.py och räknas inte här. Ingen Streamlit här.
"""

from __future__ import annotations

import pandas as pd

from src.data import ID_COLUMN
from src.risk import HIGH, LOW, MEDIUM, risk_level

OVERVIEW_COLUMNS = [
    ID_COLUMN,
    "level",
    "MonthlyCharges",
    "tenure",
    "Contract",
    "InternetService",
    "PaymentMethod",
]
LEVEL_ORDER = {HIGH: 0, MEDIUM: 1, LOW: 2}


def customer_overview(
    customers: pd.DataFrame, probabilities: pd.Series, thresholds: tuple[float, float]
) -> pd.DataFrame:
    """En rad per kund med råvärden (sidan sätter etiketter), sorterad hög -> medel -> låg
    och inom nivån på sannolikhet fallande.

    probabilities är predict_proba över customers. ValueError om den inte matchar
    customers på längd eller index.
    """
    if len(probabilities) != len(customers) or not probabilities.index.equals(customers.index):
        raise ValueError(
            f"probabilities ({len(probabilities)} rader) måste ha samma längd och index "
            f"som customers ({len(customers)} rader)."
        )
    overview = customers[[col for col in OVERVIEW_COLUMNS if col != "level"]].copy()
    overview["level"] = [risk_level(p, thresholds) for p in probabilities]
    overview["_order"] = overview["level"].map(LEVEL_ORDER)
    overview["_probability"] = probabilities.to_numpy()
    overview = overview.sort_values(["_order", "_probability"], ascending=[True, False])
    return overview[OVERVIEW_COLUMNS].reset_index(drop=True)


def search_customers(overview: pd.DataFrame, query: str) -> pd.DataFrame:
    """Filtrera arbetslistan på kund-id.

    Tom query ger allt. Rena siffror matchar början av id:t ("7590" -> 7590-VHVEG).
    Annan text matchar var som helst i id:t, skiftlägesokänsligt ("new" -> NEW-000001).
    """
    query = query.strip()
    if not query:
        return overview
    ids = overview[ID_COLUMN].astype(str)
    if query.isdigit():
        mask = ids.str.startswith(query)
    else:
        mask = ids.str.lower().str.contains(query.lower(), regex=False)
    return overview[mask]
