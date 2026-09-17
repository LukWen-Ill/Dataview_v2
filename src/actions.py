"""Åtgärdskatalog (extra) för Säljverktyg-sidan: vilka alternativ passar en kund, och vad
de ger i pris och bedömd risk.

Katalogen kommer från epiken: churn i datasettet utan respektive med åtgärden. Tre sorter:

- "what-if": vi ändrar kundens konfiguration, låter prismodellen sätta nytt pris och
  låter churnmodellen bedöma risken igen. Pris och risk hör alltid ihop: det nya priset
  sätts in som MonthlyCharges innan churnmodellen körs.
- "samband": samma what-if, men datan visar ett samband och inte en orsak. Visas så.
- "regel": en rekommendation utan siffror. Churnmodellen kan inte bedöma en uppföljning.

Ett alternativ ändrar aldrig tenure eller TotalCharges och tar aldrig bort en tjänst.
Ingen Streamlit här - sidan visar bara det som räknas fram här.
"""

from __future__ import annotations

import pandas as pd

from src.data import RAW_CATEGORICAL_COLUMNS
from src.labels import LABELS, VALUE_LABELS
from src.model import predict_proba
from src.price import predict_price
from src.risk import LOW, risk_level

MAX_ACTIONS = 3
KIND_ORDER = {"what-if": 0, "samband": 1, "regel": 2}

# change = kolumn -> nytt värde. Tomt för "regel".
ACTIONS: list[dict] = [
    {
        "key": "one_year",
        "label": "Ettårsavtal",
        "kind": "what-if",
        "change": {"Contract": "One year"},
    },
    {
        "key": "two_year",
        "label": "Tvåårsavtal",
        "kind": "what-if",
        "change": {"Contract": "Two year"},
    },
    {
        "key": "tech_support",
        "label": "Teknisk support",
        "kind": "what-if",
        "change": {"TechSupport": "Yes"},
    },
    {
        "key": "online_security",
        "label": "Onlinesäkerhet",
        "kind": "what-if",
        "change": {"OnlineSecurity": "Yes"},
    },
    {
        "key": "auto_payment",
        "label": "Automatisk banköverföring",
        "kind": "samband",
        "change": {"PaymentMethod": "Bank transfer (automatic)"},
    },
    {
        "key": "follow_up",
        "label": "Uppföljning första året",
        "kind": "regel",
        "change": {},
    },
    {
        "key": "none",
        "label": "Ingen åtgärd",
        "kind": "regel",
        "change": {},
    },
]

NOTES = {
    "auto_payment": "Samband i datan, inte orsak: kunder med automatisk betalning churnar mindre.",
    "follow_up": (
        "Internet på månadsavtal under första året. Regel utan what-if - "
        "modellen kan inte bedöma en uppföljning."
    ),
    "none": "Låg risk. Bevaka, ingen åtgärd behövs.",
}


def get_action(key: str) -> dict:
    """Slå upp en åtgärd i katalogen. KeyError om nyckeln inte finns."""
    for action in ACTIONS:
        if action["key"] == key:
            return action
    known = [action["key"] for action in ACTIONS]
    raise KeyError(f"Okänd åtgärd: {key!r}. Kända: {known}")


def fits(action: dict, customer: pd.DataFrame) -> bool:
    """Passar åtgärden kundens konfiguration? Reglerna kommer från epikens katalog.

    "none" passar aldrig här - den väljs av suggest_actions när risken är låg.
    """
    row = customer.iloc[0]
    has_internet = row["InternetService"] != "No"
    monthly = row["Contract"] == "Month-to-month"
    key = action["key"]

    if key in ("one_year", "two_year"):
        return monthly
    if key == "tech_support":
        return has_internet and row["TechSupport"] == "No"
    if key == "online_security":
        return has_internet and row["OnlineSecurity"] == "No"
    if key == "auto_payment":
        return row["PaymentMethod"] == "Electronic check"
    if key == "follow_up":
        return has_internet and monthly and row["tenure"] < 12
    return False


def apply_change(customer: pd.DataFrame, change: dict[str, str]) -> pd.DataFrame:
    """Ny enrads-DataFrame med ändringen införd. Kundens DataFrame rörs inte.

    ValueError om change rör något annat än en kategorikolumn (tenure, TotalCharges,
    MonthlyCharges), sätter en tjänst till "No" eller använder ett okänt värde.
    Okända värden måste stoppas här: prismodellen ignorerar dem tyst (handle_unknown).
    """
    if len(customer) != 1:
        raise ValueError(f"customer ska vara exakt en rad, fick {len(customer)}.")
    out = customer.copy()
    for column, value in change.items():
        if column not in RAW_CATEGORICAL_COLUMNS:
            raise ValueError(f"Ett alternativ får inte ändra {column}.")
        if value == "No":
            raise ValueError(f"Ett alternativ får inte ta bort en tjänst ({column} -> No).")
        if value not in VALUE_LABELS[column]:
            raise ValueError(
                f"Okänt värde i {column}: {value!r}. Tillåtna: {sorted(VALUE_LABELS[column])}"
            )
        out[column] = value
    return out


def suggest_actions(
    customer: pd.DataFrame, pipeline, price_pipeline, thresholds: tuple[float, float]
) -> list[dict]:
    """Max tre alternativ för kunden, sorterade efter bedömd effekt.

    Varje alternativ: key, label, kind, price, probability, level, note. Siffrorna är None
    för "regel". Låg risk ger bara "Ingen åtgärd". Sortering: what-if på lägst churn-
    sannolikhet (störst sänkning mot samma utgångsläge), sen "samband", sist "regel".
    """
    baseline = float(predict_proba(pipeline, customer).iloc[0])
    if risk_level(baseline, thresholds) == LOW:
        return [_row(get_action("none"), note=NOTES["none"])]

    rows = []
    for action in ACTIONS:
        if action["key"] == "none" or not fits(action, customer):
            continue
        if action["kind"] == "regel":
            rows.append(_row(action, note=NOTES[action["key"]]))
            continue

        # What-if: ändra, sätt nytt pris, bedöm risken igen med det nya priset.
        changed = apply_change(customer, action["change"])
        price = float(predict_price(price_pipeline, changed).iloc[0])
        changed["MonthlyCharges"] = price
        probability = float(predict_proba(pipeline, changed).iloc[0])
        rows.append(
            _row(
                action,
                price=price,
                probability=probability,
                level=risk_level(probability, thresholds),
                note=_describe_change(customer, action),
            )
        )

    rows.sort(key=lambda r: (KIND_ORDER[r["kind"]], r["probability"] or 0.0))
    return rows[:MAX_ACTIONS]


def _row(action: dict, price=None, probability=None, level=None, note="") -> dict:
    return {
        "key": action["key"],
        "label": action["label"],
        "kind": action["kind"],
        "price": price,
        "probability": probability,
        "level": level,
        "note": note,
    }


def _describe_change(customer: pd.DataFrame, action: dict) -> str:
    """Ändringen i klartext, t.ex. "Avtal: Månadsavtal -> Ettårsavtal"."""
    row = customer.iloc[0]
    parts = [
        f"{LABELS[column]}: {VALUE_LABELS[column][row[column]]} -> {VALUE_LABELS[column][value]}"
        for column, value in action["change"].items()
    ]
    text = ", ".join(parts)
    if action["key"] in NOTES:
        text += " " + NOTES[action["key"]]
    return text
