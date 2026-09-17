"""Riskkärna för Säljverktyg-sidan: risknivå, riskfaktorer och segmentjämförelse.

Rå sannolikhet visas aldrig för säljaren. class_weight="balanced" i churnmodellen gör
sannolikheterna uppblåsta, så vi visar bara *relativ* risk:

- Risknivå: var kunden ligger i modellens fördelning över träningsdatan (tertiler).
- Riskfaktorer: vilka av kundens val som i träningsdatan hänger ihop med mer churn.
- Segment: kundens K-Means-segment och dess snittrisk mot hela kundbasens.

Sannolikheterna kommer från predict_proba i src/model.py. Ingen Streamlit här.
"""

from __future__ import annotations

import pandas as pd

from src.data import RAW_CATEGORICAL_COLUMNS, RAW_FEATURE_COLUMNS, TARGET_COLUMN
from src.labels import VALUE_LABELS
from src.segment import DEFAULT_K, fit_segments, preprocess_for_clustering

LOW, MEDIUM, HIGH = "låg", "medel", "hög"
MAX_FACTORS = 3
MIN_LIFT = 0.01  # minst en procentenhet, annars blir raden "27 % churn mot 27 % i snitt"


def risk_thresholds(probabilities: pd.Series) -> tuple[float, float]:
    """Tertilgränser (q1/3, q2/3) för churnsannolikheterna över träningsdatan.

    Deterministiskt: samma modell och data ger samma gränser. Kastar ValueError om tom.
    """
    if probabilities.empty:
        raise ValueError("Inga sannolikheter att beräkna risknivåer på.")
    low, high = probabilities.quantile([1 / 3, 2 / 3])
    return float(low), float(high)


def risk_level(probability: float, thresholds: tuple[float, float]) -> str:
    """LOW om sannolikheten är <= q1/3, HIGH om den är > q2/3, annars MEDIUM."""
    low, high = thresholds
    if probability <= low:
        return LOW
    if probability <= high:
        return MEDIUM
    return HIGH


def tenure_band(tenure: float) -> str:
    """Kundtiden i tre band. Churn är kraftigt koncentrerad till första året."""
    if tenure < 12:
        return "Kundtid under 1 år"
    if tenure <= 36:
        return "Kundtid 1–3 år"
    return "Kundtid över 3 år"


def risk_factors(customer: pd.DataFrame, customers: pd.DataFrame) -> list[str]:
    """Kundens största riskfaktorer i klartext, störst lyft först, max tre rader.

    Lyft = churnandelen i träningsdatan bland kunder med samma värde, minus snittet.
    Jämförs för varje kategorikolumn och kundtidsbandet. Bara värden som ligger minst
    MIN_LIFT *över* snittet räknas som riskfaktorer, så en trygg kund kan få färre än tre.

    customer är exakt en rad rådata. customers är träningsdatan med Churn-kolumnen.
    """
    _check_customer(customer)
    churned = customers[TARGET_COLUMN] == "Yes"
    average = churned.mean()

    factors = _describe(customers)
    mine = _describe(customer).iloc[0]

    # text -> churnandel. Nyckel på texten: "Inget internet" står i sju kolumner men ska
    # bara bli en rad, och då med det största lyftet.
    lifts: dict[str, float] = {}
    for column in factors.columns:
        rate = churned[factors[column] == mine[column]].mean()  # NaN om värdet saknas i datan
        if rate - average >= MIN_LIFT:
            lifts[mine[column]] = max(rate, lifts.get(mine[column], 0.0))

    ranked = sorted(lifts.items(), key=lambda item: item[1], reverse=True)
    return [
        f"{text}: {rate * 100:.0f} % churn mot {average * 100:.0f} % i snitt"
        for text, rate in ranked[:MAX_FACTORS]
    ]


def segment_comparison(
    customer: pd.DataFrame, customers: pd.DataFrame, probabilities: pd.Series
) -> dict:
    """Kundens segment och segmentets snittrisk mot hela kundbasens.

    Klustrar på samma sätt som Segmentering-sidan (samma förbehandling, K-Means med
    DEFAULT_K) med kunden tillagd som sista rad, så att kunden får ett segment utan att
    K-Means-modellen behöver sparas. probabilities är predict_proba över customers.
    """
    _check_customer(customer)
    if len(probabilities) != len(customers):
        raise ValueError(
            f"probabilities ({len(probabilities)}) måste ha en rad per kund ({len(customers)})."
        )

    together = pd.concat(
        [customers[RAW_FEATURE_COLUMNS], customer[RAW_FEATURE_COLUMNS]], ignore_index=True
    )
    labels, _, _ = fit_segments(preprocess_for_clustering(together), DEFAULT_K)
    segment = labels[-1]
    in_segment = labels[:-1] == segment

    return {
        "segment": int(segment),
        "segment_probability": float(probabilities.to_numpy()[in_segment].mean()),
        "average_probability": float(probabilities.mean()),
    }


def _check_customer(customer: pd.DataFrame) -> None:
    """Exakt en rad, alla råkolumner på plats och kända kategorivärden.

    Måste köras före prepare_features/predict_proba: de ger SchemaError vid saknad
    kolumn och ignorerar okända kategorier tyst (handle_unknown="ignore").
    """
    if len(customer) != 1:
        raise ValueError(f"customer ska vara exakt en rad, fick {len(customer)}.")
    for column in RAW_FEATURE_COLUMNS:
        if column not in customer.columns:
            raise KeyError(column)
    row = customer.iloc[0]
    for column in RAW_CATEGORICAL_COLUMNS:
        if row[column] not in VALUE_LABELS[column]:
            raise ValueError(
                f"Okänt värde i {column}: {row[column]!r}. Tillåtna: {sorted(VALUE_LABELS[column])}"
            )


def _describe(df: pd.DataFrame) -> pd.DataFrame:
    """Kategorikolumnerna och kundtidsbandet som kundvänlig klartext, en kolumn per faktor."""
    out = pd.DataFrame(index=df.index)
    for column in RAW_CATEGORICAL_COLUMNS:
        out[column] = df[column].map(VALUE_LABELS[column])
    out["tenure"] = df["tenure"].apply(tenure_band)
    return out
