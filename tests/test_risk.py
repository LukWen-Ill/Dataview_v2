"""Tester för riskkärnan (src/risk.py) och de kundvänliga namnen (src/labels.py).

Modellen tränas här i testet på syntetisk data - inget test beror på den incheckade
modellfilen.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data import (
    RAW_CATEGORICAL_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    split_features_target,
)
from src.labels import LABELS, VALUE_LABELS
from src.model import predict_proba
from src.models import build_pipeline
from src.risk import (
    HIGH,
    LOW,
    MEDIUM,
    risk_factors,
    risk_level,
    risk_thresholds,
    segment_comparison,
    tenure_band,
)
from src.segment import DEFAULT_K
from tests.conftest import CATEGORY_VALUES, make_raw_df

N_CUSTOMERS = 300
FACTOR_PATTERN = re.compile(r"^(.+): (\d+) % churn mot (\d+) % i snitt$")


@pytest.fixture(scope="module")
def customers() -> pd.DataFrame:
    """Syntetisk träningsdata där churn hänger ihop med avtal, fiber och kort kundtid.

    make_raw_df:s Churn växlar bara med radnumret, oberoende av kundens val. För att
    en modell ska kunna lära sig något byter vi till en regel: minst två av
    månadsavtal (väger dubbelt), fiber och kundtid under ett år ger churn.
    """
    df = make_raw_df(N_CUSTOMERS, seed=1)
    score = (
        2 * (df["Contract"] == "Month-to-month")
        + (df["InternetService"] == "Fiber optic")
        + (df["tenure"] < 12)
    )
    df[TARGET_COLUMN] = ["Yes" if s >= 2 else "No" for s in score]
    return df


@pytest.fixture(scope="module")
def pipeline(customers):
    """Liten pipeline tränad här i testet - samma förbehandling som den riktiga modellen."""
    X, y = split_features_target(customers)
    pipeline = build_pipeline(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    )
    return pipeline.fit(X, y)


@pytest.fixture(scope="module")
def probabilities(pipeline, customers) -> pd.Series:
    return predict_proba(pipeline, customers)


def _one_customer(customers, **fields) -> pd.DataFrame:
    """En rad rådata utan Churn, med angivna fält överskrivna."""
    row = customers.iloc[[0]].drop(columns=[TARGET_COLUMN]).copy()
    for column, value in fields.items():
        row[column] = value
    return row


def new_fiber_customer(customers) -> pd.DataFrame:
    """Ny kund: fiber på månadsavtal, inga tillägg, tenure 0 och tom TotalCharges som i CSV:n."""
    return _one_customer(
        customers,
        tenure=0,
        TotalCharges=" ",
        Contract="Month-to-month",
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="No",
        StreamingMovies="No",
    )


def phone_two_year_customer(customers) -> pd.DataFrame:
    """Telefonikund utan internet på tvåårsavtal."""
    return _one_customer(
        customers,
        tenure=60,
        TotalCharges="3000.00",
        Contract="Two year",
        PhoneService="Yes",
        MultipleLines="Yes",
        InternetService="No",
        OnlineSecurity="No internet service",
        OnlineBackup="No internet service",
        DeviceProtection="No internet service",
        TechSupport="No internet service",
        StreamingTV="No internet service",
        StreamingMovies="No internet service",
    )


# --- labels -----------------------------------------------------------------------


def test_labels_cover_all_feature_columns():
    assert set(LABELS) == set(RAW_FEATURE_COLUMNS)
    assert all(text for text in LABELS.values())


def test_value_labels_cover_all_categorical_values():
    assert set(VALUE_LABELS) == set(RAW_CATEGORICAL_COLUMNS)
    for column in RAW_CATEGORICAL_COLUMNS:
        assert set(VALUE_LABELS[column]) == set(CATEGORY_VALUES[column]), column


def test_value_labels_are_swedish_and_customer_friendly():
    assert VALUE_LABELS["Contract"]["Month-to-month"] == "Månadsavtal"
    assert VALUE_LABELS["PaymentMethod"]["Electronic check"] == "Elektronisk check"
    assert VALUE_LABELS["InternetService"]["Fiber optic"] == "Fiber"


# --- risknivå ---------------------------------------------------------------------


def test_risk_thresholds_are_tertiles_and_deterministic(probabilities):
    low, high = risk_thresholds(probabilities)
    assert 0.0 <= low <= high <= 1.0
    assert risk_thresholds(probabilities) == (low, high)
    assert low == pytest.approx(probabilities.quantile(1 / 3))
    assert high == pytest.approx(probabilities.quantile(2 / 3))


def test_risk_thresholds_split_customers_in_thirds(probabilities):
    thresholds = risk_thresholds(probabilities)
    levels = pd.Series([risk_level(p, thresholds) for p in probabilities]).value_counts()
    assert set(levels.index) == {LOW, MEDIUM, HIGH}
    assert all(abs(count - N_CUSTOMERS / 3) <= 3 for count in levels)


def test_risk_thresholds_reject_empty_series():
    with pytest.raises(ValueError, match="Inga sannolikheter"):
        risk_thresholds(pd.Series([], dtype=float))


def test_risk_level_boundaries():
    thresholds = (0.2, 0.6)
    assert risk_level(0.0, thresholds) == LOW
    assert risk_level(0.2, thresholds) == LOW
    assert risk_level(0.4, thresholds) == MEDIUM
    assert risk_level(0.6, thresholds) == MEDIUM
    assert risk_level(0.9, thresholds) == HIGH


def test_new_fiber_customer_is_high_and_phone_two_year_is_low(customers, pipeline, probabilities):
    thresholds = risk_thresholds(probabilities)
    fiber = predict_proba(pipeline, new_fiber_customer(customers)).iloc[0]
    phone = predict_proba(pipeline, phone_two_year_customer(customers)).iloc[0]
    assert risk_level(fiber, thresholds) == HIGH
    assert risk_level(phone, thresholds) == LOW


# --- riskfaktorer -----------------------------------------------------------------


def test_tenure_band_edges():
    assert tenure_band(0) == tenure_band(11) == "Kundtid under 1 år"
    assert tenure_band(12) == tenure_band(36) == "Kundtid 1–3 år"
    assert tenure_band(37) == "Kundtid över 3 år"


def test_risk_factors_include_month_to_month_with_correct_numbers(customers):
    factors = risk_factors(new_fiber_customer(customers), customers)
    churned = customers[TARGET_COLUMN] == "Yes"
    rate = churned[customers["Contract"] == "Month-to-month"].mean()
    expected = f"Månadsavtal: {rate * 100:.0f} % churn mot {churned.mean() * 100:.0f} % i snitt"
    assert expected in factors


def test_risk_factors_are_at_most_three_and_sorted_by_lift(customers):
    factors = risk_factors(new_fiber_customer(customers), customers)
    assert 1 <= len(factors) <= 3
    rates = [int(FACTOR_PATTERN.match(text).group(2)) for text in factors]
    assert rates == sorted(rates, reverse=True)


def test_risk_factors_only_report_values_clearly_above_average(customers):
    """Varje rad ska visa ett tal som är större än snittet även efter avrundning."""
    for text in risk_factors(phone_two_year_customer(customers), customers):
        match = FACTOR_PATTERN.match(text)
        assert match, text
        assert int(match.group(2)) > int(match.group(3))
        assert "Månadsavtal" not in text


def test_risk_factors_merge_columns_that_share_the_same_customer_group():
    """ "Inget internet" står i sju kolumner men är samma kunder - ska bli en enda rad."""
    df = make_raw_df(60, seed=2)
    df[TARGET_COLUMN] = ["Yes" if v == "No" else "No" for v in df["InternetService"]]
    factors = risk_factors(phone_two_year_customer(df), df)
    internet_lines = [text for text in factors if text.startswith("Inget internet: ")]
    assert len(internet_lines) == 1
    assert internet_lines[0].startswith("Inget internet: 100 % churn mot ")


def test_risk_factors_reject_missing_column(customers):
    customer = new_fiber_customer(customers).drop(columns=["Contract"])
    with pytest.raises(KeyError, match="Contract"):
        risk_factors(customer, customers)


def test_risk_factors_reject_unknown_category(customers):
    customer = new_fiber_customer(customers)
    customer["PaymentMethod"] = "Swish"
    with pytest.raises(ValueError, match="PaymentMethod.*Swish"):
        risk_factors(customer, customers)


def test_risk_factors_reject_more_than_one_row(customers):
    with pytest.raises(ValueError, match="exakt en rad"):
        risk_factors(customers.iloc[:2], customers)


# --- segmentjämförelse -------------------------------------------------------------


def test_segment_comparison_returns_contract_and_is_deterministic(customers, probabilities):
    customer = new_fiber_customer(customers)
    result = segment_comparison(customer, customers, probabilities)
    assert set(result) == {"segment", "segment_probability", "average_probability"}
    assert isinstance(result["segment"], int)
    assert 0 <= result["segment"] < DEFAULT_K
    assert result["average_probability"] == pytest.approx(probabilities.mean())
    assert probabilities.min() <= result["segment_probability"] <= probabilities.max()
    assert segment_comparison(customer, customers, probabilities) == result


def test_segment_comparison_rejects_missing_column(customers, probabilities):
    customer = new_fiber_customer(customers).drop(columns=["tenure"])
    with pytest.raises(KeyError, match="tenure"):
        segment_comparison(customer, customers, probabilities)


def test_segment_comparison_rejects_unknown_category(customers, probabilities):
    customer = new_fiber_customer(customers)
    customer["Contract"] = "Livstid"
    with pytest.raises(ValueError, match="Contract.*Livstid"):
        segment_comparison(customer, customers, probabilities)


def test_segment_comparison_rejects_probabilities_of_wrong_length(customers, probabilities):
    with pytest.raises(ValueError, match="en rad per kund"):
        segment_comparison(new_fiber_customer(customers), customers, probabilities.iloc[:10])
