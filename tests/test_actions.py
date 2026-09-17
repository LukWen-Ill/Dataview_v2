"""Tester för åtgärdskatalogen (src/actions.py).

Churn- och prismodell tränas här i testet på syntetisk data - inget test beror på de
incheckade modellfilerna.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.actions import (
    ACTIONS,
    MAX_ACTIONS,
    apply_change,
    fits,
    get_action,
    suggest_actions,
)
from src.data import ADDON_SERVICES, TARGET_COLUMN, split_features_target
from src.model import predict_proba
from src.models import build_pipeline
from src.price import PRICE_FEATURE_COLUMNS, PRICE_TARGET, build_price_pipeline, predict_price
from src.risk import HIGH, LOW, MEDIUM, risk_thresholds
from tests.conftest import make_raw_df

N_CUSTOMERS = 300
ROW_KEYS = {"key", "label", "kind", "price", "probability", "level", "note"}
NO_INTERNET = {col: "No internet service" for col in ADDON_SERVICES}


@pytest.fixture(scope="module")
def customers() -> pd.DataFrame:
    """Syntetisk träningsdata, som i test_risk.py: churn hänger ihop med avtal, fiber och
    kort kundtid. Priset är en summa av tjänsterna så att prismodellen blir nästan exakt."""
    df = make_raw_df(N_CUSTOMERS, seed=1)
    score = (
        2 * (df["Contract"] == "Month-to-month")
        + (df["InternetService"] == "Fiber optic")
        + (df["tenure"] < 12)
    )
    df[TARGET_COLUMN] = ["Yes" if s >= 2 else "No" for s in score]
    df[PRICE_TARGET] = (
        20.0
        + 25.0 * (df["InternetService"] == "DSL")
        + 50.0 * (df["InternetService"] == "Fiber optic")
        + 5.0 * df[ADDON_SERVICES].eq("Yes").sum(axis=1)
    )
    return df


@pytest.fixture(scope="module")
def pipeline(customers):
    X, y = split_features_target(customers)
    pipeline = build_pipeline(
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=0)
    )
    return pipeline.fit(X, y)


@pytest.fixture(scope="module")
def price_pipeline(customers):
    pipeline = build_price_pipeline(LinearRegression())
    return pipeline.fit(customers[PRICE_FEATURE_COLUMNS], customers[PRICE_TARGET])


@pytest.fixture(scope="module")
def thresholds(pipeline, customers) -> tuple[float, float]:
    return risk_thresholds(predict_proba(pipeline, customers))


def _one_customer(customers, **fields) -> pd.DataFrame:
    """En rad rådata utan Churn, med angivna fält överskrivna."""
    row = customers.iloc[[0]].drop(columns=[TARGET_COLUMN]).copy()
    for column, value in fields.items():
        row[column] = value
    return row


def new_fiber_customer(customers) -> pd.DataFrame:
    """Ny kund: fiber på månadsavtal utan tillägg, elektronisk check. Sex åtgärder passar."""
    return _one_customer(
        customers,
        tenure=0,
        TotalCharges=" ",
        MonthlyCharges=999.0,  # ett pris prismodellen aldrig kan ge - avslöjar om det läcker
        Contract="Month-to-month",
        PaymentMethod="Electronic check",
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        **{col: "No" for col in ADDON_SERVICES},
    )


def dsl_all_addons_customer(customers, **fields) -> pd.DataFrame:
    """DSL med alla tillägg på månadsavtal, check per post: bara avtal och uppföljning passar."""
    defaults = {
        "tenure": 5,
        "TotalCharges": "300.00",
        "MonthlyCharges": 75.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Mailed check",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        **{col: "Yes" for col in ADDON_SERVICES},
    }
    return _one_customer(customers, **{**defaults, **fields})


def phone_two_year_customer(customers) -> pd.DataFrame:
    """Telefonikund utan internet på tvåårsavtal - låg risk i den syntetiska datan."""
    return _one_customer(
        customers,
        tenure=60,
        TotalCharges="1200.00",
        MonthlyCharges=20.0,
        Contract="Two year",
        PaymentMethod="Bank transfer (automatic)",
        PhoneService="Yes",
        MultipleLines="Yes",
        InternetService="No",
        **NO_INTERNET,
    )


# --- katalogen -------------------------------------------------------------------


def test_catalog_has_unique_keys_and_valid_kinds():
    keys = [action["key"] for action in ACTIONS]
    assert len(keys) == len(set(keys))
    for action in ACTIONS:
        assert set(action) == {"key", "label", "kind", "change"}
        assert action["kind"] in {"what-if", "samband", "regel"}
        assert (action["change"] == {}) == (action["kind"] == "regel")


def test_catalog_never_touches_tenure_total_charges_or_removes_services():
    for action in ACTIONS:
        assert "tenure" not in action["change"]
        assert "TotalCharges" not in action["change"]
        assert "No" not in action["change"].values()


def test_get_action_known_and_unknown_key():
    assert get_action("one_year")["change"] == {"Contract": "One year"}
    with pytest.raises(KeyError, match="rabatt"):
        get_action("rabatt")


# --- apply_change ----------------------------------------------------------------


def test_apply_change_returns_new_row_and_leaves_customer_untouched(customers):
    customer = new_fiber_customer(customers)
    before = customer.copy()
    changed = apply_change(customer, {"Contract": "One year", "TechSupport": "Yes"})
    assert len(changed) == 1
    assert changed is not customer
    assert changed["Contract"].iloc[0] == "One year"
    assert changed["TechSupport"].iloc[0] == "Yes"
    assert changed["InternetService"].iloc[0] == "Fiber optic"
    pd.testing.assert_frame_equal(customer, before)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"tenure": "12"}, "tenure"),
        ({"TotalCharges": "0"}, "TotalCharges"),
        ({"MonthlyCharges": "10"}, "MonthlyCharges"),
        ({"TechSupport": "No"}, "ta bort en tjänst"),
        ({"PaymentMethod": "Swish"}, "PaymentMethod.*Swish"),
    ],
)
def test_apply_change_rejects_forbidden_changes(customers, change, message):
    with pytest.raises(ValueError, match=message):
        apply_change(new_fiber_customer(customers), change)


def test_apply_change_rejects_more_than_one_row(customers):
    with pytest.raises(ValueError, match="exakt en rad"):
        apply_change(customers.iloc[:2], {"Contract": "One year"})


# --- fits ------------------------------------------------------------------------


def test_no_internet_customer_gets_no_internet_addons(customers):
    customer = _one_customer(
        customers,
        Contract="Month-to-month",
        PaymentMethod="Electronic check",
        InternetService="No",
        tenure=3,
        **NO_INTERNET,
    )
    fitting = {action["key"] for action in ACTIONS if fits(action, customer)}
    assert fitting == {"one_year", "two_year", "auto_payment"}


def test_two_year_customer_gets_no_contract_or_follow_up(customers):
    customer = new_fiber_customer(customers)
    customer["Contract"] = "Two year"
    fitting = {action["key"] for action in ACTIONS if fits(action, customer)}
    assert fitting == {"tech_support", "online_security", "auto_payment"}


def test_new_fiber_customer_fits_everything_but_none(customers):
    customer = new_fiber_customer(customers)
    fitting = {action["key"] for action in ACTIONS if fits(action, customer)}
    assert fitting == {
        "one_year",
        "two_year",
        "tech_support",
        "online_security",
        "auto_payment",
        "follow_up",
    }


def test_follow_up_requires_first_year(customers):
    customer = dsl_all_addons_customer(customers, tenure=12)
    assert not fits(get_action("follow_up"), customer)
    assert fits(get_action("follow_up"), dsl_all_addons_customer(customers, tenure=11))


# --- suggest_actions -------------------------------------------------------------


def test_low_risk_customer_gets_only_no_action(customers, pipeline, price_pipeline, thresholds):
    result = suggest_actions(
        phone_two_year_customer(customers), pipeline, price_pipeline, thresholds
    )
    assert len(result) == 1
    row = result[0]
    assert set(row) == ROW_KEYS
    assert row["key"] == "none"
    assert row["label"] == "Ingen åtgärd"
    assert row["kind"] == "regel"
    assert row["price"] is None and row["probability"] is None and row["level"] is None
    assert row["note"]


def test_at_most_three_rows_with_contract_keys_and_no_input_mutation(
    customers, pipeline, price_pipeline, thresholds
):
    customer = new_fiber_customer(customers)
    before = customer.copy()
    result = suggest_actions(customer, pipeline, price_pipeline, thresholds)
    assert 1 <= len(result) <= MAX_ACTIONS
    known = {action["key"] for action in ACTIONS}
    for row in result:
        assert set(row) == ROW_KEYS
        assert row["key"] in known and row["key"] != "none"
        assert isinstance(row["price"], float)
        assert isinstance(row["probability"], float)
        assert row["level"] in {LOW, MEDIUM, HIGH}
    pd.testing.assert_frame_equal(customer, before)


def test_longer_contract_lowers_probability_and_rule_comes_last(
    customers, pipeline, price_pipeline, thresholds
):
    """DSL-kund med alla tillägg och check per post: bara avtalen och uppföljningen passar."""
    customer = dsl_all_addons_customer(customers)
    baseline = predict_proba(pipeline, customer).iloc[0]
    result = suggest_actions(customer, pipeline, price_pipeline, thresholds)
    # Ett- och tvåårsavtal ger båda churn "No" i den syntetiska datan, så ordningen
    # mellan dem avgörs av små skillnader - vi kräver bara sortering på sannolikhet.
    assert {row["key"] for row in result[:2]} == {"one_year", "two_year"}
    assert result[2]["key"] == "follow_up"
    probabilities = [row["probability"] for row in result[:2]]
    assert probabilities == sorted(probabilities)
    for row in result[:2]:
        assert row["kind"] == "what-if"
        assert row["probability"] < baseline
    rule = result[2]
    assert rule["kind"] == "regel"
    assert rule["price"] is None and rule["probability"] is None and rule["level"] is None
    assert "Uppföljning" in rule["label"]


def test_what_ifs_sorted_by_probability_then_samband(
    customers, pipeline, price_pipeline, thresholds
):
    customer = dsl_all_addons_customer(customers, PaymentMethod="Electronic check")
    result = suggest_actions(customer, pipeline, price_pipeline, thresholds)
    kinds = [row["kind"] for row in result]
    assert kinds == ["what-if", "what-if", "samband"]
    probabilities = [row["probability"] for row in result[:2]]
    assert probabilities == sorted(probabilities)
    assert result[2]["key"] == "auto_payment"
    assert "inte orsak" in result[2]["note"]


def test_price_comes_from_price_model_not_old_monthly_charges(
    customers, pipeline, price_pipeline, thresholds
):
    customer = new_fiber_customer(customers)
    result = suggest_actions(customer, pipeline, price_pipeline, thresholds)
    for row in result:
        changed = apply_change(customer, get_action(row["key"])["change"])
        expected = predict_price(price_pipeline, changed).iloc[0]
        assert row["price"] == pytest.approx(expected)
        assert row["price"] != customer["MonthlyCharges"].iloc[0]
        assert 60 < row["price"] < 90  # fiber 70 (+ 5 för ett tillägg), aldrig 999


def test_probability_is_computed_with_the_new_price(
    customers, pipeline, price_pipeline, thresholds
):
    """Pris och risk hör ihop: churnmodellen ska ha sett det nya priset, inte det gamla."""
    customer = new_fiber_customer(customers)
    result = suggest_actions(customer, pipeline, price_pipeline, thresholds)
    for row in result:
        changed = apply_change(customer, get_action(row["key"])["change"])
        changed["MonthlyCharges"] = float(predict_price(price_pipeline, changed).iloc[0])
        expected = predict_proba(pipeline, changed).iloc[0]
        assert row["probability"] == pytest.approx(expected)


def test_note_describes_change_in_customer_friendly_words(
    customers, pipeline, price_pipeline, thresholds
):
    result = suggest_actions(
        dsl_all_addons_customer(customers), pipeline, price_pipeline, thresholds
    )
    notes = {row["key"]: row["note"] for row in result}
    assert notes["one_year"] == "Avtal: Månadsavtal -> Ettårsavtal"
    assert notes["two_year"] == "Avtal: Månadsavtal -> Tvåårsavtal"
