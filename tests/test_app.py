"""Röktest av Streamlit-appen via AppTest - kör varje sida och failar på oväntade undantag.

Sidorna använder den riktiga databasen och den sparade modellen i models/, precis som i drift.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import db
from src.data import ADDON_SERVICES, RAW_FEATURE_COLUMNS, clean
from src.labels import LABELS
from src.model import load, predict_proba
from src.price import load_price_model, predict_price
from src.risk import HIGH, risk_level, risk_thresholds

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    ROOT / "app.py",
    ROOT / "pages" / "1_Data.py",
    ROOT / "pages" / "2_Modeller.py",
    ROOT / "pages" / "3_Segmentering.py",
    ROOT / "pages" / "4_Prediktera.py",
    ROOT / "pages" / "6_Saljverktyg.py",
]
SALES_PAGE = PAGES[5]
KNOWN_CUSTOMER_ID = "7590-VHVEG"  # första raden i CSV:n

# En ny fiberkund på månadsavtal utan tillägg, i formulärets ordning.
NEW_FIBER_CUSTOMER = {
    "SeniorCitizen": 0,
    "gender": "Male",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    **{service: "No" for service in ADDON_SERVICES},
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


def run_page(path: Path) -> AppTest:
    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file(str(path), default_timeout=180)
    at.run()
    return at


@pytest.fixture(scope="module", params=PAGES, ids=lambda p: p.stem)
def page(request) -> AppTest:
    return run_page(request.param)


def test_page_runs_without_exceptions(page):
    assert not page.exception, [str(e) for e in page.exception]


def test_page_shows_no_error_box(page):
    assert not page.error, [e.value for e in page.error]


def test_page_has_a_title(page):
    assert page.title


def test_overview_reports_model_metrics():
    at = run_page(PAGES[0])
    labels = {m.label for m in at.metric}
    assert {"ROC-AUC", "Recall", "Precision", "Vald modell"} <= labels


def test_predict_page_makes_a_prediction_with_default_values():
    at = run_page(PAGES[4])
    at.button[0].click().run()
    assert not at.exception, [str(e) for e in at.exception]
    labels = {m.label for m in at.metric}
    assert "Sannolikhet för churn" in labels


def test_app_stops_with_error_when_model_is_missing(monkeypatch, tmp_path):
    """Saknad modell ska ge ett läsbart felmeddelande med träningskommandot, inte en stacktrace."""
    import src.model as model_module

    monkeypatch.setattr(model_module, "DEFAULT_MODEL_PATH", tmp_path / "finns-inte.joblib")
    at = run_page(PAGES[4])
    assert not at.exception
    assert at.error and "src.train" in at.error[0].value


def test_app_stops_with_error_when_data_is_missing(monkeypatch, tmp_path):
    """Saknad databas OCH CSV ska ge ett läsbart felmeddelande."""
    import src.data as data_module
    import src.db as db_module

    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", tmp_path / "finns-inte.db")
    monkeypatch.setattr(data_module, "DEFAULT_DATA_PATH", tmp_path / "finns-inte.csv")
    at = run_page(PAGES[0])
    assert not at.exception
    assert at.error and "Kunde inte ladda" in at.error[0].value


# --- Säljverktyg ------------------------------------------------------------------


def metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def fill_new_customer(at: AppTest, values: dict) -> AppTest:
    """Välj fälten i samtalets ordning. Varje val kör om sidan så att villkorliga fält dyker upp."""
    for column, value in values.items():
        at.selectbox(key=f"ny_{column}").select(value).run()
    return at


def expected_price_and_level(values: dict) -> tuple[float, str]:
    """Samma beräkning som sidan ska visa, gjord direkt med src-funktionerna."""
    customers = db.load_customers(db.DEFAULT_DB_PATH)
    pipeline, price_model = load(), load_price_model()
    thresholds = risk_thresholds(predict_proba(pipeline, customers))
    customer = pd.DataFrame([values])
    price = float(predict_price(price_model, customer).iloc[0])
    customer["MonthlyCharges"] = price
    probability = predict_proba(pipeline, customer[RAW_FEATURE_COLUMNS]).iloc[0]
    return price, risk_level(probability, thresholds)


def test_sales_tool_form_follows_the_conversation_and_hides_addons_without_internet():
    at = run_page(SALES_PAGE)
    labels = [s.label for s in at.selectbox]
    first = ["SeniorCitizen", "gender", "Partner", "Dependents", "PhoneService", "InternetService"]
    assert labels[: len(first)] == [LABELS[c] for c in first]
    assert LABELS["OnlineSecurity"] not in labels  # tillägg visas först när internet är valt
    assert not at.metric  # ofullständigt formulär: inget pris och ingen risknivå

    at.selectbox(key="ny_InternetService").select("Fiber optic").run()
    assert {LABELS[s] for s in ADDON_SERVICES} <= {s.label for s in at.selectbox}

    at.selectbox(key="ny_InternetService").select("No").run()
    assert not {LABELS[s] for s in ADDON_SERVICES} & {s.label for s in at.selectbox}
    assert not at.exception


def test_sales_tool_prices_and_grades_a_new_fiber_customer():
    at = fill_new_customer(run_page(SALES_PAGE), NEW_FIBER_CUSTOMER)
    assert not at.exception, [str(e) for e in at.exception]
    values = {**NEW_FIBER_CUSTOMER, "tenure": 0, "TotalCharges": 0.0}
    price, level = expected_price_and_level(values)
    assert level == HIGH
    assert metrics(at)["Pris"] == f"{price:.2f} $/mån"
    assert metrics(at)["Risknivå"] == HIGH.capitalize()
    assert any("churn mot" in m.value for m in at.markdown)  # riskfaktorer


def test_sales_tool_sets_no_internet_service_on_addons_when_customer_has_no_internet():
    without_internet = {
        col: value for col, value in NEW_FIBER_CUSTOMER.items() if col not in ADDON_SERVICES
    }
    at = fill_new_customer(run_page(SALES_PAGE), {**without_internet, "InternetService": "No"})
    assert not at.exception, [str(e) for e in at.exception]
    values = {
        **NEW_FIBER_CUSTOMER,
        "InternetService": "No",
        **{service: "No internet service" for service in ADDON_SERVICES},
        "tenure": 0,
        "TotalCharges": 0.0,
    }
    price, level = expected_price_and_level(values)
    assert metrics(at)["Pris"] == f"{price:.2f} $/mån"
    assert metrics(at)["Risknivå"] == level.capitalize()


def test_sales_tool_loads_known_csv_customer_with_contract_services_price_and_tenure():
    at = run_page(SALES_PAGE)
    at.text_input[0].input(KNOWN_CUSTOMER_ID).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error

    expected = clean(db.find_customer(KNOWN_CUSTOMER_ID, db.DEFAULT_DB_PATH)).iloc[0]
    assert metrics(at)["Dagens pris"] == f"{expected['MonthlyCharges']:.2f} $/mån"
    assert metrics(at)["Kundtid"] == f"{expected['tenure']} månader"
    assert at.number_input(key=f"befintlig_{KNOWN_CUSTOMER_ID}_tenure").value == expected["tenure"]
    for column in ["Contract", "InternetService", "OnlineBackup", "PaymentMethod"]:
        assert at.selectbox(key=f"befintlig_{KNOWN_CUSTOMER_ID}_{column}").value == expected[column]
    assert any("Kunder som liknar den här" in m.value for m in at.markdown)
    assert any("churn mot" in m.value for m in at.markdown)


def test_sales_tool_unknown_customer_id_shows_message_without_crashing():
    at = run_page(SALES_PAGE)
    at.text_input[0].input("FINNS-INTE").run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.error and "FINNS-INTE" in at.error[0].value
    assert "Dagens pris" not in metrics(at)


def test_sales_tool_finds_customer_saved_in_register(monkeypatch, tmp_path):
    """En kund sparad via save_new_customer hittas på sitt id. Skriver i en kopia av databasen."""
    tmp_db = tmp_path / "churn.db"
    shutil.copy(db.DEFAULT_DB_PATH, tmp_db)
    monkeypatch.setattr(db, "DEFAULT_DB_PATH", tmp_db)
    values = {**NEW_FIBER_CUSTOMER, "tenure": 0, "MonthlyCharges": 70.0, "TotalCharges": 0.0}
    customer_id = db.save_new_customer(values, tmp_db)

    at = run_page(SALES_PAGE)
    at.text_input[0].input(customer_id).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error
    assert metrics(at)["Dagens pris"] == "70.00 $/mån"
    assert metrics(at)["Kundtid"] == "0 månader"
    assert at.selectbox(key=f"befintlig_{customer_id}_InternetService").value == "Fiber optic"


def test_sales_tool_changing_contract_recomputes_price_and_moves_risk_level():
    at = run_page(SALES_PAGE)
    at.text_input[0].input(KNOWN_CUSTOMER_ID).run()
    level_today = metrics(at)["Risknivå"]

    at.selectbox(key=f"befintlig_{KNOWN_CUSTOMER_ID}_Contract").select("Two year").run()
    assert not at.exception, [str(e) for e in at.exception]

    stored = clean(db.find_customer(KNOWN_CUSTOMER_ID, db.DEFAULT_DB_PATH)).iloc[0]
    values = {**stored[RAW_FEATURE_COLUMNS].to_dict(), "Contract": "Two year"}
    price, level = expected_price_and_level(values)
    assert metrics(at)["Pris"] == f"{price:.2f} $/mån"
    assert metrics(at)["Risknivå"] == level.capitalize() != level_today
    assert metrics(at)["Dagens pris"] == f"{stored['MonthlyCharges']:.2f} $/mån"
