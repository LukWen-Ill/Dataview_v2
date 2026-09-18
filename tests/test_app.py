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

from src import actions, db
from src.actions import suggest_actions
from src.data import ADDON_SERVICES, ID_COLUMN, RAW_FEATURE_COLUMNS, clean
from src.labels import LABELS
from src.model import load, predict_proba
from src.overview import customer_overview
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
KNOWN_CUSTOMER_ID = "7590-VHVEG"  # första raden i CSV:n: månadsavtal, DSL, 1 månad, hög risk
LOW_RISK_CUSTOMER_ID = "5575-GNVDE"  # ettårsavtal, DSL, 34 månader
MEDIUM_RISK_CUSTOMER_ID = "6713-OKOMC"  # månadsavtal, DSL, 10 månader
NO_INTERNET_CUSTOMER_ID = "1066-JKSGK"  # bara telefoni, månadsavtal, 1 månad
AUTO_PAYMENT_CANDIDATE_ID = "5067-XJQFU"  # ettårsavtal, fiber, elektronisk check

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


def test_sales_tool_form_has_person_products_payment_columns_and_hides_addons_without_internet():
    """Kolumnerna Person | Produkter | Betalning i samtalets ordning; inuti en kolumn kan
    fälten ligga två i bredd, så bara gruppordningen är fast."""
    at = run_page(SALES_PAGE)
    labels = [s.label for s in at.selectbox]
    person = ["SeniorCitizen", "gender", "Partner", "Dependents"]
    assert set(labels[:4]) == {LABELS[c] for c in person}
    assert labels[4:6] == [LABELS["PhoneService"], LABELS["InternetService"]]
    assert labels[6:] == [LABELS[c] for c in ["Contract", "PaperlessBilling", "PaymentMethod"]]
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
    assert expected["tenure"] == 1
    assert metrics(at)["Kundtid"] == "1 månad"
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


# --- Säljverktyg: alternativ och Spara kund (#40) ----------------------------------


def ui_texts(at: AppTest) -> str:
    """All text sidan visar, för att kontrollera ordval."""
    parts = [e.value for kind in (at.markdown, at.caption, at.info, at.success) for e in kind]
    parts += [f"{m.label} {m.value}" for m in at.metric]
    parts += [frame.value.to_string() for frame in at.dataframe]
    return " | ".join(parts)


def alternatives(at: AppTest) -> pd.DataFrame | None:
    """Alternativtabellen i resultatvyn, eller None om den inte visas."""
    for frame in at.dataframe:
        if "Alternativ" in frame.value.columns:
            return frame.value
    return None


def overview_list(at: AppTest) -> pd.DataFrame | None:
    """Arbetslistan på fliken Befintlig kund, eller None om den inte visas."""
    for frame in at.dataframe:
        if frame.key and frame.key.startswith("lista_"):
            return frame.value
    return None


def search_customer(customer_id: str) -> AppTest:
    at = run_page(SALES_PAGE)
    at.text_input[0].input(customer_id).run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def expected_alternatives(customer_id: str) -> list[dict]:
    """Samma urval som sidan ska visa: suggest_actions minus what-if utan lägre sannolikhet."""
    customers = db.load_customers(db.DEFAULT_DB_PATH)
    pipeline, price_model = load(), load_price_model()
    thresholds = risk_thresholds(predict_proba(pipeline, customers))
    customer = clean(db.find_customer(customer_id, db.DEFAULT_DB_PATH))[RAW_FEATURE_COLUMNS]
    baseline = predict_proba(pipeline, customer).iloc[0]
    return [
        a
        for a in suggest_actions(customer, pipeline, price_model, thresholds)
        if a["probability"] is None or a["probability"] < baseline
    ]


def test_sales_tool_shows_alternatives_with_price_and_level_side_by_side():
    at = search_customer(KNOWN_CUSTOMER_ID)
    expected = expected_alternatives(KNOWN_CUSTOMER_ID)
    table = alternatives(at)
    assert list(table.columns) == ["Alternativ", "Pris", "Risknivå", "Underlag", "Kommentar"]
    assert table["Alternativ"].tolist() == [a["label"] for a in expected]
    assert table["Pris"].tolist() == [f"{a['price']:.2f} $/mån" for a in expected]
    assert table["Risknivå"].tolist() == [a["level"].capitalize() for a in expected]
    assert 1 <= len(table) <= 3


def test_sales_tool_hides_what_if_that_does_not_lower_the_probability():
    """Teknisk support ger den här kunden högre sannolikhet än utgångsläget och ska inte visas."""
    at = search_customer(MEDIUM_RISK_CUSTOMER_ID)
    labels = alternatives(at)["Alternativ"].tolist()
    assert labels == [a["label"] for a in expected_alternatives(MEDIUM_RISK_CUSTOMER_ID)]
    assert "Teknisk support" not in labels
    assert "Tvåårsavtal" in labels


def test_sales_tool_low_risk_customer_gets_no_action_rule_without_numbers():
    at = search_customer(LOW_RISK_CUSTOMER_ID)
    table = alternatives(at)
    assert table["Alternativ"].tolist() == ["Ingen åtgärd"]
    assert table["Pris"].tolist() == ["–"] and table["Risknivå"].tolist() == ["–"]
    assert "Regel" in table["Underlag"].iloc[0]


def test_sales_tool_marks_correlation_rows_as_not_proven():
    at = search_customer(AUTO_PAYMENT_CANDIDATE_ID)
    table = alternatives(at)
    row = table[table["Alternativ"] == "Automatisk banköverföring"]
    assert len(row) == 1
    assert row["Underlag"].iloc[0] == "Samband i datan, inte bevisad effekt"


def test_sales_tool_customer_without_internet_gets_no_internet_addons():
    at = search_customer(NO_INTERNET_CUSTOMER_ID)
    labels = set(alternatives(at)["Alternativ"])
    assert labels
    assert not labels & {"Teknisk support", "Onlinesäkerhet"}


def test_sales_tool_says_so_when_no_alternative_is_left(monkeypatch):
    """Sidan importerar suggest_actions vid varje körning, så ett byte i src.actions slår igenom."""
    worse = {
        "key": "x",
        "label": "Test",
        "kind": "what-if",
        "price": 1.0,
        "probability": 1.0,
        "level": HIGH,
        "note": "",
    }
    monkeypatch.setattr(actions, "suggest_actions", lambda *args: [worse])
    at = search_customer(KNOWN_CUSTOMER_ID)
    assert alternatives(at) is None
    assert any("inget alternativ bedöms sänka risken" in c.value.lower() for c in at.caption)


def test_sales_tool_never_claims_causation():
    at = search_customer(KNOWN_CUSTOMER_ID)
    text = ui_texts(at)
    assert "sänker" not in text.lower()
    assert "%" not in ui_texts(at).replace("% churn mot", "").replace("% i snitt", "")
    assert "statistisk modell" in text


def test_sales_tool_saves_new_customer_and_finds_it_again(monkeypatch, tmp_path):
    tmp_db = tmp_path / "churn.db"
    shutil.copy(db.DEFAULT_DB_PATH, tmp_db)
    monkeypatch.setattr(db, "DEFAULT_DB_PATH", tmp_db)

    at = fill_new_customer(run_page(SALES_PAGE), NEW_FIBER_CUSTOMER)
    price_shown = metrics(at)["Pris"]
    at.button(key="spara_ny").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.success and "NEW-000001" in at.success[0].value

    saved = db.load_new_customers(tmp_db)
    assert len(saved) == 1
    assert saved.loc[0, "tenure"] == 0 and float(saved.loc[0, "TotalCharges"]) == 0.0

    at.text_input[0].input("NEW-000001").run()
    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error
    assert metrics(at)["Dagens pris"] == price_shown
    assert metrics(at)["Kundtid"] == "0 månader"


# --- Säljverktyg: arbetslista och sök (#44) -----------------------------------------


def expected_overview() -> pd.DataFrame:
    """Samma lista som sidan ska visa, byggd direkt med src-funktionerna ur aktuell databas."""
    columns = [ID_COLUMN, *RAW_FEATURE_COLUMNS]
    pipeline = load()
    customers = db.load_customers(db.DEFAULT_DB_PATH)
    probabilities = predict_proba(pipeline, customers)
    thresholds = risk_thresholds(probabilities)
    frame = customers[columns]
    new = db.load_new_customers(db.DEFAULT_DB_PATH)
    if not new.empty:
        frame = pd.concat([frame, new[columns]], ignore_index=True)
        probabilities = pd.concat([probabilities, predict_proba(pipeline, new)], ignore_index=True)
    return customer_overview(frame, probabilities, thresholds)


def test_sales_tool_list_has_high_risk_first_and_25_rows_per_page():
    at = run_page(SALES_PAGE)
    expected = expected_overview()
    table = overview_list(at)
    assert list(table.columns) == [
        "Kund-id",
        "Risknivå",
        "Månadskostnad",
        "Kundtid (månader)",
        "Avtal",
        "Internet",
        "Betalsätt",
    ]
    assert len(table) == 25
    assert table["Risknivå"].tolist() == [HIGH.capitalize()] * 25
    assert table["Kund-id"].tolist() == expected[ID_COLUMN].iloc[:25].tolist()
    assert any(c.value.startswith("Rad 1–25 av") for c in at.caption)
    assert "Dagens pris" not in metrics(at)  # ingen kund laddad förrän man väljer

    at.number_input(key="sida_").set_value(2).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert overview_list(at)["Kund-id"].tolist() == expected[ID_COLUMN].iloc[25:50].tolist()
    assert any(c.value.startswith("Rad 26–50 av") for c in at.caption)


def test_sales_tool_digits_search_filters_list_by_id_prefix():
    at = run_page(SALES_PAGE)
    expected = expected_overview()
    at.text_input[0].input("75").run()
    hits = expected[expected[ID_COLUMN].str.startswith("75")]
    assert len(hits) > 1
    assert overview_list(at)["Kund-id"].tolist() == hits[ID_COLUMN].iloc[:25].tolist()
    assert "Dagens pris" not in metrics(at)  # flera träffar: filtrera, ladda inte

    at.text_input[0].input("7590").run()
    shown = overview_list(at)["Kund-id"].tolist()
    assert KNOWN_CUSTOMER_ID in shown
    assert all(cid.startswith("7590") for cid in shown)


def test_sales_tool_single_hit_loads_customer_directly():
    at = search_customer(KNOWN_CUSTOMER_ID)
    assert overview_list(at)["Kund-id"].tolist() == [KNOWN_CUSTOMER_ID]
    assert "Dagens pris" in metrics(at)


def test_sales_tool_search_without_hits_shows_message_and_no_list():
    at = run_page(SALES_PAGE)
    at.text_input[0].input("0000").run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.error and "0000" in at.error[0].value
    assert overview_list(at) is None
    assert "Dagens pris" not in metrics(at)


def test_sales_tool_selecting_a_row_loads_that_customer():
    at = run_page(SALES_PAGE)
    chosen = expected_overview().iloc[2]
    at.session_state["lista__1"] = {"selection": {"rows": [2], "columns": []}}
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert metrics(at)["Dagens pris"] == f"{chosen['MonthlyCharges']:.2f} $/mån"
    assert at.selectbox(key=f"befintlig_{chosen[ID_COLUMN]}_Contract").value == chosen["Contract"]


def test_sales_tool_saved_customer_appears_in_list_and_is_found_on_new(monkeypatch, tmp_path):
    tmp_db = tmp_path / "churn.db"
    shutil.copy(db.DEFAULT_DB_PATH, tmp_db)
    monkeypatch.setattr(db, "DEFAULT_DB_PATH", tmp_db)
    values = {**NEW_FIBER_CUSTOMER, "tenure": 0, "MonthlyCharges": 70.0, "TotalCharges": 0.0}
    customer_id = db.save_new_customer(values, tmp_db)

    at = run_page(SALES_PAGE)
    assert customer_id in set(expected_overview()[ID_COLUMN])
    at.text_input[0].input("new").run()
    assert not at.exception, [str(e) for e in at.exception]
    table = overview_list(at)
    # Textsök matchar var som helst i id:t, så även CSV-kunder som 7554-NEWDD kommer med.
    assert all("NEW" in cid.upper() for cid in table["Kund-id"])
    row = table[table["Kund-id"] == customer_id]
    assert len(row) == 1
    assert row["Risknivå"].iloc[0] == HIGH.capitalize()  # ny fiberkund på månadsavtal

    at.text_input[0].input(customer_id).run()
    assert metrics(at)["Dagens pris"] == "70.00 $/mån"  # fullt id: enda träffen laddas direkt
