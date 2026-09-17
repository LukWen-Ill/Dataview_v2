"""Röktest av Streamlit-appen via AppTest - kör varje sida och failar på oväntade undantag.

Sidorna använder den riktiga databasen och den sparade modellen i models/, precis som i drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    ROOT / "Dashboard.py",
    ROOT / "pages" / "0_Översikt.py",
    ROOT / "pages" / "1_Data.py",
    ROOT / "pages" / "2_Modeller.py",
    ROOT / "pages" / "3_Segmentering.py",
    ROOT / "pages" / "4_Prediktera.py",
]


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
    at = run_page(PAGES[1])
    labels = {m.label for m in at.metric}
    assert {"ROC-AUC", "Recall", "Precision", "Vald modell"} <= labels


def test_predict_page_makes_a_prediction_with_default_values():
    at = run_page(PAGES[5])
    at.button[0].click().run()
    assert not at.exception, [str(e) for e in at.exception]
    labels = {m.label for m in at.metric}
    assert "Sannolikhet för churn" in labels


def test_app_stops_with_error_when_model_is_missing(monkeypatch, tmp_path):
    """Saknad modell ska ge ett läsbart felmeddelande med träningskommandot, inte en stacktrace."""
    import src.model as model_module

    monkeypatch.setattr(model_module, "DEFAULT_MODEL_PATH", tmp_path / "finns-inte.joblib")
    at = run_page(PAGES[5])
    assert not at.exception
    assert at.error and "src.train" in at.error[0].value


def test_app_stops_with_error_when_data_is_missing(monkeypatch, tmp_path):
    """Saknad databas OCH CSV ska ge ett läsbart felmeddelande."""
    import src.data as data_module
    import src.db as db_module

    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", tmp_path / "finns-inte.db")
    monkeypatch.setattr(data_module, "DEFAULT_DATA_PATH", tmp_path / "finns-inte.csv")
    at = run_page(PAGES[1])
    assert not at.exception
    assert at.error and "Kunde inte ladda" in at.error[0].value


def test_customers_with_predictions_matches_customers():
    """churn_probability i [0, 1] och samma rader i samma ordning som get_customers()."""
    from app_helpers import get_customers, get_customers_with_predictions

    st.cache_data.clear()
    st.cache_resource.clear()
    customers = get_customers()
    result = get_customers_with_predictions()
    assert "churn_probability" in result.columns
    assert result["churn_probability"].between(0, 1).all()
    assert len(result) == len(customers)
    assert list(result["customerID"]) == list(customers["customerID"])


def test_customers_with_predictions_raises_when_model_is_missing(monkeypatch, tmp_path):
    """Saknad modell ska ge FileNotFoundError som load_or_stop fångar på sidan."""
    import src.model as model_module
    from app_helpers import get_customers_with_predictions

    monkeypatch.setattr(model_module, "DEFAULT_MODEL_PATH", tmp_path / "finns-inte.joblib")
    st.cache_data.clear()
    st.cache_resource.clear()
    with pytest.raises(FileNotFoundError):
        get_customers_with_predictions()


def test_dashboard_page_shows_kpis_and_table():
    """Default-filter (tomma listor) ska ge KPI-rad, graf och tabell utan fel."""
    at = run_page(PAGES[0])
    assert not at.exception, [str(e) for e in at.exception]
    labels = {m.label for m in at.metric}
    assert "MRR idag" in labels
    assert len(at.dataframe) >= 1


def test_dashboard_page_has_segment_filter():
    at = run_page(PAGES[0])
    assert not at.exception, [str(e) for e in at.exception]
    assert "Segment" in {m.label for m in at.multiselect}


def test_segment_labels_match_customer_rows():
    """Segmentetiketterna läggs på kundtabellen radvis, så längden måste stämma."""
    from app_helpers import get_customers, get_segments
    from src.segment import DEFAULT_K

    st.cache_data.clear()
    st.cache_resource.clear()
    labels, _, _ = get_segments(DEFAULT_K)
    assert len(labels) == len(get_customers())
