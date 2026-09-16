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
    ROOT / "app.py",
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
