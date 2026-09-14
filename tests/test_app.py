"""Röktest av Streamlit-appen via AppTest - fångar fel som bara syns vid körning."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    return at


def test_app_runs_without_exceptions(app):
    assert not app.exception, [str(e) for e in app.exception]


def test_app_shows_no_error_box(app):
    assert not app.error, [e.value for e in app.error]


def test_app_renders_title_and_tabs(app):
    assert any("Churn" in m.value for m in app.title)
    assert len(app.tabs) >= 3


def test_app_reports_model_metrics(app):
    labels = {m.label for m in app.metric}
    assert {"ROC-AUC", "Recall", "Precision"} <= labels


def test_app_stops_with_error_when_data_is_missing(monkeypatch, tmp_path):
    """Saknad datafil ska ge ett läsbart felmeddelande, inte en stacktrace."""
    import streamlit as st

    import src.data as data_module

    monkeypatch.setattr(data_module, "DEFAULT_DATA_PATH", tmp_path / "finns-inte.csv")
    st.cache_data.clear()
    st.cache_resource.clear()

    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert at.error, "Appen gav inget felmeddelande trots saknad datafil"
    assert "Kunde inte starta" in at.error[0].value
