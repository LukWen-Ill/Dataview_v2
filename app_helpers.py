"""Delade, cachade laddningsfunktioner för Streamlit-sidorna. Ingen ML-logik här."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src import data, db, model, train
from src.segment import fit_segments, kmeans_scores, preprocess_for_clustering

TRAIN_HINT = "Träna modellen först: `python -m src.train`"


@st.cache_data
def get_customers() -> pd.DataFrame:
    """Kunddatan från SQLite. Bygger databasen från CSV:n om den saknas."""
    if not db.DEFAULT_DB_PATH.is_file():
        db.init_db(data.DEFAULT_DATA_PATH, db.DEFAULT_DB_PATH)
    return db.load_customers(db.DEFAULT_DB_PATH)


@st.cache_resource
def get_model():
    """Den sparade slutmodellen. Appen tränar aldrig själv."""
    return model.load(model.DEFAULT_MODEL_PATH)


@st.cache_data
def get_results() -> dict:
    """results.json från senaste träningen."""
    if not train.RESULTS_PATH.is_file():
        raise FileNotFoundError(f"Saknar {train.RESULTS_PATH}. {TRAIN_HINT}")
    return json.loads(train.RESULTS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def get_customers_with_predictions() -> pd.DataFrame:
    """Kundtabellen med kolumnen churn_probability från den sparade modellen.

    Alla kunder får en sannolikhet, även de som redan lämnat - Ledning-sidan filtrerar
    själv via src/dashboard.py. Radordningen är samma som i get_customers().
    """
    customers = get_customers()
    return customers.assign(churn_probability=model.predict_proba(get_model(), customers))


@st.cache_data
def get_test_predictions() -> pd.DataFrame:
    """Slutmodellens sannolikheter på testmängden - används för threshold-analysen."""
    if not train.TEST_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(f"Saknar {train.TEST_PREDICTIONS_PATH}. {TRAIN_HINT}")
    return pd.read_csv(train.TEST_PREDICTIONS_PATH)


@st.cache_data
def get_kmeans_scores() -> pd.DataFrame:
    return kmeans_scores(preprocess_for_clustering(get_customers()))


@st.cache_data
def get_segments(k: int):
    return fit_segments(preprocess_for_clustering(get_customers()), k)


def load_or_stop(loader, what: str):
    """Kör en laddningsfunktion; visa ett läsbart fel och stoppa sidan om den misslyckas."""
    try:
        return loader()
    except (FileNotFoundError, data.SchemaError) as err:
        st.error(f"Kunde inte ladda {what}: {err}")
        st.stop()


def metrics_row(metrics: dict, keys=("roc_auc", "recall", "precision", "f1", "accuracy")) -> None:
    """Visa en rad st.metric för en metrics-dict."""
    labels = {
        "roc_auc": "ROC-AUC",
        "recall": "Recall",
        "precision": "Precision",
        "f1": "F1",
        "accuracy": "Accuracy",
    }
    cols = st.columns(len(keys))
    for col, key in zip(cols, keys, strict=True):
        col.metric(labels[key], f"{metrics[key]:.3f}", border=True)
