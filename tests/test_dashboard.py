"""Tester för src/dashboard.py. Syntetisk data, inget beroende på den sparade modellen."""

from __future__ import annotations

import numpy as np
import pytest

from src.dashboard import active_customers, filter_customers
from src.data import TARGET_COLUMN
from tests.conftest import make_raw_df


@pytest.fixture
def customers():
    """Rådata plus en påhittad churn-sannolikhet i [0, 1], deterministisk."""
    df = make_raw_df()
    df["churn_probability"] = np.linspace(0.0, 1.0, len(df))
    return df


def test_filter_customers_tom_dict_ger_alla_rader(customers):
    result = filter_customers(customers, {})
    assert len(result) == len(customers)


def test_filter_customers_tom_lista_ger_inget_filter(customers):
    result = filter_customers(customers, {"Contract": []})
    assert len(result) == len(customers)


def test_filter_customers_kombinerar_och_eller(customers):
    filters = {"Contract": ["One year", "Two year"], "InternetService": ["Fiber optic"]}
    result = filter_customers(customers, filters)
    expected = customers["Contract"].isin(["One year", "Two year"]) & (
        customers["InternetService"] == "Fiber optic"
    )
    assert len(result) == expected.sum()
    assert set(result["Contract"]) <= {"One year", "Two year"}
    assert (result["InternetService"] == "Fiber optic").all()


def test_filter_customers_fungerar_pa_heltalskolumn(customers):
    result = filter_customers(customers, {"SeniorCitizen": [1]})
    assert (result["SeniorCitizen"] == 1).all()


def test_filter_customers_returnerar_kopia(customers):
    result = filter_customers(customers, {"Contract": ["Two year"]})
    result["ny_kolumn"] = 1  # skulle varna om result vore en vy
    assert "ny_kolumn" not in customers.columns


def test_filter_customers_okand_kolumn_ger_keyerror(customers):
    with pytest.raises(KeyError, match="FinnsInte"):
        filter_customers(customers, {"FinnsInte": ["x"]})


def test_filter_customers_noll_rader_ger_valueerror(customers):
    with pytest.raises(ValueError):
        filter_customers(customers, {"Contract": ["Finns inte"]})


def test_active_customers_ger_bara_churn_no(customers):
    result = active_customers(customers)
    assert (result[TARGET_COLUMN] == "No").all()
    assert len(result) == len(customers) // 2  # make_raw_df: "No" på jämna index


def test_active_customers_utan_churn_kolumn_ger_keyerror(customers):
    with pytest.raises(KeyError, match=TARGET_COLUMN):
        active_customers(customers.drop(columns=[TARGET_COLUMN]))
