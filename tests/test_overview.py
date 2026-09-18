"""Tester för arbetslistan (src/overview.py). Syntetisk data och syntetiska sannolikheter -
ingen modell behövs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import ID_COLUMN
from src.overview import OVERVIEW_COLUMNS, customer_overview, search_customers
from src.risk import HIGH, LOW, MEDIUM
from tests.conftest import make_raw_df

N_CUSTOMERS = 60
THRESHOLDS = (0.3, 0.6)


@pytest.fixture(scope="module")
def customers() -> pd.DataFrame:
    return make_raw_df(N_CUSTOMERS, seed=3)


@pytest.fixture(scope="module")
def probabilities(customers) -> pd.Series:
    """Blandad ordning så att sorteringen faktiskt har något att göra."""
    values = np.linspace(0.0, 1.0, N_CUSTOMERS)
    np.random.default_rng(3).shuffle(values)
    return pd.Series(values, index=customers.index)


@pytest.fixture(scope="module")
def overview(customers, probabilities) -> pd.DataFrame:
    return customer_overview(customers, probabilities, THRESHOLDS)


# --- customer_overview ------------------------------------------------------------


def test_overview_has_contract_columns_and_one_row_per_customer(overview, customers):
    assert list(overview.columns) == OVERVIEW_COLUMNS
    assert len(overview) == len(customers)
    assert set(overview[ID_COLUMN]) == set(customers[ID_COLUMN])
    assert list(overview.index) == list(range(len(customers)))


def test_overview_sorted_high_medium_low_then_probability_descending(
    overview, customers, probabilities
):
    order = {HIGH: 0, MEDIUM: 1, LOW: 2}
    ranks = overview["level"].map(order).tolist()
    assert ranks == sorted(ranks)
    assert set(overview["level"]) == {HIGH, MEDIUM, LOW}

    by_id = dict(zip(customers[ID_COLUMN], probabilities, strict=True))
    for level in (HIGH, MEDIUM, LOW):
        probs = [by_id[cid] for cid in overview.loc[overview["level"] == level, ID_COLUMN]]
        assert probs == sorted(probs, reverse=True)


def test_overview_keeps_raw_values(overview, customers):
    first = overview.iloc[0]
    source = customers.set_index(ID_COLUMN).loc[first[ID_COLUMN]]
    for column in ["MonthlyCharges", "tenure", "Contract", "InternetService", "PaymentMethod"]:
        assert first[column] == source[column]


def test_overview_rejects_probabilities_of_other_length(customers, probabilities):
    with pytest.raises(ValueError, match="samma längd och index"):
        customer_overview(customers, probabilities.iloc[:10], THRESHOLDS)


def test_overview_rejects_probabilities_with_other_index(customers, probabilities):
    shifted = pd.Series(probabilities.to_numpy(), index=probabilities.index + 1)
    with pytest.raises(ValueError, match="samma längd och index"):
        customer_overview(customers, shifted, THRESHOLDS)


# --- search_customers -------------------------------------------------------------


def test_search_empty_or_blank_query_returns_everything(overview):
    assert search_customers(overview, "").equals(overview)
    assert search_customers(overview, "   ").equals(overview)


def test_search_digits_match_start_of_id(overview):
    hits = search_customers(overview, "000")
    assert len(hits) == 10
    assert hits[ID_COLUMN].str.startswith("000").all()
    # "5" står inne i 0005-TEST men inte i början - ingen träff där.
    assert "0005-TEST" not in set(search_customers(overview, "5")[ID_COLUMN])


def test_search_text_matches_anywhere_case_insensitively(overview):
    assert len(search_customers(overview, "test")) == N_CUSTOMERS
    assert len(search_customers(overview, "-TE")) == N_CUSTOMERS
    assert search_customers(overview, " 0012-test ")[ID_COLUMN].tolist() == ["0012-TEST"]


def test_search_without_hits_returns_empty_frame_with_same_columns(overview):
    hits = search_customers(overview, "FINNS-INTE")
    assert hits.empty
    assert list(hits.columns) == OVERVIEW_COLUMNS
