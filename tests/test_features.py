"""Tester för förbehandlaren."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, clean, split_features_target
from src.features import build_preprocessor


@pytest.fixture
def fitted(raw_df):
    X, _ = split_features_target(clean(raw_df))
    pre = build_preprocessor()
    pre.fit(X)
    return pre, X


def test_transform_returns_one_row_per_input_row(fitted):
    pre, X = fitted
    assert pre.transform(X).shape[0] == len(X)


def test_numeric_columns_are_standardised(fitted):
    pre, X = fitted
    out = pre.transform(X)
    numeric = out[:, : len(NUMERIC_COLUMNS)]
    assert np.allclose(numeric.mean(axis=0), 0, atol=1e-6)


def test_unknown_category_does_not_crash(fitted):
    pre, X = fitted
    unseen = X.iloc[[0]].copy()
    unseen.loc[:, "Contract"] = "Fyraårsavtal"
    out = pre.transform(unseen)
    assert out.shape[1] == pre.transform(X.iloc[[0]]).shape[1]
    assert np.isfinite(out).all()


def test_unknown_category_encodes_as_all_zeros(fitted):
    pre, X = fitted
    unseen = X.iloc[[0]].copy()
    unseen.loc[:, "Contract"] = "Fyraårsavtal"
    names = pre.get_feature_names_out()
    contract_idx = [i for i, n in enumerate(names) if n.startswith("cat__Contract_")]
    assert contract_idx
    assert pre.transform(unseen)[0, contract_idx].sum() == 0


def test_missing_numeric_value_is_imputed(fitted):
    pre, X = fitted
    gap = X.iloc[[0]].copy()
    gap.loc[:, "TotalCharges"] = np.nan
    assert np.isfinite(pre.transform(gap)).all()


def test_missing_categorical_value_is_imputed(fitted):
    pre, X = fitted
    gap = X.iloc[[0]].copy()
    gap.loc[:, "Contract"] = None
    assert np.isfinite(pre.transform(gap)).all()


def test_extra_column_is_dropped_not_passed_through(fitted):
    pre, X = fitted
    extra = X.copy()
    extra["skräpkolumn"] = 1
    assert pre.transform(extra).shape[1] == pre.transform(X).shape[1]


def test_every_configured_column_is_used(fitted):
    pre, _ = fitted
    names = " ".join(pre.get_feature_names_out())
    for col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS:
        assert col in names, col
