"""Tester för prisregressionen: pris som funktion av valda tjänster."""

from __future__ import annotations

import pytest

from src.pricing import PRICE_FEATURES, prepare_pricing_data, price_list, train_pricing_model


def test_prepare_excludes_customers_without_internet(raw_df):
    X, _ = prepare_pricing_data(raw_df)
    assert "No" not in X["InternetService"].unique()
    assert list(X.columns) == PRICE_FEATURES


def test_prepare_returns_matching_lengths(raw_df):
    X, y = prepare_pricing_data(raw_df)
    assert len(X) == len(y)


def test_prepare_raises_on_missing_column(raw_df):
    with pytest.raises(KeyError):
        prepare_pricing_data(raw_df.drop(columns=["InternetService"]))


def test_train_returns_pipeline_and_r2(raw_df):
    X, y = prepare_pricing_data(raw_df)
    pipeline, r2 = train_pricing_model(X, y)
    assert hasattr(pipeline, "predict")
    assert r2 <= 1.0


def test_price_list_has_one_row_per_service(raw_df):
    X, y = prepare_pricing_data(raw_df)
    pipeline, _ = train_pricing_model(X, y)
    assert len(price_list(pipeline)) == len(PRICE_FEATURES)


def test_price_list_values_are_non_negative(raw_df):
    X, y = prepare_pricing_data(raw_df)
    pipeline, _ = train_pricing_model(X, y)
    assert (price_list(pipeline)["USD_per_manad"] >= 0).all()
