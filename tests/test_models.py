"""Tester för modellkatalogen och EDA-hjälparna."""

from __future__ import annotations

import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from src.data import NUMERIC_COLUMNS, split_features_target
from src.eda import EDA_CATEGORIES, churn_correlations, churn_rate_by
from src.models import build_candidates, build_pipeline


def test_three_candidates_with_grids():
    candidates = build_candidates()
    assert set(candidates) == {"Logistisk regression", "Beslutsträd", "Random forest"}
    for pipeline, grid in candidates.values():
        assert [name for name, _ in pipeline.steps] == ["preprocess", "classifier"]
        assert grid, "varje modell ska ha ett hyperparametergrid"
        assert all(key.startswith("classifier__") for key in grid)


def test_grid_keys_are_real_hyperparameters():
    """Ett felstavat gridnamn ger annars ett kryptiskt fel först inne i GridSearchCV."""
    for pipeline, grid in build_candidates().values():
        for key, values in grid.items():
            pipeline.set_params(**{key: values[0]})


def test_all_candidates_use_balanced_class_weight():
    for pipeline, _ in build_candidates().values():
        assert pipeline.named_steps["classifier"].class_weight == "balanced"


def test_untrained_pipeline_cannot_predict(raw_df):
    X, _ = split_features_target(raw_df)
    with pytest.raises(NotFittedError):
        build_pipeline(LogisticRegression()).predict(X)


def test_churn_rate_by_sums_to_all_customers(raw_df):
    rates = churn_rate_by(raw_df, "Contract")
    assert rates["antal_kunder"].sum() == len(raw_df)
    assert rates["churn_andel"].between(0, 1).all()


def test_churn_rate_by_works_for_engineered_column(raw_df):
    assert not churn_rate_by(raw_df, "tenure_group").empty


def test_churn_rate_by_unknown_column_raises(raw_df):
    with pytest.raises(KeyError, match="finns inte"):
        churn_rate_by(raw_df, "påhittad")


def test_eda_categories_exist_after_feature_engineering(raw_df):
    for column in EDA_CATEGORIES:
        churn_rate_by(raw_df, column)


def test_churn_correlations_cover_all_numeric_columns(raw_df):
    corr = churn_correlations(raw_df)
    assert set(corr.index) == set(NUMERIC_COLUMNS)
    assert corr.between(-1, 1).all()
