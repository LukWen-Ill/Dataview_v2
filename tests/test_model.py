"""Tester för träning, prediktion och persistens."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import SchemaError, clean, load_dataset, split_features_target, prepare_features
from src.model import Metrics, build_pipeline, load, predict_proba, save, train


@pytest.fixture
def trained(raw_df):
    X, y = split_features_target(clean(raw_df))
    return train(X, y), X


def test_train_returns_pipeline_and_metrics(trained):
    (pipeline, metrics), _ = trained
    assert hasattr(pipeline, "predict_proba")
    assert isinstance(metrics, Metrics)


@pytest.mark.parametrize("field", ["accuracy", "precision", "recall", "f1", "roc_auc"])
def test_metrics_are_valid_probabilities(trained, field):
    (_, metrics), _ = trained
    assert 0.0 <= getattr(metrics, field) <= 1.0


def test_split_sizes_add_up(trained):
    (_, metrics), X = trained
    assert metrics.n_train + metrics.n_test == len(X)
    assert metrics.n_test > 0


def test_train_rejects_mismatched_lengths(raw_df):
    X, y = split_features_target(clean(raw_df))
    with pytest.raises(ValueError, match="olika längd"):
        train(X, y.iloc[:-1])


def test_train_rejects_single_class_target(raw_df):
    X, y = split_features_target(clean(raw_df))
    with pytest.raises(ValueError, match="en klass"):
        train(X, pd.Series(np.zeros(len(y), dtype=int), index=y.index))


def test_train_is_deterministic(raw_df):
    X, y = split_features_target(clean(raw_df))
    first = train(X, y, random_state=7)[1]
    second = train(X, y, random_state=7)[1]
    assert first == second


def test_predict_proba_is_in_unit_interval(trained):
    (pipeline, _), X = trained
    proba = predict_proba(pipeline, X)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_predict_proba_keeps_index(trained):
    (pipeline, _), X = trained
    subset = X.iloc[[3, 9, 15]]
    assert list(predict_proba(pipeline, subset).index) == [3, 9, 15]


def test_predict_proba_rejects_missing_column(trained):
    (pipeline, _), X = trained
    with pytest.raises(SchemaError, match="Contract"):
        predict_proba(pipeline, X.drop(columns=["Contract"]))


def test_predict_proba_rejects_empty_frame(trained):
    (pipeline, _), X = trained
    with pytest.raises(SchemaError, match="tomt"):
        predict_proba(pipeline, X.iloc[0:0])


def test_predict_proba_survives_unknown_category(trained):
    (pipeline, _), X = trained
    unseen = X.iloc[[0]].copy()
    unseen.loc[:, "PaymentMethod"] = "Swish"
    assert 0.0 <= predict_proba(pipeline, unseen).iloc[0] <= 1.0


def test_predict_proba_survives_missing_values(trained):
    (pipeline, _), X = trained
    gap = X.iloc[[0]].copy()
    gap.loc[:, "TotalCharges"] = np.nan
    assert 0.0 <= predict_proba(pipeline, gap).iloc[0] <= 1.0


def test_predict_proba_ignores_column_order(trained):
    (pipeline, _), X = trained
    shuffled = X.iloc[[0]][list(reversed(X.columns))]
    expected = predict_proba(pipeline, X.iloc[[0]]).iloc[0]
    assert predict_proba(pipeline, shuffled).iloc[0] == pytest.approx(expected)


def test_save_load_roundtrip_gives_identical_predictions(trained, tmp_path):
    (pipeline, _), X = trained
    path = save(pipeline, tmp_path / "nested" / "m.joblib")
    assert path.is_file()
    np.testing.assert_allclose(
        predict_proba(load(path), X).to_numpy(),
        predict_proba(pipeline, X).to_numpy(),
    )


def test_load_missing_model_raises_with_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        load(tmp_path / "finns-inte.joblib")


def test_untrained_pipeline_cannot_predict(raw_df):
    X, _ = split_features_target(clean(raw_df))
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        build_pipeline().predict(X)


def test_real_data_beats_majority_baseline(real_data_path):
    X, y = load_dataset(real_data_path)
    _, metrics = train(X, y)
    assert metrics.roc_auc > 0.75, f"ROC-AUC {metrics.roc_auc} - modellen är inte bättre än slump"
    assert metrics.recall > 0.5, f"Recall {metrics.recall} - missar för många churnare"

def test_predict_proba_handles_raw_unclean_frame(trained, raw_df):
    """Regression: Streamlit-batchen skickar rå CSV-data, inte förrensad."""
    (pipeline, _), _ = trained
    dirty = raw_df.iloc[[0]].copy()
    dirty.loc[:, "TotalCharges"] = " "   # som de 11 raderna med tenure=0 i CSV:n

    assert not pd.api.types.is_numeric_dtype(dirty["TotalCharges"])

    proba = predict_proba(pipeline, dirty)
    assert 0.0 <= proba.iloc[0] <= 1.0
