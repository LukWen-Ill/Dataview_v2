"""Tester för prediktion och persistens av den tränade modellen."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import TARGET_COLUMN, SchemaError
from src.model import load, predict_proba, save


@pytest.fixture
def customers(raw_df):
    """Rådata utan target - så som appen skickar in en kund."""
    return raw_df.drop(columns=[TARGET_COLUMN])


def test_predict_proba_is_in_unit_interval(fitted_pipeline, customers):
    proba = predict_proba(fitted_pipeline, customers)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert proba.name == "churn_probability"


def test_predict_proba_keeps_index(fitted_pipeline, customers):
    subset = customers.iloc[[3, 9, 15]]
    assert list(predict_proba(fitted_pipeline, subset).index) == [3, 9, 15]


def test_predict_proba_accepts_target_column_if_present(fitted_pipeline, raw_df):
    """En CSV med Churn-kolumnen ska också gå att prediktera på - kolumnen ignoreras."""
    assert len(predict_proba(fitted_pipeline, raw_df)) == len(raw_df)


def test_predict_proba_rejects_missing_column(fitted_pipeline, customers):
    with pytest.raises(SchemaError, match="Contract"):
        predict_proba(fitted_pipeline, customers.drop(columns=["Contract"]))


def test_predict_proba_rejects_empty_frame(fitted_pipeline, customers):
    with pytest.raises(SchemaError, match="tomt"):
        predict_proba(fitted_pipeline, customers.iloc[0:0])


def test_predict_proba_survives_unknown_category(fitted_pipeline, customers):
    unseen = customers.iloc[[0]].copy()
    unseen.loc[:, "PaymentMethod"] = "Swish"
    assert 0.0 <= predict_proba(fitted_pipeline, unseen).iloc[0] <= 1.0


def test_predict_proba_survives_missing_values(fitted_pipeline, customers):
    gap = customers.iloc[[0]].copy()
    gap.loc[:, "TotalCharges"] = np.nan
    assert 0.0 <= predict_proba(fitted_pipeline, gap).iloc[0] <= 1.0


def test_predict_proba_survives_text_in_totalcharges(fitted_pipeline, customers):
    """Precis som i rå-CSV:n: ett blanksteg i stället för ett tal."""
    gap = customers.iloc[[0]].copy()
    gap.loc[:, "TotalCharges"] = " "
    assert 0.0 <= predict_proba(fitted_pipeline, gap).iloc[0] <= 1.0


def test_predict_proba_ignores_column_order(fitted_pipeline, customers):
    shuffled = customers.iloc[[0]][list(reversed(customers.columns))]
    expected = predict_proba(fitted_pipeline, customers.iloc[[0]]).iloc[0]
    assert predict_proba(fitted_pipeline, shuffled).iloc[0] == pytest.approx(expected)


def test_save_load_roundtrip_gives_identical_predictions(fitted_pipeline, customers, tmp_path):
    path = save(fitted_pipeline, tmp_path / "nested" / "m.joblib")
    assert path.is_file()
    np.testing.assert_allclose(
        predict_proba(load(path), customers).to_numpy(),
        predict_proba(fitted_pipeline, customers).to_numpy(),
    )


def test_load_missing_model_raises_with_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        load(tmp_path / "finns-inte.joblib")
