"""Tester för inläsning, validering och rensning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    SchemaError,
    clean,
    load_dataset,
    load_raw,
    split_features_target,
    validate,
    prepare_features,
)


def test_load_raw_reads_csv(tmp_path, raw_df):
    path = tmp_path / "d.csv"
    raw_df.to_csv(path, index=False)
    assert load_raw(path).shape == raw_df.shape


def test_load_raw_missing_file_raises(tmp_path):
    missing = tmp_path / "finns-inte.csv"
    with pytest.raises(FileNotFoundError, match="Datafilen saknas"):
        load_raw(missing)


def test_validate_accepts_valid_frame(raw_df):
    validate(raw_df)


def test_validate_rejects_empty_frame():
    with pytest.raises(SchemaError, match="tomt"):
        validate(pd.DataFrame(columns=[ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]))


@pytest.mark.parametrize("dropped", ["tenure", "Contract", TARGET_COLUMN])
def test_validate_reports_missing_column_by_name(raw_df, dropped):
    with pytest.raises(SchemaError, match=dropped):
        validate(raw_df.drop(columns=[dropped]))


def test_validate_lists_every_missing_column(raw_df):
    with pytest.raises(SchemaError) as err:
        validate(raw_df.drop(columns=["tenure", "gender"]))
    assert "tenure" in str(err.value)
    assert "gender" in str(err.value)


def test_validate_rejects_unexpected_target_values(raw_df):
    bad = raw_df.copy()
    bad.loc[0, TARGET_COLUMN] = "Kanske"
    with pytest.raises(SchemaError, match="Kanske"):
        validate(bad)


def test_validate_rejects_missing_target_values(raw_df):
    bad = raw_df.copy()
    bad.loc[0, TARGET_COLUMN] = None
    with pytest.raises(SchemaError, match="saknade värden"):
        validate(bad)


def test_validate_without_target_ignores_target_column(raw_df):
    validate(raw_df.drop(columns=[TARGET_COLUMN]), require_target=False)


def test_validate_without_target_still_requires_features(raw_df):
    with pytest.raises(SchemaError, match="tenure"):
        validate(raw_df.drop(columns=[TARGET_COLUMN, "tenure"]), require_target=False)


def test_clean_converts_blank_totalcharges_to_nan(raw_df):
    dirty = raw_df.copy()
    dirty.loc[0, "TotalCharges"] = " "
    cleaned = clean(dirty)
    assert pd.isna(cleaned.loc[0, "TotalCharges"])


def test_clean_converts_numeric_text_to_float(raw_df):
    dirty = raw_df.copy()
    dirty.loc[0, "TotalCharges"] = "1234.50"
    assert clean(dirty).loc[0, "TotalCharges"] == pytest.approx(1234.50)


def test_clean_converts_garbage_to_nan_instead_of_raising(raw_df):
    dirty = raw_df.copy()
    dirty["MonthlyCharges"] = dirty["MonthlyCharges"].astype(object)
    dirty.loc[0, "MonthlyCharges"] = "inte ett tal"
    assert pd.isna(clean(dirty).loc[0, "MonthlyCharges"])


def test_clean_does_not_mutate_input(raw_df):
    dirty = raw_df.copy()
    dirty.loc[0, "TotalCharges"] = " "
    before = dirty.copy()
    clean(dirty)
    pd.testing.assert_frame_equal(dirty, before)


def test_clean_makes_all_numeric_columns_numeric(raw_df):
    cleaned = clean(raw_df)
    for col in NUMERIC_COLUMNS:
        assert np.issubdtype(cleaned[col].dtype, np.number), col


def test_split_maps_target_to_binary(raw_df):
    _, y = split_features_target(raw_df)
    assert set(y.unique()) == {0, 1}
    assert y.sum() == (raw_df[TARGET_COLUMN] == "Yes").sum()


def test_split_drops_id_column(raw_df):
    X, _ = split_features_target(raw_df)
    assert ID_COLUMN not in X.columns
    assert list(X.columns) == FEATURE_COLUMNS


def test_split_drops_target_from_features(raw_df):
    X, _ = split_features_target(raw_df)
    assert TARGET_COLUMN not in X.columns


def test_load_dataset_on_real_file(real_data_path):
    X, y = load_dataset(real_data_path)
    assert len(X) == len(y) == 7043
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y.unique()) == {0, 1}


def test_load_dataset_rejects_broken_file(tmp_path, raw_df):
    path = tmp_path / "trasig.csv"
    raw_df.drop(columns=["Contract"]).to_csv(path, index=False)
    with pytest.raises(SchemaError, match="Contract"):
        load_dataset(path)

def test_prepare_features_cleans_and_selects(raw_df):
    X = prepare_features(raw_df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert np.issubdtype(X["TotalCharges"].dtype, np.number)


def test_prepare_features_works_without_target(raw_df):
    X = prepare_features(raw_df.drop(columns=[TARGET_COLUMN]))
    assert len(X) == len(raw_df)
