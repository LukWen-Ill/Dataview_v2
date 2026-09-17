"""Tester för SQLite-databasen."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pytest

from src.data import RAW_FEATURE_COLUMNS, REQUIRED_COLUMNS, SchemaError
from src.db import (
    find_customer,
    init_db,
    load_customers,
    load_model_runs,
    load_new_customers,
    load_predictions,
    log_model_run,
    log_prediction,
    main,
    save_new_customer,
)


def test_init_db_creates_file_and_all_tables(db_path):
    assert db_path.is_file()
    with closing(sqlite3.connect(db_path)) as conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"customers", "new_customers", "model_runs", "predictions"} <= tables


def test_customers_table_has_every_row_and_column(db_path, raw_df):
    customers = load_customers(db_path)
    assert len(customers) == len(raw_df)
    assert set(REQUIRED_COLUMNS) <= set(customers.columns)


def test_init_db_is_idempotent(tmp_path, raw_df):
    csv_path = tmp_path / "raw.csv"
    raw_df.to_csv(csv_path, index=False)
    init_db(csv_path, tmp_path / "x.db")
    init_db(csv_path, tmp_path / "x.db")
    assert len(load_customers(tmp_path / "x.db")) == len(raw_df)


def test_init_db_rejects_broken_csv(tmp_path, raw_df):
    csv_path = tmp_path / "trasig.csv"
    raw_df.drop(columns=["Contract"]).to_csv(csv_path, index=False)
    with pytest.raises(SchemaError, match="Contract"):
        init_db(csv_path, tmp_path / "x.db")


def test_init_db_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Datafilen saknas"):
        init_db(tmp_path / "finns-inte.csv", tmp_path / "x.db")


def test_load_customers_missing_db_says_how_to_create_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.db"):
        load_customers(tmp_path / "finns-inte.db")


def test_log_model_run_roundtrip(db_path):
    log_model_run("Testmodell", "validation", {"classifier__C": 1}, {"recall": 0.8}, db_path)
    runs = load_model_runs(db_path)
    assert len(runs) == 1
    assert runs.loc[0, "model_name"] == "Testmodell"
    assert json.loads(runs.loc[0, "metrics"]) == {"recall": 0.8}


def test_log_prediction_applies_threshold(db_path):
    log_prediction({"tenure": 3}, 0.42, threshold=0.5, db_path=db_path)
    log_prediction({"tenure": 3}, 0.42, threshold=0.3, db_path=db_path)
    logged = load_predictions(db_path)
    # Senaste först: threshold 0.3 -> churn, threshold 0.5 -> inte churn.
    assert logged["predicted_churn"].tolist() == [1, 0]
    assert json.loads(logged.loc[0, "customer_input"]) == {"tenure": 3}


def test_load_predictions_respects_limit(db_path):
    for _ in range(5):
        log_prediction({}, 0.9, 0.5, db_path)
    assert len(load_predictions(db_path, limit=2)) == 2


def test_logging_without_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.db"):
        log_prediction({}, 0.5, 0.5, tmp_path / "finns-inte.db")


def test_cli_builds_database(tmp_path, raw_df, capsys):
    csv_path = tmp_path / "raw.csv"
    raw_df.to_csv(csv_path, index=False)
    assert main(["--csv", str(csv_path), "--db", str(tmp_path / "cli.db")]) == 0
    assert "60 kunder" in capsys.readouterr().out


# --- Register för nya kunder ---


@pytest.fixture
def new_values(raw_df) -> dict:
    """Giltiga formulärvärden: första kunden i den syntetiska datan, utan id och Churn."""
    return raw_df.loc[0, RAW_FEATURE_COLUMNS].to_dict()


def test_save_new_customer_returns_running_ids(db_path, new_values):
    assert save_new_customer(new_values, db_path) == "NEW-000001"
    assert save_new_customer(new_values, db_path) == "NEW-000002"


def test_save_new_customer_never_touches_customers(db_path, new_values):
    before = len(load_customers(db_path))
    save_new_customer(new_values, db_path)
    assert len(load_customers(db_path)) == before


def test_load_new_customers_has_expected_columns_and_no_churn(db_path, new_values):
    expected = ["customerID", *RAW_FEATURE_COLUMNS, "created_at"]
    assert list(load_new_customers(db_path).columns) == expected
    assert load_new_customers(db_path).empty

    save_new_customer(new_values, db_path)
    saved = load_new_customers(db_path)
    assert list(saved.columns) == expected
    assert saved.loc[0, "customerID"] == "NEW-000001"
    assert saved.loc[0, "Contract"] == new_values["Contract"]


def test_init_db_rebuilds_customers_but_keeps_new_customers(tmp_path, raw_df, new_values):
    csv_path = tmp_path / "raw.csv"
    raw_df.to_csv(csv_path, index=False)
    db = init_db(csv_path, tmp_path / "x.db")
    save_new_customer(new_values, db)

    init_db(csv_path, db)
    assert len(load_customers(db)) == len(raw_df)
    assert load_new_customers(db)["customerID"].tolist() == ["NEW-000001"]


def test_find_customer_in_csv_data_includes_churn(db_path, raw_df):
    row = find_customer(raw_df.loc[3, "customerID"], db_path)
    assert len(row) == 1
    assert row.loc[0, "customerID"] == raw_df.loc[3, "customerID"]
    assert row.loc[0, "Churn"] in {"Yes", "No"}


def test_find_customer_in_register_has_no_churn(db_path, new_values):
    customer_id = save_new_customer(new_values, db_path)
    row = find_customer(customer_id, db_path)
    assert len(row) == 1
    assert row.loc[0, "customerID"] == customer_id
    assert row.loc[0, "tenure"] == new_values["tenure"]
    assert "Churn" not in row.columns


def test_find_customer_unknown_id_raises_key_error(db_path):
    with pytest.raises(KeyError, match="FINNS-INTE"):
        find_customer("FINNS-INTE", db_path)


def test_save_new_customer_rejects_invalid_category_without_saving(db_path, new_values):
    new_values["Contract"] = "Fem år"
    with pytest.raises(ValueError, match="Contract"):
        save_new_customer(new_values, db_path)
    assert load_new_customers(db_path).empty


def test_save_new_customer_rejects_missing_column_without_saving(db_path, new_values):
    del new_values["tenure"]
    with pytest.raises(ValueError, match="tenure"):
        save_new_customer(new_values, db_path)
    assert load_new_customers(db_path).empty


def test_save_new_customer_rejects_unknown_column_without_saving(db_path, new_values):
    new_values["Churn"] = "No"
    with pytest.raises(ValueError, match="Churn"):
        save_new_customer(new_values, db_path)
    assert load_new_customers(db_path).empty


def test_save_new_customer_rejects_non_numeric_value_without_saving(db_path, new_values):
    new_values["MonthlyCharges"] = "dyrt"
    with pytest.raises(ValueError, match="MonthlyCharges"):
        save_new_customer(new_values, db_path)
    assert load_new_customers(db_path).empty


def test_register_functions_without_db_raise(tmp_path, new_values):
    missing = tmp_path / "finns-inte.db"
    with pytest.raises(FileNotFoundError, match="src.db"):
        save_new_customer(new_values, missing)
    with pytest.raises(FileNotFoundError, match="src.db"):
        load_new_customers(missing)
    with pytest.raises(FileNotFoundError, match="src.db"):
        find_customer("NEW-000001", missing)
