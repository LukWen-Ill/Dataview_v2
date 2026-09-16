"""Tester för SQLite-databasen."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pytest

from src.data import REQUIRED_COLUMNS, SchemaError
from src.db import (
    init_db,
    load_customers,
    load_model_runs,
    load_predictions,
    log_model_run,
    log_prediction,
    main,
)


def test_init_db_creates_file_and_all_tables(db_path):
    assert db_path.is_file()
    with closing(sqlite3.connect(db_path)) as conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"customers", "model_runs", "predictions"} <= tables


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
