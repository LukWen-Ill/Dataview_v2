"""SQLite-databas med kunddatan, loggade modellkörningar och loggade prediktioner.

Kör `python -m src.db` för att bygga data/churn.db från rå-CSV:n.

Tre tabeller:
    customers    rådatan, en rad per kund (fylls från data/raw/telco_churn.csv)
    model_runs   en rad per utvärderad modell när python -m src.train körs
    predictions  en rad per prediktion som görs i appen
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data import DEFAULT_DATA_PATH, load_raw, validate

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "churn.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    model_name  TEXT NOT NULL,
    stage       TEXT NOT NULL,   -- 'validation' (modelljämförelse) eller 'test' (slutmodell)
    best_params TEXT NOT NULL,   -- JSON
    metrics     TEXT NOT NULL    -- JSON
);

CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT NOT NULL,
    customer_input    TEXT NOT NULL,   -- JSON med värdena användaren fyllde i
    churn_probability REAL NOT NULL,
    threshold         REAL NOT NULL,
    predicted_churn   INTEGER NOT NULL
);
"""


def _connect(db_path: Path):
    """Öppna en anslutning som stängs automatiskt efter with-blocket."""
    return closing(sqlite3.connect(db_path))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_db(
    csv_path: str | Path = DEFAULT_DATA_PATH, db_path: str | Path = DEFAULT_DB_PATH
) -> Path:
    """Bygg databasen från CSV:n. Skriver över tabellen customers om den redan finns."""
    df = load_raw(csv_path)
    validate(df)

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        df.to_sql("customers", conn, if_exists="replace", index=False)
        conn.executescript(SCHEMA)
        conn.commit()
    return db_path


def _require_db(db_path: Path) -> None:
    if not db_path.is_file():
        raise FileNotFoundError(f"Ingen databas på {db_path}. Kör: python -m src.db")


def load_customers(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Läs hela kundtabellen."""
    db_path = Path(db_path)
    _require_db(db_path)
    with _connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM customers", conn)


def log_model_run(
    model_name: str,
    stage: str,
    best_params: dict,
    metrics: dict,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Spara resultatet av en modellutvärdering."""
    db_path = Path(db_path)
    _require_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO model_runs (created_at, model_name, stage, best_params, metrics) "
            "VALUES (?, ?, ?, ?, ?)",
            (_now(), model_name, stage, json.dumps(best_params), json.dumps(metrics)),
        )
        conn.commit()


def log_prediction(
    customer_input: dict,
    churn_probability: float,
    threshold: float,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Spara en prediktion från appen."""
    db_path = Path(db_path)
    _require_db(db_path)
    predicted_churn = int(churn_probability >= threshold)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO predictions "
            "(created_at, customer_input, churn_probability, threshold, predicted_churn) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                _now(),
                json.dumps(customer_input, default=str),
                float(churn_probability),
                float(threshold),
                predicted_churn,
            ),
        )
        conn.commit()


def load_model_runs(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Alla loggade modellkörningar, senaste först."""
    db_path = Path(db_path)
    _require_db(db_path)
    with _connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM model_runs ORDER BY id DESC", conn)


def load_predictions(db_path: str | Path = DEFAULT_DB_PATH, limit: int = 50) -> pd.DataFrame:
    """De senaste loggade prediktionerna."""
    db_path = Path(db_path)
    _require_db(db_path)
    with _connect(db_path) as conn:
        return pd.read_sql(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", conn, params=(limit,)
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bygg SQLite-databasen från rå-CSV:n.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args(argv)

    path = init_db(args.csv, args.db)
    n_rows = len(load_customers(path))
    print(f"Databas skapad: {path} ({n_rows} kunder i tabellen customers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
