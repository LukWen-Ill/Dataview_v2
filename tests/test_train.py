"""Tester för träningsflödet (src/train.py)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data import TARGET_COLUMN, split_features_target
from src.db import load_model_runs
from src.evaluate import METRIC_NAMES
from src.model import load
from src.train import main, run, select_best, split_data


@pytest.fixture
def paths(tmp_path, raw_df):
    csv_path = tmp_path / "raw.csv"
    raw_df.to_csv(csv_path, index=False)
    return {
        "data_path": csv_path,
        "db_path": tmp_path / "churn.db",
        "model_path": tmp_path / "model.joblib",
        "results_path": tmp_path / "results.json",
        "predictions_path": tmp_path / "test_predictions.csv",
    }


def test_split_is_60_20_20_and_stratified(raw_df):
    X, y = split_features_target(raw_df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    assert len(X_train) == 36 and len(X_val) == 12 and len(X_test) == 12
    # Stratifierat: churn-andelen (50 % i fixturen) är densamma i alla delar.
    assert y_train.mean() == y_val.mean() == y_test.mean() == 0.5


def test_split_parts_do_not_overlap(raw_df):
    X, y = split_features_target(raw_df)
    X_train, X_val, X_test, *_ = split_data(X, y)
    assert not set(X_train.index) & set(X_val.index)
    assert not set(X_train.index) & set(X_test.index)
    assert not set(X_val.index) & set(X_test.index)


def test_select_best_uses_validation_roc_auc():
    results = [
        {"model": "a", "validation": {"roc_auc": 0.7, "recall": 0.9}},
        {"model": "b", "validation": {"roc_auc": 0.8, "recall": 0.5}},
    ]
    assert select_best(results)["model"] == "b"


def test_run_writes_model_results_predictions_and_db_log(paths, tiny_candidates):
    summary = run(**paths, candidates=tiny_candidates)

    assert paths["model_path"].is_file()
    assert load(paths["model_path"]).named_steps["classifier"].C in (0.1, 1.0)

    on_disk = json.loads(paths["results_path"].read_text(encoding="utf-8"))
    assert on_disk["best_model"] == summary["best_model"] == "Logistisk regression"
    assert set(on_disk["test"]) == set(METRIC_NAMES)
    assert on_disk["split"] == {"n_train": 36, "n_validation": 12, "n_test": 12}
    assert "estimator" not in on_disk["models"][0]

    predictions = pd.read_csv(paths["predictions_path"])
    assert list(predictions.columns) == ["customerID", "y_true", "churn_probability"]
    assert len(predictions) == 12

    runs = load_model_runs(paths["db_path"])
    assert runs["stage"].tolist() == ["test", "validation"]


def test_run_builds_database_from_csv_when_missing(paths, tiny_candidates):
    assert not paths["db_path"].exists()
    run(**paths, candidates=tiny_candidates)
    assert paths["db_path"].is_file()


def test_run_is_deterministic(paths, tiny_candidates):
    first = run(**paths, candidates=tiny_candidates)["test"]
    second = run(**paths, candidates=tiny_candidates)["test"]
    assert first == second


def test_run_rejects_single_class_target(paths, raw_df, tiny_candidates):
    raw_df[TARGET_COLUMN] = "No"
    raw_df.to_csv(paths["data_path"], index=False)
    with pytest.raises(ValueError, match="en klass"):
        run(**paths, candidates=tiny_candidates)


def test_run_fails_loudly_on_missing_data(paths, tiny_candidates):
    paths["data_path"].unlink()
    with pytest.raises(FileNotFoundError):
        run(**paths, candidates=tiny_candidates)


def test_run_fails_loudly_on_broken_schema(paths, raw_df, tiny_candidates):
    raw_df.drop(columns=["tenure"]).to_csv(paths["data_path"], index=False)
    with pytest.raises(ValueError, match="tenure"):
        run(**paths, candidates=tiny_candidates)


def test_cli_runs_end_to_end(paths, monkeypatch, capsys, tiny_candidates):
    """CLI:t med de riktiga argumenten men ett litet grid (via monkeypatch) för snabbhet."""
    import src.train as train_module

    monkeypatch.setattr(train_module, "build_candidates", lambda random_state: tiny_candidates)
    argv = [
        "--data", str(paths["data_path"]),
        "--db", str(paths["db_path"]),
        "--out", str(paths["model_path"]),
        "--results", str(paths["results_path"]),
        "--predictions", str(paths["predictions_path"]),
    ]  # fmt: skip
    assert main(argv) == 0
    assert "Modell sparad" in capsys.readouterr().out
