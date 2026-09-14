"""Tester för CLI-skriptet."""

from __future__ import annotations

import json

import pytest

from src.data import SchemaError
from src.train import main


def test_main_writes_model_and_prints_metrics(tmp_path, raw_df, capsys):
    data = tmp_path / "d.csv"
    raw_df.to_csv(data, index=False)
    out = tmp_path / "m.joblib"

    assert main(["--data", str(data), "--out", str(out)]) == 0
    assert out.is_file()

    printed = capsys.readouterr().out
    metrics = json.loads(printed[printed.index("{") : printed.index("}") + 1])
    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "n_train",
        "n_test",
    }


def test_main_fails_loudly_on_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        main(["--data", str(tmp_path / "finns-inte.csv"), "--out", str(tmp_path / "m.joblib")])


def test_main_fails_loudly_on_broken_schema(tmp_path, raw_df):
    data = tmp_path / "trasig.csv"
    raw_df.drop(columns=["tenure"]).to_csv(data, index=False)
    with pytest.raises(SchemaError, match="tenure"):
        main(["--data", str(data), "--out", str(tmp_path / "m.joblib")])
