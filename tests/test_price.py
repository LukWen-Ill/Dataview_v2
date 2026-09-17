"""Tester för prismodellen (src/price.py)."""

from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.data import (
    ADDON_SERVICES,
    RAW_CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    split_features_target,
)
from src.db import load_model_runs
from src.price import (
    METRIC_NAMES,
    PRICE_FEATURE_COLUMNS,
    PRICE_MODEL_PATH,
    PRICE_TARGET,
    build_price_candidates,
    build_price_pipeline,
    build_price_preprocessor,
    compute_price_metrics,
    load_price_model,
    main,
    predict_price,
    split_price_data,
    train_price_model,
)
from src.train import split_data


@pytest.fixture
def price_pipeline(raw_df):
    """En snabbt tränad prispipeline (utan GridSearch)."""
    pipeline = build_price_pipeline(LinearRegression())
    pipeline.fit(raw_df[PRICE_FEATURE_COLUMNS], raw_df[PRICE_TARGET])
    return pipeline


@pytest.fixture
def tiny_price_candidates():
    """En modell med ett minimalt grid så att hela flödet kan testas på sekunder."""
    pipeline = build_price_pipeline(LinearRegression())
    return {"Prismodell: Linjär regression": (pipeline, {"regressor__positive": [False, True]})}


@pytest.fixture
def paths(tmp_path):
    return {"model_path": tmp_path / "price.joblib", "results_path": tmp_path / "price.json"}


# --- Feature-lista och guards ---


def test_feature_columns_exclude_leaking_columns():
    assert "TotalCharges" not in PRICE_FEATURE_COLUMNS
    assert "MonthlyCharges" not in PRICE_FEATURE_COLUMNS
    assert "avg_monthly_charge" not in PRICE_FEATURE_COLUMNS
    assert "tenure" in PRICE_FEATURE_COLUMNS
    assert set(RAW_CATEGORICAL_COLUMNS) <= set(PRICE_FEATURE_COLUMNS)


@pytest.mark.parametrize("leak", ["TotalCharges", "MonthlyCharges"])
def test_preprocessor_rejects_leaking_column(leak):
    with pytest.raises(ValueError, match=leak):
        build_price_preprocessor([*PRICE_FEATURE_COLUMNS, leak])


def test_candidates_are_linear_regression_and_random_forest():
    candidates = build_price_candidates(random_state=0)
    assert list(candidates) == ["Prismodell: Linjär regression", "Prismodell: Random forest"]
    for pipeline, grid in candidates.values():
        assert list(pipeline.named_steps) == ["preprocess", "regressor"]
        assert all(key.startswith("regressor__") for key in grid)


# --- Split: samma rader som churnmodellen ---


def test_split_uses_same_rows_as_churn_model(raw_df):
    X, y = split_features_target(raw_df)
    churn_train, churn_val, churn_test, *_ = split_data(X, y)
    X_train, X_val, X_test, y_train, y_val, y_test = split_price_data(raw_df)
    assert list(X_train.index) == list(churn_train.index)
    assert list(X_val.index) == list(churn_val.index)
    assert list(X_test.index) == list(churn_test.index)
    # Priset följer med raden.
    assert (y_test == raw_df.loc[X_test.index, PRICE_TARGET]).all()
    assert list(X_train.columns) == PRICE_FEATURE_COLUMNS


# --- Prediktion ---


def test_predict_price_keeps_index_and_ignores_extra_columns(price_pipeline, raw_df):
    """Rådata med Churn och TotalCharges ska gå att prediktera på - extra kolumner ignoreras."""
    subset = raw_df.iloc[[3, 9, 15]]
    prices = predict_price(price_pipeline, subset)
    assert list(prices.index) == [3, 9, 15]
    assert prices.name == "predicted_monthly_charges"
    assert len(prices) == 3


def test_predict_price_ignores_column_order(price_pipeline, raw_df):
    row = raw_df.iloc[[0]]
    shuffled = row[list(reversed(row.columns))]
    assert predict_price(price_pipeline, shuffled).iloc[0] == pytest.approx(
        predict_price(price_pipeline, row).iloc[0]
    )


def test_predict_price_survives_unknown_category(price_pipeline, raw_df):
    unseen = raw_df.iloc[[0]].copy()
    unseen.loc[:, "PaymentMethod"] = "Swish"
    assert predict_price(price_pipeline, unseen).notna().all()


def test_predict_price_rejects_missing_column(price_pipeline, raw_df):
    with pytest.raises(KeyError, match="Contract"):
        predict_price(price_pipeline, raw_df.drop(columns=["Contract"]))


def test_compute_price_metrics_perfect_prediction():
    metrics = compute_price_metrics([10.0, 20.0, 30.0], [10.0, 20.0, 30.0])
    assert list(metrics) == METRIC_NAMES
    assert metrics == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


# --- Träningsflödet ---


def test_train_writes_model_results_and_db_log(db_path, paths, tiny_price_candidates):
    summary = train_price_model(db_path, **paths, candidates=tiny_price_candidates)

    assert paths["model_path"].is_file()
    on_disk = json.loads(paths["results_path"].read_text(encoding="utf-8"))
    assert on_disk["best_model"] == summary["best_model"] == "Prismodell: Linjär regression"
    assert on_disk["target"] == PRICE_TARGET
    assert on_disk["feature_columns"] == PRICE_FEATURE_COLUMNS
    assert set(on_disk["test"]) == set(METRIC_NAMES)
    assert on_disk["split"] == {"n_train": 36, "n_validation": 12, "n_test": 12}
    assert on_disk["random_state"] == 42
    assert "created_at" in on_disk
    assert "estimator" not in on_disk["models"][0]
    assert on_disk["models"][0]["best_params"]["regressor__positive"] in (True, False)

    runs = load_model_runs(db_path)
    assert runs["stage"].tolist() == ["test", "validation"]
    assert runs["model_name"].tolist() == ["Prismodell: Linjär regression"] * 2

    loaded = load_price_model(paths["model_path"])
    assert loaded.named_steps["regressor"].positive in (True, False)


def test_train_is_deterministic(db_path, paths, tiny_price_candidates):
    first = train_price_model(db_path, **paths, candidates=tiny_price_candidates)["test"]
    second = train_price_model(db_path, **paths, candidates=tiny_price_candidates)["test"]
    assert first == second


def test_train_fails_loudly_on_missing_db(tmp_path, paths, tiny_price_candidates):
    with pytest.raises(FileNotFoundError, match="src.db"):
        train_price_model(tmp_path / "finns-inte.db", **paths, candidates=tiny_price_candidates)


def test_train_fails_loudly_on_broken_schema(tmp_path, raw_df, paths, tiny_price_candidates):
    """Utan Churn går det inte att göra samma split som churnmodellen - ska fallera tydligt."""
    broken_db = tmp_path / "broken.db"
    with sqlite3.connect(broken_db) as conn:
        raw_df.drop(columns=[TARGET_COLUMN]).to_sql("customers", conn, index=False)
    with pytest.raises(ValueError, match=TARGET_COLUMN):
        train_price_model(broken_db, **paths, candidates=tiny_price_candidates)


def test_load_missing_model_raises_with_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.price"):
        load_price_model(tmp_path / "finns-inte.joblib")


def test_cli_runs_end_to_end(db_path, paths, monkeypatch, capsys, tiny_price_candidates):
    """CLI:t med de riktiga argumenten men ett litet grid (via monkeypatch) för snabbhet."""
    import src.price as price_module

    monkeypatch.setattr(
        price_module, "build_price_candidates", lambda random_state: tiny_price_candidates
    )
    argv = [
        "--db", str(db_path),
        "--out", str(paths["model_path"]),
        "--results", str(paths["results_path"]),
    ]  # fmt: skip
    assert main(argv) == 0
    assert "Prismodell sparad" in capsys.readouterr().out


# --- Den incheckade modellen: enda testet som beror på models/price_model.joblib ---


def test_checked_in_model_loads_and_prices_archetypes():
    """Fiber med alla tillägg och flera linjer ~115 dollar (datan: 111-119), enbart telefoni ~20."""
    assert PRICE_MODEL_PATH.is_file(), f"Prismodellen saknas: {PRICE_MODEL_PATH}"
    pipeline = load_price_model()
    fiber_everything = {
        "SeniorCitizen": 0,
        "tenure": 0,
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }
    phone_only = {
        **fiber_everything,
        "MultipleLines": "No",
        "InternetService": "No",
        **{col: "No internet service" for col in ADDON_SERVICES},
    }
    prices = predict_price(pipeline, pd.DataFrame([fiber_everything, phone_only]))
    assert prices.iloc[0] == pytest.approx(115, abs=5)
    assert prices.iloc[1] == pytest.approx(20, abs=3)
