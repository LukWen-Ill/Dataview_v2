"""Prismodell (extra): regression som predikterar MonthlyCharges. Kör: python -m src.price

Används i Säljverktyget för att ge en ny kund ett pris innan churnrisken räknas, och
för att visa pris på alternativ ur åtgärdskatalogen. Flödet speglar src/train.py:

1. Läs kunddata från SQLite.
2. Features = kundens konfiguration (tjänster, avtal, betalsätt, demografi, tenure).
   Inte TotalCharges eller avg_monthly_charge - båda är funktioner av targeten.
3. Samma rader i train/validation/test som churnmodellen (split_data, stratifierat på Churn).
4. Linjär regression och random forest i Pipeline, små grids i GridSearchCV, 5-delad CV.
5. Välj på validation med R². Träna om på train + validation. Utvärdera EN gång på test.
6. Spara modell (joblib) och resultat (JSON), logga i databasen.

Förvänta R² nära 1: priset i datan är nästan en summa av tjänsterna. Det är inte en bugg.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import (
    RAW_CATEGORICAL_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_MAP,
    clean,
    validate,
)
from src.db import DEFAULT_DB_PATH, load_customers, log_model_run
from src.models import RANDOM_STATE
from src.train import CV_FOLDS, split_data

ROOT = Path(__file__).resolve().parents[1]
PRICE_MODEL_PATH = ROOT / "models" / "price_model.joblib"
PRICE_RESULTS_PATH = ROOT / "models" / "price_results.json"

PRICE_TARGET = "MonthlyCharges"
# Kolumner som är funktioner av targeten och därför aldrig får vara features.
LEAKING_COLUMNS = ["MonthlyCharges", "TotalCharges"]
PRICE_FEATURE_COLUMNS = [col for col in RAW_FEATURE_COLUMNS if col not in LEAKING_COLUMNS]

SCORING = "r2"  # för GridSearchCV och för att välja modell på validation
METRIC_NAMES = ["mae", "rmse", "r2"]


def build_price_preprocessor(
    feature_columns: list[str] = PRICE_FEATURE_COLUMNS,
) -> ColumnTransformer:
    """One-hot på kategorier, skalning på numeriska (tenure, SeniorCitizen).

    Kastar ValueError om en läckande kolumn (funktion av targeten) finns i feature-listan.
    """
    leaking = [col for col in feature_columns if col in LEAKING_COLUMNS]
    if leaking:
        raise ValueError(f"Läckande kolumner i feature-listan: {leaking}")
    numeric = [col for col in feature_columns if col not in RAW_CATEGORICAL_COLUMNS]
    categorical = [col for col in feature_columns if col in RAW_CATEGORICAL_COLUMNS]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )


def build_price_pipeline(regressor, feature_columns: list[str] = PRICE_FEATURE_COLUMNS) -> Pipeline:
    """Förbehandling + regressor i en enda pipeline."""
    return Pipeline(
        [("preprocess", build_price_preprocessor(feature_columns)), ("regressor", regressor)]
    )


def build_price_candidates(random_state: int = RANDOM_STATE) -> dict[str, tuple[Pipeline, dict]]:
    """Modellnamn -> (pipeline, hyperparametergrid). Små grids, som i src/models.py."""
    return {
        "Prismodell: Linjär regression": (
            build_price_pipeline(LinearRegression()),
            # positive=True tvingar alla koefficienter >= 0: varje tjänst kan bara höja priset.
            {"regressor__positive": [False, True]},
        ),
        "Prismodell: Random forest": (
            build_price_pipeline(RandomForestRegressor(random_state=random_state)),
            {
                "regressor__n_estimators": [100, 200],
                "regressor__max_depth": [10, 20],
            },
        ),
    }


def compute_price_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE och RMSE i dollar/månad, R² som andel förklarad varians."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def split_price_data(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """Samma 60/20/20-rader som churnmodellen: split_data stratifierar på Churn, inte på priset.

    Radindelningen beror bara på antal rader, ordning, random_state och stratifieringsvariabeln,
    så att X har andra kolumner än i churnflödet spelar ingen roll. Priset hämtas efteråt via index.
    """
    X = df[PRICE_FEATURE_COLUMNS]
    churn = df[TARGET_COLUMN].map(TARGET_MAP).astype(int)
    X_train, X_val, X_test, *_ = split_data(X, churn, random_state)
    y = df[PRICE_TARGET]
    return X_train, X_val, X_test, y.loc[X_train.index], y.loc[X_val.index], y.loc[X_test.index]


def train_price_model(
    db_path: Path = DEFAULT_DB_PATH,
    random_state: int = RANDOM_STATE,
    model_path: Path = PRICE_MODEL_PATH,
    results_path: Path = PRICE_RESULTS_PATH,
    candidates: dict | None = None,
) -> dict:
    """Kör hela flödet. Returnerar samma innehåll som skrivs till price_results.json."""
    # 1-3. Data -> samma split som churnmodellen
    df = clean(load_customers(db_path))
    validate(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_price_data(df, random_state)
    print(f"Split: train {len(X_train)}, validation {len(X_val)}, test {len(X_test)}")

    # 4-5. Tuna varje modell på train med k-delad CV, jämför på validation
    if candidates is None:
        candidates = build_price_candidates(random_state)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    results = []
    for name, (pipeline, grid) in candidates.items():
        search = GridSearchCV(pipeline, grid, cv=cv, scoring=SCORING, n_jobs=-1)
        search.fit(X_train, y_train)
        val_metrics = compute_price_metrics(y_val, search.predict(X_val))
        results.append(
            {
                "model": name,
                "best_params": search.best_params_,
                "cv_r2": float(search.best_score_),
                "validation": val_metrics,
                "estimator": search.best_estimator_,
            }
        )
        print(f"{name:32s} CV R² {search.best_score_:.3f}  val: {_fmt(val_metrics)}")
    best = max(results, key=lambda r: r["validation"][SCORING])
    print(f"Bäst på validation ({SCORING}): {best['model']}")

    # Träna om den valda modellen på train + validation
    final_model = clone(best["estimator"])
    final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    # Slutlig utvärdering på test - första och enda gången testdatan används
    test_metrics = compute_price_metrics(y_test, final_model.predict(X_test))
    print(f"Test: {_fmt(test_metrics)}")

    # 6. Spara och logga
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": random_state,
        "cv_folds": CV_FOLDS,
        "scoring": SCORING,
        "target": PRICE_TARGET,
        "feature_columns": PRICE_FEATURE_COLUMNS,
        "split": {"n_train": len(X_train), "n_validation": len(X_val), "n_test": len(X_test)},
        "models": [{k: v for k, v in r.items() if k != "estimator"} for r in results],
        "best_model": best["model"],
        "test": test_metrics,
    }
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(results_path).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for r in results:
        log_model_run(r["model"], "validation", r["best_params"], r["validation"], db_path)
    log_model_run(best["model"], "test", best["best_params"], test_metrics, db_path)

    print(f"Prismodell sparad: {model_path}")
    return summary


def load_price_model(path: str | Path = PRICE_MODEL_PATH) -> Pipeline:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ingen tränad prismodell på {path}. Kör: python -m src.price")
    return joblib.load(path)


def predict_price(pipeline: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Predikterat månadspris i dollar per rad. Extra kolumner i df ignoreras."""
    missing = [col for col in PRICE_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Kolumner saknas: {missing}")
    prices = pipeline.predict(df[PRICE_FEATURE_COLUMNS])
    return pd.Series(prices, index=df.index, name="predicted_monthly_charges")


def _fmt(metrics: dict) -> str:
    return "  ".join(f"{k} {v:.3f}" for k, v in metrics.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Träna prismodellen (regression på MonthlyCharges)."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--out", type=Path, default=PRICE_MODEL_PATH)
    parser.add_argument("--results", type=Path, default=PRICE_RESULTS_PATH)
    args = parser.parse_args(argv)

    train_price_model(args.db, model_path=args.out, results_path=args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
