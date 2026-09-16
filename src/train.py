"""Träningsskript - hela ML-flödet från databas till sparad modell. Kör: python -m src.train

    1. Läs kunddata från SQLite (databasen byggs från CSV:n om den saknas).
    2. Rensa + feature engineering -> X, y.
    3. Dela stratifierat i train (60 %) / validation (20 %) / test (20 %).
    4. För varje modell: GridSearchCV med 5-delad korsvalidering på train.
    5. Jämför modellerna på validation och välj den bästa.
    6. Träna om den valda modellen på train + validation.
    7. Utvärdera EN gång på test - testdatan har inte rörts innan dess.
    8. Spara modell (joblib), resultat (JSON), testprediktioner (CSV) och logga i databasen.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.data import DEFAULT_DATA_PATH, ID_COLUMN, split_features_target, validate
from src.db import DEFAULT_DB_PATH, init_db, load_customers, log_model_run
from src.evaluate import compute_metrics
from src.model import DEFAULT_MODEL_PATH, MODELS_DIR, save
from src.models import RANDOM_STATE, build_candidates

RESULTS_PATH = MODELS_DIR / "results.json"
TEST_PREDICTIONS_PATH = MODELS_DIR / "test_predictions.csv"

TEST_SIZE = 0.2  # 20 % av allt
VALIDATION_SIZE = 0.25  # 25 % av de återstående 80 % = 20 % av allt
CV_FOLDS = 5
SCORING = "roc_auc"  # för GridSearchCV: oberoende av threshold, bra vid obalanserade klasser
SELECTION_METRIC = "roc_auc"  # för att välja mellan modellerna på validation


def split_data(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE):
    """Stratifierad 60/20/20-split. Stratify gör att churn-andelen blir lika i alla tre delar."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VALIDATION_SIZE, stratify=y_trainval, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def tune(pipeline: Pipeline, grid: dict, X_train, y_train, random_state: int) -> GridSearchCV:
    """Hitta bästa hyperparametrarna med k-delad korsvalidering - enbart på träningsdatan."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
    search = GridSearchCV(pipeline, grid, cv=cv, scoring=SCORING, n_jobs=-1)
    search.fit(X_train, y_train)
    return search


def compare_models(candidates: dict, X_train, y_train, X_val, y_val, random_state: int) -> list[dict]:
    """Tuna varje modell på train och utvärdera den på validation."""
    results = []
    for name, (pipeline, grid) in candidates.items():
        search = tune(pipeline, grid, X_train, y_train, random_state)
        val_proba = search.predict_proba(X_val)[:, 1]
        results.append(
            {
                "model": name,
                "best_params": search.best_params_,
                "cv_roc_auc": float(search.best_score_),
                "validation": compute_metrics(y_val, val_proba),
                "estimator": search.best_estimator_,
            }
        )
        print(f"{name:22s} CV ROC-AUC {search.best_score_:.3f}  val: {_fmt(results[-1]['validation'])}")
    return results


def select_best(results: list[dict]) -> dict:
    """Välj modellen med högst SELECTION_METRIC på validation."""
    return max(results, key=lambda r: r["validation"][SELECTION_METRIC])


def run(
    data_path: Path = DEFAULT_DATA_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    results_path: Path = RESULTS_PATH,
    predictions_path: Path = TEST_PREDICTIONS_PATH,
    random_state: int = RANDOM_STATE,
    candidates: dict | None = None,
) -> dict:
    """Kör hela flödet. Returnerar samma innehåll som skrivs till results.json."""
    # 1-2. Data från databasen -> X, y
    if not Path(db_path).is_file():
        init_db(data_path, db_path)
    df = load_customers(db_path)
    validate(df)
    X, y = split_features_target(df)
    if y.nunique() < 2:
        raise ValueError("Targeten har bara en klass - modellen går inte att träna.")

    # 3. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, random_state)
    print(f"Split: train {len(X_train)}, validation {len(X_val)}, test {len(X_test)}")

    # 4-5. Tuna och jämför
    if candidates is None:
        candidates = build_candidates(random_state)
    results = compare_models(candidates, X_train, y_train, X_val, y_val, random_state)
    best = select_best(results)
    print(f"Bäst på validation ({SELECTION_METRIC}): {best['model']}")

    # 6. Träna om den valda modellen på train + validation
    final_model = clone(best["estimator"])
    final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    # 7. Slutlig utvärdering på test - första och enda gången testdatan används
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_proba)
    print(f"Test: {_fmt(test_metrics)}")

    # 8. Spara allt
    save(final_model, model_path)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "random_state": random_state,
        "cv_folds": CV_FOLDS,
        "scoring": SCORING,
        "selection_metric": SELECTION_METRIC,
        "split": {"n_train": len(X_train), "n_validation": len(X_val), "n_test": len(X_test)},
        "models": [{k: v for k, v in r.items() if k != "estimator"} for r in results],
        "best_model": best["model"],
        "test": test_metrics,
    }
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(results_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(
        {
            ID_COLUMN: df.loc[X_test.index, ID_COLUMN].to_numpy(),
            "y_true": y_test.to_numpy(),
            "churn_probability": test_proba,
        }
    ).to_csv(predictions_path, index=False)

    for r in results:
        log_model_run(r["model"], "validation", r["best_params"], r["validation"], db_path)
    log_model_run(best["model"], "test", best["best_params"], test_metrics, db_path)

    print(f"Modell sparad: {model_path}")
    return summary


def _fmt(metrics: dict) -> str:
    return "  ".join(f"{k} {v:.3f}" for k, v in metrics.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Träna och jämför churn-modellerna.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--predictions", type=Path, default=TEST_PREDICTIONS_PATH)
    args = parser.parse_args(argv)

    run(args.data, args.db, args.out, args.results, args.predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
