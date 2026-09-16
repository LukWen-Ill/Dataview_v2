"""Tester för utvärderingsmått, threshold och feature importance."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.data import split_features_target
from src.evaluate import (
    METRIC_NAMES,
    apply_threshold,
    compute_metrics,
    feature_importance,
    feature_names,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    report,
    threshold_table,
)
from src.models import build_pipeline

Y_TRUE = np.array([0, 0, 0, 0, 1, 1, 1, 1])
PROBA = np.array([0.1, 0.2, 0.6, 0.4, 0.9, 0.8, 0.3, 0.55])


def test_apply_threshold_uses_greater_or_equal():
    assert apply_threshold([0.5, 0.49], 0.5).tolist() == [1, 0]


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_apply_threshold_rejects_values_outside_unit_interval(bad):
    with pytest.raises(ValueError, match="mellan 0 och 1"):
        apply_threshold(PROBA, bad)


def test_compute_metrics_returns_all_five_in_unit_interval():
    metrics = compute_metrics(Y_TRUE, PROBA)
    assert set(metrics) == set(METRIC_NAMES)
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_compute_metrics_by_hand():
    # threshold 0.5 -> pred = [0,0,1,0,1,1,0,1]: TP=3, FP=1, FN=1, TN=3
    metrics = compute_metrics(Y_TRUE, PROBA, 0.5)
    assert metrics["accuracy"] == pytest.approx(6 / 8)
    assert metrics["precision"] == pytest.approx(3 / 4)
    assert metrics["recall"] == pytest.approx(3 / 4)


def test_roc_auc_does_not_depend_on_threshold():
    assert (
        compute_metrics(Y_TRUE, PROBA, 0.3)["roc_auc"]
        == compute_metrics(Y_TRUE, PROBA, 0.7)["roc_auc"]
    )


def test_lower_threshold_never_lowers_recall():
    table = threshold_table(Y_TRUE, PROBA, [0.3, 0.5, 0.7])
    recalls = table["recall"].tolist()
    assert recalls == sorted(recalls, reverse=True)
    assert table["flaggade_kunder"].tolist() == sorted(table["flaggade_kunder"], reverse=True)


def test_threshold_table_counts_missed_churners():
    table = threshold_table(Y_TRUE, PROBA, [0.5])
    assert table.loc[0, "missade_churnare"] == 1


def test_compute_metrics_with_single_class_raises():
    with pytest.raises(ValueError):
        compute_metrics(np.zeros(4, dtype=int), PROBA[:4])


def test_report_names_both_classes():
    text = report(Y_TRUE, PROBA)
    assert "Stannar" in text and "Churnar" in text


def test_plots_return_figures():
    assert isinstance(plot_confusion_matrix(Y_TRUE, PROBA), Figure)
    assert isinstance(plot_roc_curve(Y_TRUE, PROBA, "x"), Figure)


def test_feature_names_have_no_transformer_prefix(fitted_pipeline):
    names = feature_names(fitted_pipeline)
    assert not any(n.startswith(("num__", "cat__")) for n in names)
    assert "Contract_Month-to-month" in names
    assert "num_addon_services" in names


def test_logistic_regression_gives_coefficients(fitted_pipeline):
    table = feature_importance(fitted_pipeline)
    assert (table["kind"] == "coefficient").all()
    assert len(table) == len(feature_names(fitted_pipeline))
    # Sorterad på absolutvärde, störst först.
    assert table["value"].abs().is_monotonic_decreasing


def test_random_forest_gives_importances(raw_df):
    X, y = split_features_target(raw_df)
    pipeline = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=0)).fit(X, y)
    table = feature_importance(pipeline)
    assert (table["kind"] == "importance").all()
    assert table["value"].sum() == pytest.approx(1.0)
    assert isinstance(plot_feature_importance(table, top_n=5), Figure)


def test_model_without_importance_raises(raw_df):
    X, y = split_features_target(raw_df)
    pipeline = build_pipeline(SVC()).fit(X, y)
    with pytest.raises(ValueError, match="saknar"):
        feature_importance(pipeline)
