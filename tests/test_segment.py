"""Tester för kundsegmenteringen."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.data import TARGET_COLUMN
from src.segment import (
    SIMPLE_FEATURES,
    fit_segments,
    kmeans_scores,
    plot_elbow_and_silhouette,
    plot_segments,
    preprocess_for_clustering,
    segment_profiles,
)


@pytest.fixture
def prepared(raw_df):
    return preprocess_for_clustering(raw_df)


def test_preprocess_gives_one_row_per_customer_and_finite_values(raw_df, prepared):
    assert prepared.shape[0] == len(raw_df)
    assert np.isfinite(prepared).all()


def test_kmeans_scores_has_one_row_per_k(prepared):
    scores = kmeans_scores(prepared, k_values=[2, 3, 4])
    assert scores["k"].tolist() == [2, 3, 4]
    # Inertia sjunker alltid när k ökar.
    assert scores["inertia"].is_monotonic_decreasing
    assert scores["silhouette"].between(-1, 1).all()


def test_kmeans_scores_rejects_too_few_rows(prepared):
    with pytest.raises(ValueError, match="För få"):
        kmeans_scores(prepared[:2])


def test_fit_segments_returns_labels_coords_and_variance(prepared):
    labels, coords, explained = fit_segments(prepared, k=3)
    assert set(labels) == {0, 1, 2}
    assert coords.shape == (len(prepared), 2)
    assert 0 < explained.sum() <= 1


def test_fit_segments_rejects_k_below_two(prepared):
    with pytest.raises(ValueError, match="minst 2"):
        fit_segments(prepared, k=1)


def test_fit_segments_is_deterministic(prepared):
    first, _, _ = fit_segments(prepared, k=3, random_state=1)
    second, _, _ = fit_segments(prepared, k=3, random_state=1)
    assert (first == second).all()


def test_segment_profiles_cover_every_customer(raw_df, prepared):
    labels, _, _ = fit_segments(prepared, k=3)
    profiles = segment_profiles(raw_df, labels)
    assert profiles["antal_kunder"].sum() == len(raw_df)
    assert profiles["churn_andel"].between(0, 1).all()
    assert set(profiles["vanligaste_avtal"]) <= {"Month-to-month", "One year", "Two year"}


def test_segment_profiles_work_without_target(raw_df, prepared):
    labels, _, _ = fit_segments(prepared, k=2)
    profiles = segment_profiles(raw_df.drop(columns=[TARGET_COLUMN]), labels)
    assert "churn_andel" not in profiles.columns


def test_plots_return_figures(prepared):
    scores = kmeans_scores(prepared, k_values=[2, 3])
    labels, coords, explained = fit_segments(prepared, k=2)
    assert isinstance(plot_elbow_and_silhouette(scores), Figure)
    assert isinstance(plot_segments(coords, labels, explained), Figure)


def test_preprocess_with_columns_returns_only_those(raw_df):
    X = preprocess_for_clustering(raw_df, SIMPLE_FEATURES)
    assert X.shape == (len(raw_df), len(SIMPLE_FEATURES))
    assert X.mean(axis=0) == pytest.approx(0, abs=1e-6)
