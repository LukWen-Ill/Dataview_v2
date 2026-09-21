"""Kundsegmentering med K-Means, visualiserad med PCA.

Detta är icke-vägledd inlärning: vi använder inte Churn-kolumnen när klustren skapas.
Churn-modellen svarar på "vilka kunder riskerar att lämna?", segmenteringen på
"vilka typer av kunder liknar varandra?". Efteråt kan vi titta på churn-andelen per
segment - det är ett sätt att tolka klustren, inte något modellen tränats på.

Flöde: features -> samma förbehandling som churn-modellen (skalning + one-hot)
       -> K-Means (klustring) -> PCA till 2 dimensioner (bara för att kunna rita).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data import TARGET_COLUMN, prepare_features
from src.features import build_preprocessor
from src.models import RANDOM_STATE

DEFAULT_K = 4
K_RANGE = range(2, 9)
SILHOUETTE_SAMPLE_SIZE = 2000  # silhouette är O(n^2); ett urval räcker för att välja k
SIMPLE_FEATURES = ["tenure", "MonthlyCharges", "num_addon_services"]


def preprocess_for_clustering(df: pd.DataFrame, columns=None) -> np.ndarray:
    X = prepare_features(df)
    if columns is None:
        return build_preprocessor().fit_transform(X)
    return StandardScaler().fit_transform(X[columns])


def kmeans_scores(
    X_prepared: np.ndarray, k_values=K_RANGE, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Inertia (för elbow-metoden) och silhouette score för varje antal kluster."""
    if len(X_prepared) < 3:
        raise ValueError("För få rader för att klustra.")
    rows = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X_prepared)
        sample = min(SILHOUETTE_SAMPLE_SIZE, len(X_prepared))
        rows.append(
            {
                "k": k,
                "inertia": float(kmeans.inertia_),
                "silhouette": float(
                    silhouette_score(
                        X_prepared, kmeans.labels_, sample_size=sample, random_state=random_state
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def fit_segments(
    X_prepared: np.ndarray, k: int = DEFAULT_K, random_state: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Klustra och projicera till 2D.

    Returnerar (segment per kund, PCA-koordinater (n, 2), förklarad varians per komponent).
    """
    if k < 2:
        raise ValueError(f"Antalet kluster måste vara minst 2, fick {k}")
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X_prepared)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_prepared)
    return kmeans.labels_, coords, pca.explained_variance_ratio_


def segment_profiles(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Beskriv varje segment med några få begripliga siffror."""
    data = prepare_features(df).assign(segment=labels)
    data["Contract"] = df["Contract"].to_numpy()
    if TARGET_COLUMN in df.columns:
        data["churn"] = (df[TARGET_COLUMN] == "Yes").astype(int).to_numpy()

    profile = data.groupby("segment").agg(
        antal_kunder=("tenure", "size"),
        snitt_tenure=("tenure", "mean"),
        snitt_manadskostnad=("MonthlyCharges", "mean"),
        snitt_tillaggstjanster=("num_addon_services", "mean"),
        vanligaste_avtal=("Contract", lambda s: s.mode().iloc[0]),
    )
    if "churn" in data.columns:
        profile["churn_andel"] = data.groupby("segment")["churn"].mean()
    return profile.round(2).reset_index()


def plot_elbow_and_silhouette(scores: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(9, 3.5))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(scores["k"], scores["inertia"], "o-")
    ax1.set_xlabel("Antal kluster (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow-metoden")
    ax2.plot(scores["k"], scores["silhouette"], "o-", color="tab:green")
    ax2.set_xlabel("Antal kluster (k)")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette (högre = tydligare kluster)")
    fig.tight_layout()
    return fig


def plot_segments(coords: np.ndarray, labels: np.ndarray, explained: np.ndarray) -> Figure:
    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    for segment in np.unique(labels):
        mask = labels == segment
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.5, label=f"Segment {segment}")
    ax.set_xlabel(f"PC1 ({explained[0]:.0%} av variansen)")
    ax.set_ylabel(f"PC2 ({explained[1]:.0%} av variansen)")
    ax.set_title("Kundsegment i PCA-rummet")
    ax.legend(markerscale=2)
    return fig
