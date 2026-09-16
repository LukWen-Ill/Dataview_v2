"""Utvärderingsmått, threshold-analys, feature importance och grafer.

Alla funktioner utgår från *sannolikheter* (predict_proba) snarare än färdiga klasser,
så att threshold kan väljas efteråt - det är själva poängen med threshold-analysen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

DEFAULT_THRESHOLD = 0.5
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]
CLASS_NAMES = ["Stannar", "Churnar"]


def apply_threshold(proba, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Sannolikhet -> klass. Kund flaggas som churnare om sannolikheten >= threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold måste ligga mellan 0 och 1, fick {threshold}")
    return (np.asarray(proba) >= threshold).astype(int)


def compute_metrics(y_true, proba, threshold: float = DEFAULT_THRESHOLD) -> dict[str, float]:
    """De fem måtten vi jämför modeller på. ROC-AUC beror inte på threshold, övriga gör det."""
    y_pred = apply_threshold(proba, threshold)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def threshold_table(y_true, proba, thresholds: list[float] = THRESHOLDS) -> pd.DataFrame:
    """Hur precision, recall och antal flaggade kunder ändras när threshold ändras.

    Lägre threshold -> fler kunder flaggas -> högre recall men lägre precision.
    """
    y_true = np.asarray(y_true)
    rows = []
    for t in thresholds:
        y_pred = apply_threshold(proba, t)
        rows.append(
            {
                "threshold": t,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "flaggade_kunder": int(y_pred.sum()),
                "missade_churnare": int(((y_pred == 0) & (y_true == 1)).sum()),
            }
        )
    return pd.DataFrame(rows)


def report(y_true, proba, threshold: float = DEFAULT_THRESHOLD) -> str:
    """classification_report som text, för att visas i appen och rapporten."""
    y_pred = apply_threshold(proba, threshold)
    return classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)


def plot_confusion_matrix(y_true, proba, threshold: float = DEFAULT_THRESHOLD) -> Figure:
    fig = Figure(figsize=(4.5, 4))
    ax = fig.subplots()
    cm = confusion_matrix(y_true, apply_threshold(proba, threshold))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion matrix (threshold {threshold:.2f})")
    ax.set_xlabel("Predikterat")
    ax.set_ylabel("Sant")
    return fig


def plot_roc_curve(y_true, proba, label: str = "Modell") -> Figure:
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    fig = Figure(figsize=(4.5, 4))
    ax = fig.subplots()
    ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Slumpmässig modell")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC-kurva")
    ax.legend(loc="lower right")
    return fig


def feature_names(pipeline: Pipeline) -> list[str]:
    """Kolumnnamnen efter förbehandlingen, utan prefixen num__/cat__ som ColumnTransformer lägger på."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    return [name.split("__", 1)[1] for name in names]


def feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Vilka features modellen lutar sig mest mot.

    - Trädmodeller: feature_importances_ (summerar till 1, alltid >= 0).
    - Logistisk regression: koefficienterna. Positiv koefficient = ökar churn-risken,
      negativ = minskar den. Koefficienterna är jämförbara eftersom numeriska
      kolumner standardiserats i pipelinen.
    """
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        kind = "importance"
    elif hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
        kind = "coefficient"
    else:
        raise ValueError(f"{type(classifier).__name__} saknar både feature_importances_ och coef_")

    table = pd.DataFrame({"feature": feature_names(pipeline), "value": values, "kind": kind})
    return table.reindex(table["value"].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )


def plot_feature_importance(table: pd.DataFrame, top_n: int = 15) -> Figure:
    top = table.head(top_n).iloc[::-1]  # störst överst i grafen
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top["value"]]
    fig = Figure(figsize=(6, 0.35 * len(top) + 1))
    ax = fig.subplots()
    ax.barh(top["feature"], top["value"], color=colors)
    kind = top["kind"].iloc[0] if len(top) else "value"
    ax.set_xlabel("Koefficient (röd ökar churn-risk)" if kind == "coefficient" else "Importance")
    ax.set_title(f"Topp {len(top)} features")
    return fig
