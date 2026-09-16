"""Modeller: jämförelse, slutlig utvärdering på test, threshold och feature importance."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_helpers import get_model, get_results, get_test_predictions, load_or_stop, metrics_row
from src.evaluate import (
    THRESHOLDS,
    compute_metrics,
    feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    report,
    threshold_table,
)

st.set_page_config(page_title="Modeller", page_icon="🧪", layout="wide")
st.title("🧪 Modeller och utvärdering")

results = load_or_stop(get_results, "träningsresultatet")
pipeline = load_or_stop(get_model, "modellen")
test = load_or_stop(get_test_predictions, "testprediktionerna")
y_true, proba = test["y_true"], test["churn_probability"]

# --- 1. Jämförelse på validation ---
st.subheader("1. Modelljämförelse på valideringsmängden")
st.caption(
    f"Varje modell har fått sina hyperparametrar via GridSearchCV ({results['cv_folds']}-delad "
    f"korsvalidering på träningsmängden, scoring = {results['scoring']}). Den bästa pipelinen per "
    f"modell utvärderas sedan på valideringsmängden ({results['split']['n_validation']} kunder). "
    "Testmängden rörs inte här."
)
table = pd.DataFrame(
    [
        {"Modell": m["model"], "CV ROC-AUC": m["cv_roc_auc"], **m["validation"]}
        for m in results["models"]
    ]
).set_index("Modell")
st.dataframe(table.style.format("{:.3f}").highlight_max(axis=0, color="#d4edda"), width="stretch")

with st.expander("Valda hyperparametrar"):
    for m in results["models"]:
        params = {k.replace("classifier__", ""): v for k, v in m["best_params"].items()}
        st.markdown(f"**{m['model']}**: `{params}`")

st.markdown(
    f"""
Modellerna ligger nära varandra. **{results['best_model']}** valdes eftersom den hade högst
{results['selection_metric'].upper().replace('_', '-')} på validation. Läs måtten så här:

- **Recall** – andel av de som faktiskt churnar som vi hittar. Viktigast för oss: en missad churnare
  är en förlorad kund.
- **Precision** – andel av de vi flaggar som faktiskt churnar. Låg precision = många onödiga
  åtgärder.
- **F1** – balansen mellan de två.
- **ROC-AUC** – hur bra modellen rangordnar kunder efter risk, oberoende av threshold.
- **Accuracy** – vilseledande vid obalans: 73 % nås genom att alltid svara "stannar".
"""
)

# --- 2. Test ---
st.subheader(f"2. Slutmodellen ({results['best_model']}) på testmängden")
st.caption(
    f"Tränad om på train + validation, utvärderad en enda gång på {results['split']['n_test']} "
    "kunder den aldrig sett. Det är den här siffran vi kan förvänta oss på nya kunder."
)
metrics_row(results["test"])

# --- 3. Threshold ---
st.subheader("3. Threshold: hur många churnare vill vi fånga?")
st.markdown(
    """
Modellen ger en **sannolikhet** för churn. Threshold är gränsen där vi bestämmer oss för att
kalla kunden "churnare". Standard är 0,5, men det är ett affärsbeslut:

- **Lägre threshold** → fler kunder flaggas → högre recall (vi missar färre churnare) men lägre
  precision (fler falska alarm och fler kunder att kontakta).
- **Högre threshold** → färre flaggas → högre precision men vi missar fler som faktiskt lämnar.
"""
)
threshold = st.select_slider("Threshold", options=THRESHOLDS, value=0.5)
metrics_row(compute_metrics(y_true, proba, threshold))

left, right = st.columns(2)
with left:
    st.pyplot(plot_confusion_matrix(y_true, proba, threshold), width="stretch")
with right:
    st.pyplot(plot_roc_curve(y_true, proba, results["best_model"]), width="stretch")

st.markdown("**Precision och recall för olika thresholds (testmängden):**")
st.dataframe(
    threshold_table(y_true, proba).style.format(
        {"threshold": "{:.2f}", "precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}
    ),
    width="stretch",
    hide_index=True,
)

with st.expander(f"classification_report vid threshold {threshold:.2f}"):
    st.code(report(y_true, proba, threshold))

# --- 4. Feature importance ---
st.subheader("4. Vad tittar modellen på?")
importance = feature_importance(pipeline)
kind = importance["kind"].iloc[0]
if kind == "coefficient":
    st.caption(
        "Logistisk regression: koefficienter. Röd (positiv) ökar churn-risken, blå minskar den. "
        "Numeriska kolumner är standardiserade så koefficienterna går att jämföra."
    )
else:
    st.caption(
        "Trädmodell: feature importance - hur mycket varje kolumn bidrar till att dela upp kunderna. "
        "Summerar till 1. Säger hur *viktig* en kolumn är, inte åt vilket håll den påverkar."
    )
left, right = st.columns([3, 2])
with left:
    st.pyplot(plot_feature_importance(importance), width="stretch")
with right:
    st.dataframe(
        importance[["feature", "value"]].head(15).style.format({"value": "{:.3f}"}),
        width="stretch",
        hide_index=True,
    )
st.caption(
    "Kolumner som `Contract_Month-to-month` kommer från one-hot-kodningen: en kolumn per kategori."
)
