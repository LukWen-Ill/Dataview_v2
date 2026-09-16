"""Streamlit-appens startsida: översikt. Kör: streamlit run app.py

Övriga sidor ligger i pages/. All ML-logik ligger i src/ - sidorna visar bara resultat.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_helpers import get_customers, get_results, load_or_stop, metrics_row
from src.data import TARGET_COLUMN

st.set_page_config(page_title="Churn-prediktion", page_icon="📉", layout="wide")

customers = load_or_stop(get_customers, "kunddatan")
results = load_or_stop(get_results, "träningsresultatet")

st.title("📉 Churn-prediktion – Telco Customer Churn")
st.caption(
    "Vilka kunder riskerar att säga upp sitt abonnemang? "
    "Data i SQLite → scikit-learn-pipeline → Streamlit."
)

churn_rate = (customers[TARGET_COLUMN] == "Yes").mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Kunder i databasen", f"{len(customers):,}".replace(",", " "), border=True)
c2.metric("Churn-andel", f"{churn_rate:.1%}", border=True)
c3.metric("Vald modell", results["best_model"], border=True)
c4.metric("Tränad", results["created_at"][:10], border=True)

st.subheader("Slutmodellen på testmängden")
st.caption(
    f"{results['split']['n_test']} kunder som modellen aldrig sett under träning eller modellval. "
    "Threshold 0,5."
)
metrics_row(results["test"])

st.subheader("Modelljämförelse på valideringsmängden")
table = pd.DataFrame(
    [
        {"Modell": m["model"], "CV ROC-AUC": m["cv_roc_auc"], **m["validation"]}
        for m in results["models"]
    ]
).set_index("Modell")
st.dataframe(table.style.format("{:.3f}").highlight_max(axis=0, color="#d4edda"), width="stretch")

st.subheader("Så här hänger det ihop")
st.markdown(
    f"""
1. **Data** – rå-CSV:n laddas in i SQLite (`data/churn.db`). Appen och träningen läser därifrån.
2. **Feature engineering** – tre härledda kolumner: antal tilläggstjänster, snittkostnad per månad
   och kundtid i grupper.
3. **Split** – stratifierat i {results["split"]["n_train"]} train /
   {results["split"]["n_validation"]} validation / {results["split"]["n_test"]} test.
4. **Modeller** – logistisk regression, beslutsträd och random forest, alla i samma pipeline med
   imputering, skalning och one-hot. Hyperparametrar väljs med GridSearchCV
   ({results["cv_folds"]}-delad korsvalidering på train, scoring = {results["scoring"]}).
5. **Modellval** – på validation. Slutmodellen tränas om på train + validation och utvärderas
   en enda gång på test.
6. **Prediktion** – appen laddar `models/churn_model.joblib`, du väljer threshold och varje
   prediktion loggas i databasen.

Använd menyn till vänster: **Data** (EDA), **Modeller** (utvärdering, threshold, feature
importance), **Segmentering** (K-Means + PCA) och **Prediktera**.
"""
)
