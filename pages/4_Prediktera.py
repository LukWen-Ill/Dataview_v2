"""Prediktera: en kund via formulär (loggas i databasen) eller många via CSV."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_helpers import get_customers, get_model, get_results, load_or_stop
from src import db
from src.data import RAW_CATEGORICAL_COLUMNS, RAW_FEATURE_COLUMNS, SchemaError, clean
from src.evaluate import THRESHOLDS
from src.model import predict_proba

st.set_page_config(page_title="Prediktera", page_icon="🔮", layout="wide")
st.title("🔮 Prediktera churn")

customers = load_or_stop(get_customers, "kunddatan")
pipeline = load_or_stop(get_model, "modellen")
results = load_or_stop(get_results, "träningsresultatet")
cleaned = clean(customers)

st.caption(f"Modell: {results['best_model']} (tränad {results['created_at'][:10]}).")

threshold = st.select_slider(
    "Threshold – från vilken sannolikhet räknar vi kunden som churnare?",
    options=THRESHOLDS,
    value=0.5,
)
st.caption(
    "Lägre threshold fångar fler potentiella churnare men ger fler falska positiva. "
    "Se sidan Modeller för hur precision och recall påverkas."
)

single_tab, batch_tab = st.tabs(["En kund", "Många kunder (CSV)"])

with single_tab:
    with st.form("kund"):
        cols = st.columns(3)
        values: dict[str, object] = {}
        values["SeniorCitizen"] = cols[0].selectbox("SeniorCitizen", [0, 1])
        values["tenure"] = cols[1].number_input("tenure (månader)", 0, 100, 12)
        values["MonthlyCharges"] = cols[2].number_input(
            "MonthlyCharges", 0.0, 500.0, float(cleaned["MonthlyCharges"].median())
        )
        values["TotalCharges"] = cols[0].number_input(
            "TotalCharges", 0.0, 20000.0, float(cleaned["TotalCharges"].median())
        )
        for i, col in enumerate(RAW_CATEGORICAL_COLUMNS, start=1):
            options = sorted(customers[col].dropna().unique())
            values[col] = cols[i % 3].selectbox(col, options)
        log_it = st.checkbox("Logga prediktionen i databasen", value=True)
        submitted = st.form_submit_button("Prediktera")

    if submitted:
        customer = pd.DataFrame([values])[RAW_FEATURE_COLUMNS]
        probability = float(predict_proba(pipeline, customer).iloc[0])
        churns = probability >= threshold

        c1, c2 = st.columns([1, 2])
        c1.metric("Sannolikhet för churn", f"{probability:.1%}", border=True)
        c2.metric(
            f"Beslut vid threshold {threshold:.2f}",
            "⚠️ Risk för churn" if churns else "✅ Förväntas stanna",
            border=True,
        )
        st.progress(probability)
        if churns:
            st.warning(
                f"Sannolikheten {probability:.1%} ligger över threshold {threshold:.2f}. "
                "Kunden bör prioriteras för en åtgärd, t.ex. erbjudande om längre avtal."
            )
        else:
            st.success(
                f"Sannolikheten {probability:.1%} ligger under threshold {threshold:.2f}. "
                "Modellen bedömer att kunden sannolikt stannar."
            )
        if log_it:
            db.log_prediction(values, probability, threshold, db.DEFAULT_DB_PATH)
            st.caption("Prediktionen sparades i tabellen `predictions`.")

    with st.expander("Senaste loggade prediktioner"):
        try:
            logged = db.load_predictions(db.DEFAULT_DB_PATH, limit=20)
        except FileNotFoundError as err:
            st.info(str(err))
        else:
            if logged.empty:
                st.info("Inga prediktioner loggade ännu.")
            else:
                st.dataframe(
                    logged.drop(columns=["customer_input"]), width="stretch", hide_index=True
                )

with batch_tab:
    st.caption("Ladda upp en CSV med samma kolumner som datasettet (Churn-kolumnen är frivillig).")
    uploaded = st.file_uploader("CSV", type="csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            result = df.assign(
                churn_probability=predict_proba(pipeline, df),
            )
            result["predicted_churn"] = (result["churn_probability"] >= threshold).astype(int)
        except (SchemaError, ValueError) as err:
            st.error(f"Filen gick inte att prediktera på: {err}")
        else:
            n_flagged = int(result["predicted_churn"].sum())
            st.metric(
                "Kunder flaggade som churn-risk", f"{n_flagged} av {len(result)}", border=True
            )
            st.dataframe(result.sort_values("churn_probability", ascending=False), width="stretch")
            st.download_button(
                "Ladda ner resultat",
                result.to_csv(index=False).encode("utf-8"),
                "churn_predictions.csv",
                "text/csv",
            )
