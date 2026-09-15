"""Streamlit-app för churn-prediktion. Kör: streamlit run app.py"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve

from src.data import (
    CATEGORICAL_COLUMNS,
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    SchemaError,
    clean,
    load_dataset,
    load_raw,
)
from src.model import predict_proba, train

st.set_page_config(page_title="Churn-prediktion", page_icon="📉", layout="wide")


@st.cache_data
def get_raw() -> pd.DataFrame:
    return load_raw(DEFAULT_DATA_PATH)


@st.cache_resource
def get_model():
    """Tränar vid första anropet och cachar. Tar under en sekund på 7 000 rader."""
    X, y = load_dataset(DEFAULT_DATA_PATH)
    return train(X, y)


try:
    raw = get_raw()
    trained = get_model()
    pipeline, metrics = trained.pipeline, trained.metrics
except (FileNotFoundError, SchemaError) as err:
    st.error(f"Kunde inte starta appen: {err}")
    st.stop()

st.title("📉 Churn-prediktion")
st.caption("Telco Customer Churn · logistisk regression · scikit-learn")

data_tab, model_tab, predict_tab = st.tabs(["Data", "Modell", "Prediktera"])

with data_tab:
    churn_rate = (raw[TARGET_COLUMN] == "Yes").mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Kunder", f"{len(raw):,}".replace(",", " "))
    c2.metric("Churn-andel", f"{churn_rate:.1%}")
    c3.metric("Kolumner", len(raw.columns))

    st.subheader("Churn per avtalstyp")
    by_contract = (
        raw.assign(churn=(raw[TARGET_COLUMN] == "Yes").astype(int))
        .groupby("Contract", as_index=False)["churn"]
        .mean()
    )
    st.altair_chart(
        alt.Chart(by_contract)
        .mark_bar()
        .encode(
            x=alt.X("Contract:N", title="Avtal", sort="-y"),
            y=alt.Y("churn:Q", title="Churn-andel", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("churn:Q", format=".1%", title="Churn")],
        ),
        use_container_width=True,
    )

    st.subheader("Rådata")
    st.dataframe(raw.head(100), use_container_width=True)

with model_tab:
    st.subheader("Utvärdering på hållet testset")
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{metrics.roc_auc:.3f}")
    c2.metric("Recall", f"{metrics.recall:.3f}")
    c3.metric("Precision", f"{metrics.precision:.3f}")
    c1.metric("Träffsäkerhet", f"{metrics.accuracy:.3f}")
    c2.metric("F1", f"{metrics.f1:.3f}")
    c3.metric("Rader (träning/test)", f"{metrics.n_train} / {metrics.n_test}")
    st.info(
        "Modellen är viktad mot minoritetsklassen (`class_weight='balanced'`). "
        "Det ger högre recall - den hittar fler churnare - till priset av lägre precision."
    )

    st.subheader("ROC-kurva")
    fpr, tpr, _ = roc_curve(trained.y_test, trained.y_proba)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    st.altair_chart(
        alt.Chart(roc_df)
        .mark_line()
        .encode(
            x=alt.X("fpr:Q", title="Falska larm", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("tpr:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
        ),
        use_container_width=True,
    )
    st.caption(f"AUC = {metrics.roc_auc:.3f}")

with predict_tab:
    single, batch = st.tabs(["En kund", "Ladda upp CSV"])

    with single:
        cleaned = clean(raw)
        with st.form("kund"):
            cols = st.columns(3)
            values: dict[str, object] = {}
            for i, col in enumerate(NUMERIC_COLUMNS):
                series = cleaned[col].dropna()
                values[col] = cols[i % 3].number_input(
                    col,
                    min_value=float(series.min()),
                    max_value=float(series.max()),
                    value=float(series.median()),
                )
            for i, col in enumerate(CATEGORICAL_COLUMNS):
                options = sorted(raw[col].dropna().unique())
                values[col] = cols[i % 3].selectbox(col, options)
            submitted = st.form_submit_button("Prediktera")

        if submitted:
            proba = predict_proba(pipeline, pd.DataFrame([values])[FEATURE_COLUMNS]).iloc[0]
            st.metric("Sannolikhet för churn", f"{proba:.1%}")
            st.progress(float(proba))

    with batch:
        uploaded = st.file_uploader("CSV med samma kolumner som datasettet", type="csv")
        if uploaded is not None:
            try:
                df = clean(pd.read_csv(uploaded))
                result = df.assign(churn_probability=predict_proba(pipeline, df))
            except (SchemaError, ValueError) as err:
                st.error(f"Filen gick inte att prediktera på: {err}")
            else:
                st.dataframe(
                    result.sort_values("churn_probability", ascending=False),
                    use_container_width=True,
                )
                st.download_button(
                    "Ladda ner resultat",
                    result.to_csv(index=False).encode("utf-8"),
                    "churn_predictions.csv",
                    "text/csv",
                )
