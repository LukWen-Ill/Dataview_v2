"""Data / EDA. Några få grafer som faktiskt säger något om churn-problemet."""

from __future__ import annotations

import altair as alt
import streamlit as st

from app_helpers import get_customers, load_or_stop
from src.data import TARGET_COLUMN
from src.eda import EDA_CATEGORIES, churn_correlations, churn_rate_by, with_churn_flag

st.set_page_config(page_title="Data", page_icon="📊", layout="wide")
st.title("📊 Data och EDA")

customers = load_or_stop(get_customers, "kunddatan")
data = with_churn_flag(customers)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Kunder", f"{len(customers):,}".replace(",", " "), border=True)
c2.metric("Churn-andel", f"{data['churn'].mean():.1%}", border=True)
c3.metric("Medelkundtid", f"{data['tenure'].mean():.0f} mån", border=True)
c4.metric("Medelmånadskostnad", f"{data['MonthlyCharges'].mean():.0f} $", border=True)

st.info(
    "Klasserna är obalanserade: bara var fjärde kund churnar. Därför räcker inte accuracy som "
    "mått - en modell som alltid svarar 'stannar' får 73 % rätt utan att hitta en enda churnare."
)

st.subheader("Churn-andel per kategori")
column = st.selectbox("Välj kolumn", EDA_CATEGORIES, index=0)
rates = churn_rate_by(customers, column)
chart = (
    alt.Chart(rates)
    .mark_bar()
    .encode(
        x=alt.X(f"{column}:N", sort="-y", title=column),
        y=alt.Y("churn_andel:Q", title="Churn-andel", axis=alt.Axis(format="%")),
        tooltip=[
            alt.Tooltip(f"{column}:N"),
            alt.Tooltip("churn_andel:Q", format=".1%", title="Churn"),
            alt.Tooltip("antal_kunder:Q", title="Kunder"),
        ],
    )
    .properties(height=300)
)
st.altair_chart(chart, width="stretch")

st.subheader("Kundtid och månadskostnad")
left, right = st.columns(2)
with left:
    tenure_chart = (
        alt.Chart(data.assign(Churn=customers[TARGET_COLUMN]))
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("tenure:Q", bin=alt.Bin(step=6), title="Kundtid (månader)"),
            y=alt.Y("count()", title="Antal kunder"),
            color=alt.Color(
                "Churn:N", scale=alt.Scale(domain=["No", "Yes"], range=["#4c78a8", "#e45756"])
            ),
        )
        .properties(height=280, title="De flesta som churnar gör det under första året")
    )
    st.altair_chart(tenure_chart, width="stretch")
with right:
    charges_chart = (
        alt.Chart(data.assign(Churn=customers[TARGET_COLUMN]))
        .mark_boxplot()
        .encode(
            x=alt.X("Churn:N"),
            y=alt.Y("MonthlyCharges:Q", title="Månadskostnad"),
            color=alt.Color(
                "Churn:N",
                legend=None,
                scale=alt.Scale(domain=["No", "Yes"], range=["#4c78a8", "#e45756"]),
            ),
        )
        .properties(height=280, title="Churnare betalar mer per månad")
    )
    st.altair_chart(charges_chart, width="stretch")

st.subheader("Korrelation mellan numeriska kolumner och churn")
st.caption(
    "Positivt värde = högre värde hänger ihop med mer churn. Mäter bara linjära samband, "
    "så det är en första ledtråd, inte ett facit."
)
corr = churn_correlations(customers).reset_index()
corr.columns = ["kolumn", "korrelation"]
corr_chart = (
    alt.Chart(corr)
    .mark_bar()
    .encode(
        x=alt.X("korrelation:Q", title="Korrelation med churn"),
        y=alt.Y("kolumn:N", sort="x", title=None),
        color=alt.condition(alt.datum.korrelation > 0, alt.value("#e45756"), alt.value("#4c78a8")),
        tooltip=[alt.Tooltip("korrelation:Q", format=".3f")],
    )
    .properties(height=220)
)
st.altair_chart(corr_chart, width="stretch")

with st.expander("Rådata från databasen (första 200 raderna)"):
    st.dataframe(customers.head(200), width="stretch")
    st.caption(
        "Härledda kolumner som modellen också får: `num_addon_services`, `avg_monthly_charge`, "
        "`tenure_group`. 11 kunder har tom `TotalCharges` (alla har tenure 0) - de blir NaN och "
        "medianimputeras i pipelinen."
    )
