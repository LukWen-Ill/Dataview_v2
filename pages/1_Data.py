"""Data / EDA. Några få grafer som faktiskt säger något om churn-problemet."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app_helpers import get_customers, load_or_stop
from src.data import TARGET_COLUMN
from src.eda import (
    EDA_CATEGORIES,
    PRICE_LABELS,
    churn_correlations,
    churn_rate_by,
    churn_rate_by_price_band,
    data_quality,
    with_churn_flag,
)

CHURN_COLORS = alt.Scale(domain=["No", "Yes"], range=["#4c78a8", "#e45756"])

st.set_page_config(page_title="Data", page_icon="📊", layout="wide")
st.title("📊 Data och EDA")

customers = load_or_stop(get_customers, "kunddatan")
data = with_churn_flag(customers)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Kunder", f"{len(customers):,}".replace(",", " "), border=True)
c2.metric("Churn-andel", f"{data['churn'].mean():.1%}", border=True)
c3.metric("Medelkundtid", f"{data['tenure'].mean():.0f} mån", border=True)
c4.metric("Medelmånadskostnad", f"{data['MonthlyCharges'].mean():.0f} USD", border=True)

st.info(
    "Klasserna är obalanserade: bara var fjärde kund churnar. Därför räcker inte accuracy som "
    "mått - en modell som alltid svarar 'stannar' får 73 % rätt utan att hitta en enda churnare."
)

# --- Datakontroll -------------------------------------------------------------------------
st.subheader("Datakontroll")
quality = data_quality(customers)
q1, q2, q3 = st.columns(3)
q1.metric("Unika kund-id", f"{quality['unika_id']:,}".replace(",", " "), border=True)
q2.metric("Dubbletter", quality["dubbletter"], border=True)
q3.metric("Tomma TotalCharges", quality["tomma_totalcharges"], border=True)
st.caption(
    "Datan är ren: varje rad är en unik kund. De kunder som saknar TotalCharges har alla "
    "tenure 0, alltså nya kunder som inte fakturerats än. De behålls och medianimputeras i "
    "pipelinen - en ny kund är precis vad modellen ska kunna prediktera på."
)

# --- Churn per kategori -------------------------------------------------------------------
st.subheader("Churn-andel per kategori")
column = st.selectbox("Välj kolumn", EDA_CATEGORIES, index=0)
rates = churn_rate_by(customers, column)
chart = (
    alt.Chart(rates)
    .mark_bar()
    .encode(
        y=alt.Y(f"{column}:N", sort="-x", title=None),
        x=alt.X("churn_andel:Q", title="Churn-andel", axis=alt.Axis(format="%")),
        tooltip=[
            alt.Tooltip(f"{column}:N"),
            alt.Tooltip("churn_andel:Q", format=".1%", title="Churn"),
            alt.Tooltip("antal_kunder:Q", title="Kunder"),
        ],
    )
)
# Streckad linje vid den totala churn-andelen, så man ser vilka grupper som ligger över snittet.
mean_line = (
    alt.Chart(pd.DataFrame({"snitt": [data["churn"].mean()]}))
    .mark_rule(strokeDash=[4, 4], color="grey")
    .encode(x="snitt:Q", tooltip=[alt.Tooltip("snitt:Q", format=".1%", title="Snitt")])
)
# Höjden måste sättas på det lagrade diagrammet, inte på ett enskilt lager.
layered = (chart + mean_line).properties(height=max(180, 60 * len(rates)))
st.altair_chart(layered, width="stretch")
st.caption(
    "Streckad linje = churn-andel för alla kunder. "
    "Fyra kolumner har en grupp med churn över 40 %: månadsavtal, första året som kund, "
    "fiber och e-check. Kunder med internet men utan TechSupport eller OnlineSecurity churnar "
    "också över 40 %, mot cirka 15 % för de som har tjänsten. Tilläggstjänster verkar binda kunden."
)

# --- Kundtid och pris ---------------------------------------------------------------------
st.subheader("Kundtid och månadskostnad")
left, right = st.columns(2)
with left:
    tenure_chart = (
        alt.Chart(data.assign(Churn=customers[TARGET_COLUMN]))
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("tenure:Q", bin=alt.Bin(step=6), title="Kundtid (månader)"),
            y=alt.Y("count()", title="Antal kunder"),
            color=alt.Color("Churn:N", scale=CHURN_COLORS),
            tooltip=[
                alt.Tooltip("tenure:Q", bin=alt.Bin(step=6), title="Kundtid (månader)"),
                alt.Tooltip("count()", title="Antal kunder"),
                alt.Tooltip("Churn:N"),
            ],
        )
        .properties(height=280, title="De flesta som churnar gör det under första året")
    )
    st.altair_chart(tenure_chart, width="stretch")
with right:
    bands = churn_rate_by_price_band(customers)
    price_chart = (
        alt.Chart(bands)
        .mark_bar()
        .encode(
            x=alt.X("prisintervall:N", sort=PRICE_LABELS, title="Månadskostnad (USD)"),
            y=alt.Y("churn_andel:Q", title="Churn-andel", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("prisintervall:N", title="Intervall"),
                alt.Tooltip("churn_andel:Q", format=".1%", title="Churn"),
                alt.Tooltip("antal_kunder:Q", title="Kunder"),
            ],
        )
        .properties(height=280, title="Churn följer inte priset rakt av")
    )
    st.altair_chart(price_chart, width="stretch")
st.caption(
    "Hälften av churnarna lämnar inom 10 månader. Churn per prisintervall går upp och ner: "
    "lågt för enbart telefoni (18-30), högt för DSL utan tillägg (30-50), lägre igen för "
    "DSL med tjänster (50-70), högst för fiber (70-110) och lägst hos de dyraste kunderna som "
    "har fiber plus många tilläggstjänster. Tjänstepaketet styr, inte priset i sig."
)

# --- Korrelationer ------------------------------------------------------------------------
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
        y=alt.Y("kolumn:N", sort="x", title=None, axis=alt.Axis(labelLimit=200)),
        color=alt.condition(alt.datum.korrelation > 0, alt.value("#e45756"), alt.value("#4c78a8")),
        tooltip=[alt.Tooltip("kolumn:N"), alt.Tooltip("korrelation:Q", format=".3f")],
    )
    .properties(height=220)
)
st.altair_chart(corr_chart, width="stretch")
st.caption(
    "Tenure har starkast samband med churn. Flera kolumner mäter dock samma sak: "
    "avg_monthly_charge korrelerar 1,00 med MonthlyCharges och TotalCharges 0,83 med tenure. "
    "Det stör inte trädmodeller, men gör koefficienterna i logistisk regression svårare att tolka."
)

# --- Slutsatser ---------------------------------------------------------------------------
st.subheader("Vad EDA:n betyder för modelleringen")
st.markdown(
    """
1. **Datan är ren.** Inga dubbletter, inga saknade värden utom 11 nya kunder utan
   TotalCharges. De behålls och imputeras med medianen i pipelinen.
2. **Churn är obalanserat.** 26,5 % lämnar. Därför stratifierad split, `class_weight="balanced"`
   och utvärdering med precision, recall, F1 och ROC-AUC i stället för accuracy.
3. **Churn sker tidigt och hos obundna kunder.** Månadsavtal, första året, fiber och e-check
   ligger alla över 40 %. Contract och tenure väntas bli de viktigaste variablerna.
4. **Tjänstepaketet styr, inte priset.** Sambandet mellan pris och churn är inte linjärt,
   vilket talar för trädmodeller eller för grupperade variabler som tenure_group.
5. **Flera kolumner mäter samma sak.** avg_monthly_charge är i praktiken identisk med
   MonthlyCharges och tillför inget. Vi behöll den eftersom trädmodeller inte skadas av det,
   men den kunde ha tagits bort.

EDA:n visar samband, inte orsak. Vi vet inte om tilläggstjänster håller kvar kunder eller om
lojala kunder väljer fler tjänster. Modellen predikterar risk, den förklarar inte varför.
"""
)

with st.expander("Rådata från databasen (första 200 raderna)"):
    st.dataframe(customers.head(200), width="stretch")
    st.caption(
        "Härledda kolumner som modellen också får: `num_addon_services`, `avg_monthly_charge`, "
        "`tenure_group`."
    )
