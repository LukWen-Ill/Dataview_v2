"""Ledning. Intäktsprognos byggd på modellens churn-sannolikheter. Bara presentation."""

from __future__ import annotations

import altair as alt
import streamlit as st

from app_helpers import get_customers_with_predictions, load_or_stop
from src.dashboard import GROUP_COLUMNS, filter_customers, group_breakdown, kpis, revenue_forecast

st.set_page_config(page_title="Ledning", page_icon="📈", layout="wide")
st.title("📈 Ledning – intäktsprognos")

customers = load_or_stop(get_customers_with_predictions, "kunddatan och modellen")

# Filter. Tom lista = inget filter, precis som filter_customers tolkar det.
# Sista rutan styr tabellen längst ned; den ligger här så att alla anrop kan göras i ett svep.
FILTER_COLUMNS = ["Contract", "InternetService", "PaymentMethod"]
f1, f2, f3, f4 = st.columns(4)
filters = {}
for col, box in zip(FILTER_COLUMNS, (f1, f2, f3), strict=True):
    filters[col] = box.multiselect(col, sorted(customers[col].unique()))
group_column = f4.selectbox("Gruppera tabellen på", GROUP_COLUMNS, index=0)

# Alla beräkningar sker i src/dashboard.py. Ett filter utan kunder ger ValueError.
try:
    selected = filter_customers(customers, filters)
    numbers = kpis(selected)
    forecast = revenue_forecast(selected)
    table = group_breakdown(selected, group_column)
except ValueError:
    st.warning("Inga kunder matchar filtret.")
    st.stop()


def money(value: float) -> str:
    return f"{value:,.0f} $".replace(",", " ")


c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("MRR idag", money(numbers["mrr_today"]), border=True)
c2.metric("Förväntad förlust nästa månad", money(numbers["expected_loss_next_month"]), border=True)
c3.metric("Faktisk förlust senaste månaden", money(numbers["actual_loss_last_month"]), border=True)
c4.metric("Predikterad churn", f"{numbers['predicted_churn_rate']:.1%}", border=True)
c5.metric("Aktiva kunder", f"{numbers['n_customers']:,}".replace(",", " "), border=True)

st.subheader("Intäktsprognos 24 månader framåt")
# Två serier i långt format så att Altair kan färga dem. Kopiorna behövs för att
# expected_mrr och cumulative_loss också ska finnas kvar som tooltip-kolumner.
long = forecast.assign(
    **{
        "Kvarvarande intäkt": forecast["expected_mrr"],
        "Ackumulerad förlust": forecast["cumulative_loss"],
    }
).melt(
    id_vars=[
        "months_ahead",
        "month_label",
        "expected_customers",
        "expected_mrr",
        "expected_loss",
        "cumulative_loss",
    ],
    value_vars=["Kvarvarande intäkt", "Ackumulerad förlust"],
    var_name="serie",
    value_name="belopp",
)
chart = (
    alt.Chart(long)
    .mark_line(point=True)
    .encode(
        x=alt.X("month_label:O", sort=alt.SortField("months_ahead"), title="Månad"),
        y=alt.Y("belopp:Q", title="Belopp ($)"),
        color=alt.Color(
            "serie:N",
            title=None,
            scale=alt.Scale(
                domain=["Kvarvarande intäkt", "Ackumulerad förlust"],
                range=["#4c78a8", "#e45756"],
            ),
        ),
        tooltip=[
            alt.Tooltip("month_label:O", title="Månad"),
            alt.Tooltip("expected_customers:Q", format=".0f", title="Förväntade kunder"),
            alt.Tooltip("expected_mrr:Q", format=",.0f", title="Kvarvarande intäkt"),
            alt.Tooltip("expected_loss:Q", format=",.0f", title="Förlust denna månad"),
            alt.Tooltip("cumulative_loss:Q", format=",.0f", title="Ackumulerad förlust"),
        ],
    )
    .properties(height=350)
)
st.altair_chart(chart, width="stretch")
st.caption(
    "Prognosen gäller bara dagens aktiva kunder, antar konstant churn-risk per kund och månad, "
    "frysta priser och ingen nykundsförsäljning. Den visar vad som händer om vi inte gör något."
)
st.caption(
    "Modellens sannolikhet tolkas som risk per månad, eftersom `Churn = Yes` i datasettet "
    "betyder att kunden lämnade senaste månaden. Sannolikheterna är inte kalibrerade "
    '(`class_weight="balanced"`), så kurvan är brant.'
)

st.subheader(f"Per {group_column}")
st.dataframe(
    table.style.format(
        {
            "mrr": "{:,.0f}",
            "forvantad_forlust_nasta_manad": "{:,.0f}",
            "predikterad_churn": "{:.0%}",
        }
    ),
    width="stretch",
    hide_index=True,
)
