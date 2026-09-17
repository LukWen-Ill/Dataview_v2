"""Ledning. Intäktsprognos byggd på modellens churn-sannolikheter. Bara presentation."""

from __future__ import annotations

import altair as alt
import streamlit as st

from app_helpers import get_customers_with_predictions, get_segments, load_or_stop
from src.dashboard import (
    GROUP_COLUMNS,
    NEW_CUSTOMER_MOCK,
    filter_customers,
    group_breakdown,
    kpis,
    revenue_forecast,
    with_new_customers,
)
from src.segment import DEFAULT_K

st.set_page_config(page_title="Ledning", page_icon="📈", layout="wide")
st.title("📈 Ledning – intäktsprognos")

customers = load_or_stop(get_customers_with_predictions, "kunddatan och modellen")

# K-Means-segment som filter. Etiketterna ligger i samma radordning som kundtabellen,
# så kolumnen måste läggas på hela tabellen här, innan någon filtrering ändrar ordningen.
labels, _, _ = load_or_stop(lambda: get_segments(DEFAULT_K), "segmenten")
customers = customers.assign(segment=labels)

# Filter. Tom lista = inget filter, precis som filter_customers tolkar det.
# Sista rutan styr tabellen längst ned; den ligger här så att alla anrop kan göras i ett svep.
FILTER_COLUMNS = ["Contract", "InternetService", "PaymentMethod", "segment"]
FILTER_LABELS = {"segment": "Segment"}  # övriga kolumner visas med sitt kolumnnamn
HORIZONS = [3, 6, 12]  # månader framåt i grafen, max 12
f1, f2, f3, f4, f5, f6 = st.columns(6)
filters = {}
for col, box in zip(FILTER_COLUMNS, (f1, f2, f3, f4), strict=True):
    filters[col] = box.multiselect(FILTER_LABELS.get(col, col), sorted(customers[col].unique()))
horizon = f5.selectbox("Horisont (månader)", HORIZONS, index=len(HORIZONS) - 1)
group_column = f6.selectbox("Gruppera tabellen på", GROUP_COLUMNS, index=0)

# Alla beräkningar sker i src/dashboard.py. Ett filter utan kunder ger ValueError.
try:
    selected = filter_customers(customers, filters)
    numbers = kpis(selected)
    # Mockade nykunder rullas fram med samma medelrisk som de filtrerade kunderna.
    forecast = with_new_customers(
        revenue_forecast(selected, months=horizon), numbers["predicted_churn_rate"]
    )
    table = group_breakdown(selected, group_column)
except ValueError:
    st.warning("Inga kunder matchar filtret.")
    st.stop()


def money(value: float) -> str:
    return f"{value:,.0f} $".replace(",", " ")


c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("MRR idag", money(numbers["mrr_today"]), border=True)
c2.metric("Förväntad förlust nästa månad", money(numbers["expected_loss_next_month"]), border=True)
c3.metric("Faktisk förlust senaste månaden", money(numbers["actual_loss_last_month"]), border=True)
c4.metric("Predikterad churn", f"{numbers['predicted_churn_rate']:.1%}", border=True)
c5.metric("Aktiva kunder", f"{numbers['n_customers']:,}".replace(",", " "), border=True)
c6.metric(f"MRR månad {horizon} inkl. nykunder", money(forecast["total_mrr"].iloc[-1]), border=True)

st.subheader(f"Intäktsprognos {horizon} månader framåt")
# Två serier i långt format så att Altair kan färga dem. Blå = dagens kunder plus mockade
# nykunder, röd = förväntad förlust per månad på hela den blå linjen.
SERIES = {"total_mrr": "Intäkt inkl. nykunder", "total_loss": "Förväntad förlust per månad"}
long = forecast.rename(columns=SERIES).melt(
    id_vars=["months_ahead", "month_label"],
    value_vars=list(SERIES.values()),
    var_name="serie",
    value_name="belopp",
)
HEIGHT = 350
x_axis = alt.X(
    "month_label:O",
    sort=alt.SortField("months_ahead"),
    title="Månad",
    axis=alt.Axis(grid=True, labelAngle=0),  # en lodrät rutnätslinje per månad
)
# Hover-markering per månad: en osynlig stapel över hela höjden fångar pekaren var som helst
# i kolumnen, så tooltipen visas utan att man behöver träffa en punkt på linjen.
hover = alt.selection_point(fields=["month_label"], on="pointerover", empty=False)
lines = (
    alt.Chart(long)
    .mark_line()
    .encode(
        x=x_axis,
        y=alt.Y("belopp:Q", title="Belopp ($)"),
        color=alt.Color(
            "serie:N",
            title=None,
            scale=alt.Scale(domain=list(SERIES.values()), range=["#4c78a8", "#e45756"]),
        ),
    )
)
points = lines.mark_point(filled=True, size=80).encode(
    opacity=alt.condition(hover, alt.value(1), alt.value(0))
)
rule = (
    alt.Chart(forecast)
    .mark_rule(color="gray", strokeDash=[4, 4])
    .encode(x=x_axis)
    .transform_filter(hover)
)
bands = (
    alt.Chart(forecast)
    .mark_bar(opacity=0)
    .encode(
        x=x_axis,
        y=alt.value(0),
        y2=alt.value(HEIGHT),
        tooltip=[
            alt.Tooltip("month_label:O", title="Månad"),
            alt.Tooltip("total_mrr:Q", format=",.0f", title="Intäkt inkl. nykunder"),
            alt.Tooltip("expected_mrr:Q", format=",.0f", title="  varav dagens kunder"),
            alt.Tooltip("new_mrr:Q", format=",.0f", title="  varav nykunder (mock)"),
            alt.Tooltip("total_loss:Q", format=",.0f", title="Förväntad förlust denna månad"),
            alt.Tooltip("expected_loss:Q", format=",.0f", title="  varav dagens kunder"),
            alt.Tooltip("expected_customers:Q", format=".0f", title="Dagens kunder kvar"),
            alt.Tooltip("new_customers:Q", format=",.0f", title="Nya kunder denna månad (mock)"),
        ],
    )
    .add_params(hover)
)
chart = alt.layer(lines, points, rule, bands).properties(height=HEIGHT)
st.altair_chart(chart, width="stretch")
st.caption(
    "Blå linje är dagens aktiva kunders kvarvarande intäkt plus en mockad nykundsförsäljning. "
    "Röd linje är den förväntade förlusten per månad på hela den blå linjen, räknad med "
    "churn-risken. Nykundsmocken: ca "
    f"{NEW_CUSTOMER_MOCK['start']} nya kunder månad 1, +{NEW_CUSTOMER_MOCK['growth']:.0%} per "
    f"månad med ±{NEW_CUSTOMER_MOCK['spread']:.0%} slump, "
    f"{NEW_CUSTOMER_MOCK['monthly_charge']:.0f} $ per kund och samma churn-risk som de "
    "befintliga. Hela prognosen antar konstant churn-risk per kund och månad samt frysta priser."
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
