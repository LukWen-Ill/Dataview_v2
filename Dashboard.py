"""Streamlit-appens startsida: Dashboard med intäktsprognos. Kör: streamlit run Dashboard.py

Övriga sidor ligger i pages/. All ML-logik ligger i src/ - sidorna visar bara resultat.
Bygger på modellens churn-sannolikheter. Bara presentation.
"""

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

st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")
st.title("📈 Dashboard – intäktsprognos")

customers = load_or_stop(get_customers_with_predictions, "kunddatan och modellen")

# K-Means-segment som filter. Etiketterna ligger i samma radordning som kundtabellen,
# så kolumnen måste läggas på hela tabellen här, innan någon filtrering ändrar ordningen.
labels, _, _ = load_or_stop(lambda: get_segments(DEFAULT_K), "segmenten")
customers = customers.assign(segment=labels)

# Kundvänliga namn på kolumnerna. Datavärdena (t.ex. "Month-to-month") visas som de är.
COLUMN_LABELS = {
    "Contract": "Avtal",
    "InternetService": "Internet",
    "PaymentMethod": "Betalsätt",
    "segment": "Segment",
    "tenure_group": "Kundtid (månader)",
}

# Filter. Tom lista = inget filter, precis som filter_customers tolkar det.
# Sista rutan styr korten längst ned; den ligger här så att alla anrop kan göras i ett svep.
FILTER_COLUMNS = ["Contract", "InternetService", "PaymentMethod", "segment"]
HORIZONS = [3, 6, 12]  # månader framåt i grafen, max 12
f1, f2, f3, f4, f5, f6 = st.columns(6)
filters = {}
for col, box in zip(FILTER_COLUMNS, (f1, f2, f3, f4), strict=True):
    filters[col] = box.multiselect(COLUMN_LABELS[col], sorted(customers[col].unique()))
horizon = f5.selectbox("Prognos (månader)", HORIZONS, index=len(HORIZONS) - 1)
group_column = f6.selectbox("Visa per", GROUP_COLUMNS, index=0, format_func=COLUMN_LABELS.get)

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


def churn_color(rate: float) -> str:
    """Färg för churn-risk i korten: grön under 15 %, orange upp till 30 %, röd över."""
    if rate >= 0.30:
        return "red"
    if rate >= 0.15:
        return "orange"
    return "green"


c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Månadsintäkt idag", money(numbers["mrr_today"]), border=True)
c2.metric("Väntad förlust nästa månad", money(numbers["expected_loss_next_month"]), border=True)
c3.metric("Förlorad intäkt förra månaden", money(numbers["actual_loss_last_month"]), border=True)
c4.metric("Churn-risk", f"{numbers['predicted_churn_rate']:.1%}", border=True)
c5.metric("Aktiva kunder", f"{numbers['n_customers']:,}".replace(",", " "), border=True)
c6.metric(f"Månadsintäkt om {horizon} mån", money(forecast["total_mrr"].iloc[-1]), border=True)

st.subheader(f"Prognos {horizon} månader framåt")
# Två serier i långt format så att Altair kan färga dem. Blå = dagens kunder plus mockade
# nykunder, röd = bara dagens kunder. Båda startar i dagens månadsintäkt; gapet är nykunderna.
SERIES = {"total_mrr": "Månadsintäkt", "expected_mrr": "Dagens kunder utan nykunder"}
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
            alt.Tooltip("total_mrr:Q", format=",.0f", title="Månadsintäkt"),
            alt.Tooltip("expected_mrr:Q", format=",.0f", title="  varav dagens kunder"),
            alt.Tooltip("new_mrr:Q", format=",.0f", title="  varav nya kunder"),
            alt.Tooltip("total_loss:Q", format=",.0f", title="Väntad förlust"),
            alt.Tooltip("expected_loss:Q", format=",.0f", title="  varav dagens kunder"),
            alt.Tooltip("expected_customers:Q", format=".0f", title="Dagens kunder kvar"),
            alt.Tooltip("new_customers:Q", format=",.0f", title="Nya kunder"),
        ],
    )
    .add_params(hover)
)
chart = alt.layer(lines, points, rule, bands).properties(height=HEIGHT)
st.altair_chart(chart, width="stretch")
st.caption(
    "Prognosen bygger på dagens aktiva kunder plus en antagen nykundstillväxt "
    f"(ca {NEW_CUSTOMER_MOCK['start']} nya kunder per månad, +{NEW_CUSTOMER_MOCK['growth']:.0%} "
    "per månad), konstant churn-risk per kund och oförändrade priser."
)

# Ett kort per grupp med churn-risken i fokus. Sorterat på väntad förlust, störst först.
st.subheader(f"Per {COLUMN_LABELS[group_column].lower()}")
for card, (_, row) in zip(st.columns(len(table)), table.iterrows(), strict=True):
    with card.container(border=True):
        rate = row["predikterad_churn"]
        st.markdown(f"**{row[group_column]}**")
        st.markdown(f"## :{churn_color(rate)}[{rate:.0%}]")
        st.caption("churn-risk")
        st.markdown(
            f"{row['antal_kunder']:,} kunder  \n"
            f"{money(row['mrr'])} per månad  \n"
            f"Väntad förlust {money(row['forvantad_forlust_nasta_manad'])}".replace(",", " ")
        )
