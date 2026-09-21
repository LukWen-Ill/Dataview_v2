"""Prissättning: vad kostar varje tjänst, och vad kostar ett valt abonnemang?"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app_helpers import get_customers, get_pricing_model, load_or_stop
from src.pricing import PRICE_FEATURES, prepare_pricing_data, price_list

st.set_page_config(page_title="Prissättning", page_icon="💰", layout="wide")
st.title("💰 Prissättning")

customers = load_or_stop(get_customers, "kunddatan")
pipeline, r2 = load_or_stop(get_pricing_model, "prismodellen")

st.metric("R²", f"{r2:.4f}", border=True)

st.markdown(
    """
Månadspriset borde vara en summa av de tjänster kunden valt. För att testa det tränade vi en
linjär regression på enbart tjänstekolumnerna - `tenure` och `TotalCharges` uteslöts, eftersom de
innehåller priset och skulle gjort uppgiften trivial. Vi använde bara de kunder som faktiskt har
internet, eftersom `No internet service` inte är ett tjänsteval utan ett "frågan gäller inte".

R² = 0,997 betyder att modellen träffar nästan exakt. Det är inte ett tecken på en bra modell,
utan på att det inte fanns något att prediktera: priset är aritmetik, inte ett osäkert utfall.
Jämför med churn-modellens ROC-AUC 0,84, där utfallet beror på mänskliga beslut.

Koefficienterna blir därför en prislista.
"""
)

st.subheader("Prislista")
st.dataframe(
    price_list(pipeline).round(2),
    width="stretch",
)

st.subheader("Sätt ihop ett abonnemang")

X_pricing, _ = prepare_pricing_data(customers)

with st.form("abonnemang"):
    cols = st.columns(3)
    values: dict[str, object] = {}
    for i, col in enumerate(PRICE_FEATURES):
        options = sorted(X_pricing[col].dropna().unique())
        values[col] = cols[i % 3].selectbox(col, options)
    submitted = st.form_submit_button("Räkna ut pris")

if submitted:
    pris = pipeline.predict(pd.DataFrame([values])[PRICE_FEATURES])[0]
    st.metric("Månadspris", f"{pris:.2f} USD", border=True)

    kombination = (X_pricing["InternetService"] == values["InternetService"]) & (
        X_pricing["PhoneService"] == values["PhoneService"]
    )
    if not kombination.any():
        st.warning(
            "Den kombinationen finns inte bland kunderna i datasettet. "
            "Priset är extrapolerat och bygger inte på observerade abonnemang."
        )
