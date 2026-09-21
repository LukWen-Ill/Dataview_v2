"""Kundsegmentering med K-Means, visualiserad med PCA."""

from __future__ import annotations

import streamlit as st

from app_helpers import get_customers, get_kmeans_scores, get_segments, load_or_stop
from src.segment import DEFAULT_K, plot_elbow_and_silhouette, plot_segments, segment_profiles

st.set_page_config(page_title="Segmentering", page_icon="🧩", layout="wide")
st.title("🧩 Kundsegmentering (K-Means + PCA)")

customers = load_or_stop(get_customers, "kunddatan")

st.markdown(
    """
Churn-modellen är **vägledd inlärning**: den tränas på facit (Churn) och svarar på
*"vilka kunder riskerar att lämna?"*. Segmenteringen är **icke-vägledd**: K-Means får inte se
Churn-kolumnen utan grupperar kunder som liknar varandra och svarar på *"vilka typer av kunder
har vi?"*. Efteråt kan vi titta på churn-andelen per segment för att tolka grupperna.

Kunderna förbehandlas på samma sätt som för churn-modellen (skalning + one-hot), klustras med
K-Means och projiceras med **PCA** till två dimensioner - bara för att kunna ritas.
"""
)

st.subheader("Hur många kluster?")
simple = st.toggle(
    "Klustra bara på tenure (antal månader kunden varit kund), månadskostnad och tilläggstjänster"
)
with st.spinner("Beräknar inertia och silhouette för k = 2–8 …"):
    scores = load_or_stop(lambda: get_kmeans_scores(simple), "klusterpoängen")
st.pyplot(plot_elbow_and_silhouette(scores), width="stretch")
st.caption(
    "Elbow-metoden: inertia (summan av kvadratavstånd till närmaste centroid) sjunker alltid när k "
    "ökar - vi letar efter 'armbågen' där det slutar löna sig. Silhouette score mäter hur väl "
    "separerade klustren är (högre är bättre). Här pekar silhouette på 2 kluster och armbågen på "
    "3–4. Vi väljer 4 som standard: fler segment ger mer användbara kundgrupper för verksamheten, "
    "vilket är ett omdömesbeslut - precis som boken beskriver."
)

k = st.slider("Antal kluster (k)", min_value=2, max_value=8, value=DEFAULT_K)
labels, coords, explained = get_segments(k, simple)

left, right = st.columns([1, 1])
with left:
    st.pyplot(plot_segments(coords, labels, explained), width="stretch")
    st.caption(
        f"De två första principalkomponenterna förklarar {explained.sum():.0%} av variansen i den "
        "förbehandlade datan. Klustren skapades i alla dimensioner, PCA används bara för bilden."
    )
with right:
    st.markdown("**Segmentprofiler**")
    profiles = segment_profiles(customers, labels)
    st.dataframe(
        profiles.style.format(
            {
                "snitt_tenure": "{:.0f}",
                "snitt_manadskostnad": "{:.0f}",
                "snitt_tillaggstjanster": "{:.1f}",
                "churn_andel": "{:.0%}",
            }
        ).background_gradient(subset=["churn_andel"], cmap="Reds"),
        width="stretch",
        hide_index=True,
    )
    riskiest = profiles.loc[profiles["churn_andel"].idxmax()]
    safest = profiles.loc[profiles["churn_andel"].idxmin()]
    st.markdown(
        f"""
- **Segment {int(riskiest["segment"])}** har högst churn ({riskiest["churn_andel"]:.0%}):
  {riskiest["vanligaste_avtal"]}, ca {riskiest["snitt_manadskostnad"]:.0f} $/mån,
  {riskiest["snitt_tenure"]:.0f} månaders kundtid i snitt. Här finns störst potential för
  riktade åtgärder.
- **Segment {int(safest["segment"])}** har lägst churn ({safest["churn_andel"]:.0%}):
  {safest["vanligaste_avtal"]}, ca {safest["snitt_manadskostnad"]:.0f} $/mån.
"""
    )
