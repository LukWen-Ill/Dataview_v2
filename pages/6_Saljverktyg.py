"""Säljverktyg (extra): månadspris och churnrisk för en ny eller befintlig kund.

Säljaren fyller i kunden i samtalets ordning, i tre kolumner: person -> produkter (telefoni,
internet, tillägg) -> betalning. Prismodellen ger månadspriset, som går in i churnmodellen som
MonthlyCharges. Risken visas relativt (låg/medel/hög mot modellens fördelning över
träningsdatan), aldrig som rå procent. All logik ligger i src/ - sidan visar bara resultat.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from app_helpers import (
    find_customer,
    get_customers,
    get_model,
    get_overview,
    get_price_model,
    get_risk_thresholds,
    get_segment_comparison,
    load_or_stop,
)
from src import db
from src.actions import suggest_actions
from src.data import ADDON_SERVICES, ID_COLUMN, RAW_FEATURE_COLUMNS, clean
from src.labels import LABELS, VALUE_LABELS
from src.model import predict_proba
from src.overview import search_customers
from src.price import PRICE_FEATURE_COLUMNS, predict_price
from src.risk import risk_factors, risk_level

st.set_page_config(page_title="Säljverktyg", page_icon="🤝", layout="wide")
st.title("🤝 Säljverktyg")
st.caption(
    "Pris från prismodellen, risknivå från churnmodellen. "
    "Modellen bedömer utifrån mönster i datan - den förklarar inte orsaker."
)

customers = load_or_stop(get_customers, "kunddatan")
pipeline = load_or_stop(get_model, "churnmodellen")
price_model = load_or_stop(get_price_model, "prismodellen")
thresholds = load_or_stop(get_risk_thresholds, "risknivåerna")

YES_NO = ["Yes", "No"]
SENIOR_LABELS = {0: "Nej", 1: "Ja"}
# Vad ett alternativs siffror bygger på, per sort i åtgärdskatalogen.
KIND_TEXT = {
    "what-if": "Bedömt av modellen",
    "samband": "Samband i datan, inte bevisad effekt",
    "regel": "Regel, ingen bedömning",
}
DISCLAIMER = (
    "Bedömningarna bygger på en statistisk modell och visar samband i datan, "
    "inte vad som händer med just den här kunden."
)
PAGE_SIZE = 25
# Arbetslistans kolumner -> kundvänliga rubriker.
LIST_HEADERS = {
    ID_COLUMN: "Kund-id",
    "level": "Risknivå",
    "MonthlyCharges": LABELS["MonthlyCharges"],
    "tenure": LABELS["tenure"],
    "Contract": LABELS["Contract"],
    "InternetService": LABELS["InternetService"],
    "PaymentMethod": LABELS["PaymentMethod"],
}


def choose(column: str, defaults: dict, key: str, container, options=None, labels=None):
    """En selectbox med kundvänlig etikett. Värdena är datasettets råvärden, texten bara visning.

    Utan default (ny kund) måste säljaren välja - tills dess är fältet None.
    """
    options = options if options is not None else list(VALUE_LABELS[column])
    labels = labels if labels is not None else VALUE_LABELS[column]
    stored = defaults.get(column)
    return container.selectbox(
        LABELS[column],
        options,
        index=options.index(stored) if stored in options else None,
        format_func=labels.get,
        key=f"{key}_{column}",
        placeholder="Välj …",
    )


def customer_form(defaults: dict, key: str) -> dict | None:
    """Formuläret i tre kolumner i samtalets ordning: Person | Produkter | Betalning.
    Returnerar kundens konfiguration utan MonthlyCharges, eller None om något fält inte är valt.

    defaults förifyller fälten (befintlig kund). Flera linjer visas bara om kunden har
    telefoni och tilläggen bara om kunden har internet - annars sätts datasettets värden
    "No phone service" respektive "No internet service". Dolda fält lämnar sin plats tom
    så att kolumnerna behåller formen.
    """
    values: dict[str, object] = {}
    person, products, payment = st.columns([2, 3, 2], gap="medium")

    person.markdown("**Person**")
    if "tenure" in defaults:
        values["tenure"] = person.number_input(
            LABELS["tenure"], 0, 100, int(defaults["tenure"]), key=f"{key}_tenure"
        )
        values["TotalCharges"] = defaults["TotalCharges"]
    else:
        values["tenure"] = 0
        values["TotalCharges"] = 0.0
    left, right = person.columns(2)
    values["SeniorCitizen"] = choose(
        "SeniorCitizen", defaults, key, left, options=[0, 1], labels=SENIOR_LABELS
    )
    values["gender"] = choose("gender", defaults, key, right)
    values["Partner"] = choose("Partner", defaults, key, left, options=YES_NO)
    values["Dependents"] = choose("Dependents", defaults, key, right, options=YES_NO)

    products.markdown("**Produkter**")
    left, right = products.columns(2)
    values["PhoneService"] = choose("PhoneService", defaults, key, left, options=YES_NO)
    if values["PhoneService"] == "Yes":
        values["MultipleLines"] = choose("MultipleLines", defaults, key, right, options=YES_NO)
    else:
        values["MultipleLines"] = "No phone service"
    values["InternetService"] = choose("InternetService", defaults, key, left)
    left, right = products.columns(2)
    if values["InternetService"] in ("DSL", "Fiber optic"):
        for i, service in enumerate(ADDON_SERVICES):
            values[service] = choose(service, defaults, key, (left, right)[i % 2], options=YES_NO)
    else:
        left.caption("Tillägg kräver internet.")
        for service in ADDON_SERVICES:
            values[service] = "No internet service"

    payment.markdown("**Betalning**")
    values["Contract"] = choose("Contract", defaults, key, payment)
    values["PaperlessBilling"] = choose("PaperlessBilling", defaults, key, payment, options=YES_NO)
    values["PaymentMethod"] = choose("PaymentMethod", defaults, key, payment)

    if any(value is None for value in values.values()):
        return None
    return values


def stored_values(row: pd.DataFrame) -> dict:
    """Kundens råvärden ur databasen med talen som Python-tal (predict_price kör ingen clean)."""
    values = clean(row).iloc[0][RAW_FEATURE_COLUMNS].to_dict()
    values["SeniorCitizen"] = int(values["SeniorCitizen"])
    values["tenure"] = int(values["tenure"])
    values["MonthlyCharges"] = float(values["MonthlyCharges"])
    values["TotalCharges"] = float(values["TotalCharges"])  # NaN om tenure är 0, som i CSV:n
    return values


def evaluate(values: dict, monthly_charges: float | None) -> tuple[pd.DataFrame, float, float]:
    """Kunden som en rad rådata, månadspris och churnsannolikhet. Utan givet pris predikteras det.

    Sannolikheten visas aldrig rå - den blir en risknivå och används för att sortera bort
    alternativ som inte bedöms ge lägre risk.
    """
    customer = pd.DataFrame([values])
    if monthly_charges is None:
        monthly_charges = float(predict_price(price_model, customer).iloc[0])
    customer["MonthlyCharges"] = monthly_charges
    customer = customer[RAW_FEATURE_COLUMNS]
    probability = float(predict_proba(pipeline, customer).iloc[0])
    return customer, monthly_charges, probability


def months(tenure: int) -> str:
    return "1 månad" if tenure == 1 else f"{tenure} månader"


def alternatives_table(actions: list[dict]) -> pd.DataFrame:
    """Alternativen som en tabell med pris och risknivå sida vid sida. Regler har inga siffror."""
    rows = []
    for action in actions:
        has_numbers = action["kind"] != "regel"
        rows.append(
            {
                "Alternativ": action["label"],
                "Pris": f"{action['price']:.2f} $/mån" if has_numbers else "–",
                "Risknivå": action["level"].capitalize() if has_numbers else "–",
                "Underlag": KIND_TEXT[action["kind"]],
                "Kommentar": action["note"],
            }
        )
    return pd.DataFrame(rows)


def show_result(customer: pd.DataFrame, monthly_charges: float, probability: float) -> None:
    """Pris, risknivå, de största riskfaktorerna och alternativen ur åtgärdskatalogen."""
    c1, c2 = st.columns(2)
    c1.metric("Pris", f"{monthly_charges:.2f} $/mån", border=True)
    c2.metric("Risknivå", risk_level(probability, thresholds).capitalize(), border=True)
    st.caption(
        "Risknivån är relativ: lägsta tredjedelen av kunderna i träningsdatan är låg, "
        "mellersta medel och högsta hög."
    )

    st.markdown("**Största riskfaktorer**")
    factors = risk_factors(customer, customers)
    if not factors:
        st.caption("Inget av kundens val ligger tydligt över snittet i churn.")
    for factor in factors:
        st.markdown(f"- {factor}")

    st.markdown("**Alternativ att lägga fram**")
    # Ett what-if som inte bedöms ge lägre sannolikhet än utgångsläget visas inte.
    actions = [
        action
        for action in suggest_actions(customer, pipeline, price_model, thresholds)
        if action["probability"] is None or action["probability"] < probability
    ]
    if actions:
        st.dataframe(alternatives_table(actions), width="stretch", hide_index=True)
    else:
        st.caption("Inget alternativ bedöms sänka risken.")
    st.caption(DISCLAIMER)


def overview_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Arbetslistan med kundvänliga namn. Priset förblir ett tal så att kolumnen sorterar rätt."""
    table = rows.copy()
    table["level"] = table["level"].str.capitalize()
    for column in ["Contract", "InternetService", "PaymentMethod"]:
        table[column] = table[column].map(VALUE_LABELS[column])
    return table.rename(columns=LIST_HEADERS)


def pick_from_list(hits: pd.DataFrame, query: str) -> str | None:
    """Visar träffarna 25 per sida och returnerar kund-id för vald rad, annars None.

    Nyckeln innehåller sökning och sida så att valet nollställs när listan byter innehåll.
    """
    n_pages = math.ceil(len(hits) / PAGE_SIZE)
    page = st.number_input("Sida", 1, n_pages, 1, key=f"sida_{query}") if n_pages > 1 else 1
    start = (page - 1) * PAGE_SIZE
    shown = hits.iloc[start : start + PAGE_SIZE]
    st.caption(
        f"Rad {start + 1}–{start + len(shown)} av {len(hits)}. Klicka på en rad för att ladda "
        "kunden, på en kolumnrubrik för att sortera."
    )
    event = st.dataframe(
        overview_table(shown),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=f"lista_{query}_{page}",
        column_config={LABELS["MonthlyCharges"]: st.column_config.NumberColumn(format="%.2f $")},
    )
    rows = event.selection.rows
    return str(shown[ID_COLUMN].iloc[rows[0]]) if rows else None


new_tab, existing_tab = st.tabs(["Ny kund", "Befintlig kund"])

with new_tab:
    st.caption(
        "Ny kund har kundtid 0. Priset predikteras först och går sedan in i riskbedömningen."
    )
    values = customer_form({}, key="ny")
    if values is None:
        st.info("Fyll i alla fält så räknas pris och risknivå fram.")
    else:
        customer, monthly_charges, probability = evaluate(values, None)
        st.subheader("Resultat")
        show_result(customer, monthly_charges, probability)
        if st.button("Spara kund", key="spara_ny"):
            # Sparas i registret new_customers - aldrig i träningsdatan.
            customer_id = db.save_new_customer(
                {**values, "MonthlyCharges": monthly_charges}, db.DEFAULT_DB_PATH
            )
            st.success(f"Kunden sparades med id {customer_id}. Sök på id:t under Befintlig kund.")

with existing_tab:
    overview = load_or_stop(get_overview, "arbetslistan")
    query = st.text_input(
        "Sök kund-id",
        placeholder="t.ex. 7590, 7590-VHVEG eller NEW",
        help="Siffror matchar början av kund-id, annan text var som helst i id:t.",
    ).strip()
    hits = search_customers(overview, query)
    customer_id = None
    if hits.empty:
        st.error(f"Ingen kund matchar {query!r}. Kontrollera id:t och försök igen.")
    else:
        customer_id = pick_from_list(hits, query)
        if customer_id is None and query and len(hits) == 1:
            customer_id = str(hits[ID_COLUMN].iloc[0])  # exakt en träff laddas direkt

    row = find_customer(customer_id) if customer_id else None
    if row is not None:
        stored = stored_values(row)
        c1, c2 = st.columns(2)
        c1.metric("Dagens pris", f"{stored['MonthlyCharges']:.2f} $/mån", border=True)
        c2.metric("Kundtid", months(stored["tenure"]), border=True)

        # Segmentet gäller kunden som den är sparad - räknas vid sök, inte vid varje justering.
        with st.spinner("Jämför med liknande kunder …"):
            comparison = get_segment_comparison(customer_id)
        ratio = comparison["segment_probability"] / comparison["average_probability"]
        st.markdown(
            f"**Kunder som liknar den här** bedöms av modellen ha risknivå "
            f"{risk_level(comparison['segment_probability'], thresholds)}, "
            f"ungefär {ratio:.1f} gånger snittkundens risk. Snittkunden ligger på nivå "
            f"{risk_level(comparison['average_probability'], thresholds)}."
        )

        st.caption("Justera fälten för att se hur pris och risknivå ändras (what-if).")
        values = customer_form(stored, key=f"befintlig_{customer_id}")
        if values is None:
            st.info("Fyll i de nya fälten så räknas pris och risknivå fram.")
        else:
            # Oförändrad konfiguration: dagens faktiska pris. Ändrad: prismodellen räknar om.
            changed = any(values[col] != stored[col] for col in PRICE_FEATURE_COLUMNS)
            customer, monthly_charges, probability = evaluate(
                values, None if changed else stored["MonthlyCharges"]
            )
            st.subheader("Resultat")
            if changed:
                st.caption("Ändrat från dagens läge - pris och risknivå är omräknade.")
            show_result(customer, monthly_charges, probability)
