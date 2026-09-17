"""Kundvänliga namn på svenska för kolumner och kategorivärden.

Används av riskfaktorerna i src/risk.py och av Säljverktyg-sidan. Datasettet rörs inte -
här översätter vi bara vad som visas för säljaren.
"""

from __future__ import annotations

# Kolumn -> kundvänligt namn. Täcker alla RAW_FEATURE_COLUMNS i src/data.py.
LABELS: dict[str, str] = {
    "SeniorCitizen": "Pensionär",
    "tenure": "Kundtid (månader)",
    "MonthlyCharges": "Månadskostnad",
    "TotalCharges": "Total kostnad",
    "gender": "Kön",
    "Partner": "Partner",
    "Dependents": "Anhöriga i hushållet",
    "PhoneService": "Telefoni",
    "MultipleLines": "Flera linjer",
    "InternetService": "Internet",
    "OnlineSecurity": "Onlinesäkerhet",
    "OnlineBackup": "Onlinebackup",
    "DeviceProtection": "Enhetsskydd",
    "TechSupport": "Teknisk support",
    "StreamingTV": "Streamad TV",
    "StreamingMovies": "Streamade filmer",
    "Contract": "Avtal",
    "PaperlessBilling": "Fakturering",
    "PaymentMethod": "Betalsätt",
}

# Kolumn -> värde -> text. Täcker alla värden i alla RAW_CATEGORICAL_COLUMNS.
# Texten ska kunna stå för sig själv i en riskfaktor, t.ex.
# "Ingen teknisk support: 42 % churn mot 27 % i snitt" - därför "Ingen teknisk support"
# i stället för bara "No".
VALUE_LABELS: dict[str, dict[str, str]] = {
    "gender": {"Female": "Kvinna", "Male": "Man"},
    "Partner": {"Yes": "Har partner", "No": "Ingen partner"},
    "Dependents": {"Yes": "Har anhöriga i hushållet", "No": "Inga anhöriga i hushållet"},
    "PhoneService": {"Yes": "Telefoni", "No": "Ingen telefoni"},
    "MultipleLines": {
        "Yes": "Flera telefonlinjer",
        "No": "En telefonlinje",
        "No phone service": "Ingen telefoni",
    },
    "InternetService": {"DSL": "DSL", "Fiber optic": "Fiber", "No": "Inget internet"},
    "OnlineSecurity": {
        "Yes": "Onlinesäkerhet",
        "No": "Ingen onlinesäkerhet",
        "No internet service": "Inget internet",
    },
    "OnlineBackup": {
        "Yes": "Onlinebackup",
        "No": "Ingen onlinebackup",
        "No internet service": "Inget internet",
    },
    "DeviceProtection": {
        "Yes": "Enhetsskydd",
        "No": "Inget enhetsskydd",
        "No internet service": "Inget internet",
    },
    "TechSupport": {
        "Yes": "Teknisk support",
        "No": "Ingen teknisk support",
        "No internet service": "Inget internet",
    },
    "StreamingTV": {
        "Yes": "Streamad TV",
        "No": "Ingen streamad TV",
        "No internet service": "Inget internet",
    },
    "StreamingMovies": {
        "Yes": "Streamade filmer",
        "No": "Inga streamade filmer",
        "No internet service": "Inget internet",
    },
    "Contract": {
        "Month-to-month": "Månadsavtal",
        "One year": "Ettårsavtal",
        "Two year": "Tvåårsavtal",
    },
    "PaperlessBilling": {"Yes": "E-faktura", "No": "Pappersfaktura"},
    "PaymentMethod": {
        "Electronic check": "Elektronisk check",
        "Mailed check": "Check per post",
        "Bank transfer (automatic)": "Automatisk banköverföring",
        "Credit card (automatic)": "Automatisk kortbetalning",
    },
}
