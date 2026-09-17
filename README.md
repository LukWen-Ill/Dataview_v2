# Churn-prediktion – Telco Customer Churn

Fullstack-ML-projekt som förutsäger vilka telekomkunder som riskerar att säga upp sitt
abonnemang. Data lagras i SQLite, modelleringen görs med scikit-learn och resultatet
presenteras i en Streamlit-app.

![CI](https://github.com/LukWen-Ill/Dataview_v2/actions/workflows/ci.yml/badge.svg)

Projektarbete Del 2 i kursen *AI – teori och tillämpning, del 1* (NBI/Handelsakademin).

## Vad projektet gör

"Churn" betyder att en kund lämnar. Appen svarar på två frågor (plus en extra):

- **Vilka kunder riskerar att lämna?** – en klassificeringsmodell ger en sannolikhet per kund.
  Användaren väljer threshold och ser hur avvägningen mellan precision och recall ändras.
- **Vilka typer av kunder har vi?** – K-Means grupperar kunderna i segment som sedan
  tolkas med churn-andel per segment.
- **Hur mycket intäkt förväntas gå förlorad?** (extra – inte ett kurskrav) – Ledning-sidan
  rullar fram dagens aktiva kunder 24 månader med modellens sannolikheter.

Hela flödet:

```
CSV  →  SQLite  →  rensning + feature engineering  →  train / validation / test
     →  3 modeller i sklearn-Pipeline + GridSearchCV  →  modellval på validation
     →  utvärdering på test  →  joblib  →  Streamlit  →  prediktioner loggas i SQLite
```

## Dataset

`data/raw/telco_churn.csv` – IBM:s [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn):
7 043 kunder, 21 kolumner. Target är `Churn` (Yes/No), 26,5 % av kunderna har lämnat.
`customerID` används inte som feature.

Tre härledda kolumner skapas i `src/data.py` (`add_features`):

| Kolumn | Vad | Varför |
|---|---|---|
| `num_addon_services` | Antal tilläggstjänster (0–6) | En kund med många tjänster är mer inlåst |
| `avg_monthly_charge` | `TotalCharges / tenure` | Fångar om priset ändrats över kundtiden |
| `tenure_group` | Kundtid i grupper (0–12, 13–24, 25–48, 49+) | Churn är koncentrerad till första året |

## ML-problem

Binär klassificering med obalanserade klasser. Accuracy räcker därför inte: en modell som
alltid svarar "stannar" får 73 % rätt. Vi jämför modellerna på **precision, recall, F1 och
ROC-AUC** och resonerar utifrån affärsperspektivet att en missad churnare kostar mer än en
onödig kontakt. Därför används `class_weight="balanced"` i alla modeller.

## Tech stack

Python 3.11+, pandas, scikit-learn, SQLite (`sqlite3` i standardbiblioteket), joblib,
Streamlit, Altair, matplotlib. Tester med pytest, lint med ruff, CI på GitHub Actions.

## Projektstruktur

```
data/raw/telco_churn.csv   rådata
data/churn.db              SQLite (byggs med python -m src.db, versionshanteras inte)
models/                    tränad modell + resultat (versionshanteras så appen startar direkt)
  churn_model.joblib         slutmodellen (hela pipelinen)
  results.json               jämförelse på validation, bästa params, testresultat
  test_predictions.csv       slutmodellens sannolikheter på testmängden
src/data.py                inläsning, schemavalidering, rensning, feature engineering
src/db.py                  SQLite: tabeller customers, model_runs, predictions
src/features.py            ColumnTransformer: imputering, skalning, one-hot
src/models.py              de tre modellerna och deras hyperparametergrids
src/train.py               hela träningsflödet – börja läsa här
src/evaluate.py            metrics, threshold-tabell, confusion matrix, ROC, feature importance
src/segment.py             K-Means, elbow/silhouette, PCA, segmentprofiler
src/eda.py                 aggregeringar för EDA-sidan
src/dashboard.py           filtrering, intäktsprognos och nyckeltal för Ledning-sidan (extra)
app.py                     Streamlit: startsida (översikt)
pages/                     Streamlit: Data, Modeller, Segmentering, Prediktera, Ledning (extra)
app_helpers.py             cachade laddningsfunktioner för sidorna
tests/                     pytest (162 tester)
docs/rapport.md            teknisk rapport
.github/workflows/ci.yml   CI
```

## Installation

```bash
git clone https://github.com/LukWen-Ill/Dataview_v2.git
cd Dataview_v2

python -m venv .venv
.venv\Scripts\activate          # Windows (Git Bash: source .venv/Scripts/activate)
source .venv/bin/activate       # macOS/Linux

pip install -r requirements-dev.txt
```

## Databas

Databasen byggs från CSV:n med ett kommando:

```bash
python -m src.db
```

Det skapar `data/churn.db` med tre tabeller:

| Tabell | Innehåll |
|---|---|
| `customers` | Rådatan, en rad per kund |
| `model_runs` | En rad per utvärderad modell varje gång träningen körs (params + metrics som JSON) |
| `predictions` | En rad per prediktion som görs i appen (indata, sannolikhet, threshold, beslut) |

Appen och träningsskriptet bygger databasen automatiskt om den saknas.

## Träning

```bash
python -m src.train
```

Tar cirka 20–30 sekunder och gör följande (se `src/train.py`):

1. Läser kunderna från SQLite.
2. Delar stratifierat i 60 % train / 20 % validation / 20 % test.
3. Kör `GridSearchCV` (5-delad korsvalidering, scoring ROC-AUC) på train för logistisk
   regression, beslutsträd och random forest.
4. Jämför de tre på validation och väljer den med högst ROC-AUC.
5. Tränar om den valda modellen på train + validation.
6. Utvärderar en enda gång på test.
7. Sparar `models/churn_model.joblib`, `models/results.json`, `models/test_predictions.csv`
   och loggar alla körningar i tabellen `model_runs`.

Den tränade modellen är incheckad i repot, så appen fungerar direkt efter klon.
Träna om när koden i `src/` ändras.

## Streamlit

```bash
streamlit run app.py
```

Appen laddar den sparade modellen – den tränar aldrig själv. Saknas modellen visas ett
felmeddelande med träningskommandot.

| Sida | Innehåll |
|---|---|
| Översikt | Antal kunder, churn-andel, vald modell, testresultat, modelljämförelse |
| Data | EDA: churn per kategori, kundtid, månadskostnad, korrelationer |
| Modeller | Jämförelse på validation, slutresultat på test, threshold-slider med confusion matrix, ROC-kurva, classification report, feature importance |
| Segmentering | Elbow och silhouette, K-Means-kluster i PCA-rummet, segmentprofiler med churn-andel |
| Prediktera | En kund via formulär (loggas i databasen) eller många via CSV, med valbar threshold |
| Ledning (extra) | Intäktsprognos 24 månader framåt för aktiva kunder: KPI-rad, blå/röd linje för kvarvarande intäkt och ackumulerad förlust, filter på avtal, internet, betalsätt och K-Means-segment, sorterbar tabell per grupp |

### Ledning (extra – inte ett kurskrav)

`Churn = Yes` i datasettet betyder "lämnade senaste månaden", så modellens sannolikhet p
tolkas som risk per månad. Varje aktiv kund (`Churn = No`) rullas fram med (1 − p)^m för
m = 0–24 månader. Kvarvarande MRR är Σ MonthlyCharges × (1 − p)^m och förlusten en månad
är skillnaden mot månaden före. Beräkningarna ligger i `src/dashboard.py`; sidan
`pages/5_Ledning.py` visar bara resultatet och tränar ingenting om.
Blå linje är dagens aktiva kunders kvarvarande intäkt plus en **mockad nykundsförsäljning**;
röd linje är den ackumulerade förlusten på dagens kunder, alltså vad som händer om vi inte gör
något. Nykundsmocken (`NEW_CUSTOMER_MOCK` i
`src/dashboard.py`): ca 600 nya kunder månad 1 (datasetets senaste kohort), +5 % per månad
med ±20 % seedad slump, 50 $ per kund och samma churn-risk som de befintliga. Hela prognosen
antar konstant churn-risk per kund och månad samt frysta priser.
Sannolikheterna är inte kalibrerade (`class_weight="balanced"`), så kurvan är brant.

## Tester och lint

```bash
pytest --cov=src                        # 162 tester, täckningskrav 90 %
ruff check . && ruff format --check .
```

Testsviten är byggd för att inte passera tyst: `filterwarnings = error`, `--strict-markers`,
`xfail_strict`, inga `continue-on-error` i CI. Felfall testas, inte bara happy path: saknad
fil, saknad kolumn, tom dataram, ogiltig target, okänd kategori vid prediktion, saknade
värden, threshold utanför 0–1, för få kluster, saknad modell och saknad databas i appen.

## Git och GitHub

`main` är alltid körbar. Allt arbete sker på feature-branches (`feature/<namn>`) och går in
via pull request. CI kör lint, tester (Python 3.11 och 3.12) samt en end-to-end-träning på
varje PR, och måste vara grön före merge.

## Resultat

Modelljämförelse på valideringsmängden (1 409 kunder, threshold 0,5):

| Modell | CV ROC-AUC | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistisk regression (C=10) | 0,849 | 0,748 | 0,517 | 0,786 | 0,624 | 0,838 |
| Beslutsträd (depth 5) | 0,819 | 0,740 | 0,506 | 0,786 | 0,616 | 0,817 |
| **Random forest** (200 träd, depth 10) | 0,847 | 0,754 | 0,524 | 0,775 | 0,626 | **0,838** |

Random forest valdes (marginellt högst ROC-AUC på validation). Slutmodellen på
**testmängden** (1 409 kunder som aldrig använts för träning eller modellval):

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0,759 | 0,532 | 0,783 | 0,634 | 0,841 |

Modellen hittar alltså 78 % av de kunder som faktiskt lämnar, till priset av att ungefär
hälften av de flaggade kunderna inte hade lämnat. Med threshold 0,3 stiger recall till 91 %
(781 kunder flaggas, 35 churnare missas); med 0,7 sjunker den till 53 % (305 flaggas, 177
missas). Viktigaste features: `Contract_Month-to-month`, `tenure`, `TotalCharges`,
`avg_monthly_charge`, `OnlineSecurity_No`.

Segmenteringen (k = 4) ger bland annat ett segment med månadsavtal, hög månadskostnad och få
tilläggstjänster där 56 % churnar, mot 7 % i segmentet med billiga tvåårsavtal.

## Kända begränsningar

- Modellerna ligger nära varandra; logistisk regression är nästan lika bra som random
  forest och enklare att tolka. Valet gjordes strikt på validerings-ROC-AUC.
- SQLite-databasen är lokal. På Streamlit Community Cloud byggs den om vid varje start, så
  loggade prediktioner där är flyktiga.
- Datasettet är ett tvärsnitt utan tidsstämplar, så vi kan inte utvärdera modellen "framåt i
  tiden" som man skulle göra i drift.
