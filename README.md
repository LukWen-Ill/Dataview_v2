# Churn-prediktion – Telco Customer Churn

Fullstack-ML-projekt som förutsäger vilka telekomkunder som riskerar att säga upp sitt
abonnemang. Data lagras i SQLite, modelleringen görs med scikit-learn och resultatet
presenteras i en Streamlit-app.

![CI](https://github.com/LukWen-Ill/Dataview_v2/actions/workflows/ci.yml/badge.svg)

Projektarbete Del 2 i kursen *AI – teori och tillämpning, del 1* (NBI/Handelsakademin).

## Vad projektet gör

"Churn" betyder att en kund lämnar. Appen svarar på två frågor:

- **Vilka kunder riskerar att lämna?** – en klassificeringsmodell ger en sannolikhet per kund.
  Användaren väljer threshold och ser hur avvägningen mellan precision och recall ändras.
- **Vilka typer av kunder har vi?** – K-Means grupperar kunderna i segment som sedan
  tolkas med churn-andel per segment.

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
models/                    tränade modeller + resultat (versionshanteras så appen startar direkt)
  churn_model.joblib         slutmodellen (hela pipelinen)
  results.json               jämförelse på validation, bästa params, testresultat
  test_predictions.csv       slutmodellens sannolikheter på testmängden
  price_model.joblib         prismodellen för Säljverktyget (extra)
  price_results.json         prismodellens jämförelse och testresultat (extra)
src/data.py                inläsning, schemavalidering, rensning, feature engineering
src/db.py                  SQLite: tabeller customers, new_customers, model_runs, predictions
src/features.py            ColumnTransformer: imputering, skalning, one-hot
src/models.py              de tre modellerna och deras hyperparametergrids
src/train.py               hela träningsflödet – börja läsa här
src/evaluate.py            metrics, threshold-tabell, confusion matrix, ROC, feature importance
src/segment.py             K-Means, elbow/silhouette, PCA, segmentprofiler
src/eda.py                 aggregeringar för EDA-sidan
src/price.py               (extra) regressionsmodell för månadspris (CLI: python -m src.price)
src/risk.py                (extra) risknivåer (tertiler), riskfaktorer, segmentjämförelse
src/labels.py              (extra) kundvänliga namn på kolumner och värden
src/actions.py             (extra) åtgärdskatalog med what-if genom pris- och churnmodell
app.py                     Streamlit: startsida (översikt)
pages/                     Streamlit: Data, Modeller, Segmentering, Prediktera, Säljverktyg (extra)
app_helpers.py             cachade laddningsfunktioner för sidorna
tests/                     pytest (212 tester)
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

Det skapar `data/churn.db` med fyra tabeller:

| Tabell | Innehåll |
|---|---|
| `customers` | Rådatan, en rad per kund. Byggs om från CSV:n, appen skriver aldrig här |
| `new_customers` | Kunder sparade från Säljverktyget (extra). Ingen `Churn`-kolumn, läses aldrig av träningen |
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

### Prismodell (extra)

```bash
python -m src.price
```

Tränar regressionsmodellen som ger Säljverktyget ett månadspris: linjär regression och random
forest i `Pipeline`, `GridSearchCV` med 5-delad CV, samma train/validation/test-rader som
churnmodellen. Features är kundens tjänster, avtal, betalsätt, demografi och kundtid – inte
`TotalCharges`, som är en funktion av priset. Sparar `models/price_model.joblib` och
`models/price_results.json` (incheckade – träna om och committa när något som påverkar
modellen ändras) och loggar körningen i `model_runs`. Testresultat: MAE 0,78 $, RMSE 1,01 $,
R² 0,999. Det höga R² är väntat: priset i datan är nästan en summa av tjänsterna.

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
| Säljverktyg (extra) | Ny eller befintlig kund: predikterat pris, risknivå låg/medel/hög, riskfaktorer och alternativ ur åtgärdskatalogen |

### Säljverktyg (extra)

Sidan ligger utanför kursens krav och är byggd för en säljare med kunden i telefon.

- **Ny kund.** Formulär i samtalets ordning: hushåll → telefoni → internet → tillägg (visas
  bara om kunden har internet) → avtal, fakturering, betalsätt. Prismodellen predikterar
  månadspriset, som går in i churnmodellen som `MonthlyCharges`. Sidan visar pris, risknivå,
  de största riskfaktorerna och upp till tre alternativ ur åtgärdskatalogen (`src/actions.py`)
  med pris och bedömd risknivå sida vid sida. "Spara kund" lägger kunden i `new_customers`.
- **Befintlig kund.** Sök på kund-id i träningsdatan eller registret. Fälten förifylls och
  sidan visar dagens pris, risknivå, jämförelse med liknande kunder (K-Means) och alternativ.
  Fälten kan justeras för what-if – pris och risknivå räknas om.

Risknivån är relativ: tertiler av churnmodellens sannolikheter över träningsdatan (gränser
0,188 och 0,568). Rå procent visas inte, eftersom `class_weight="balanced"` gör
sannolikheterna uppblåsta. Alternativen är modellens bedömning av samband i datan, inte
orsakspåståenden, och ett alternativ ändrar aldrig kundtid eller tar bort en tjänst.
Nya kunder läses aldrig av träningen: `python -m src.train` ger samma split och samma
`results.json` oavsett vad som lagts in via appen.

## Tester och lint

```bash
pytest --cov=src                        # 212 tester, täckningskrav 90 %
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
