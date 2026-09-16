# CLAUDE.md

Projektspecifika regler för Claude Code. Läs tillsammans med README.md.

## Vad projektet är

Studentprojekt (3 personer, betyg IG/G) i kursen "AI – teori och tillämpning, del 1":
Projektarbete Del 2. Fullstack-ML-flöde som predikterar kundchurn på Telco Customer
Churn-datasettet (7 043 kunder, 21 kolumner, 26,5 % churn).

Uppgiftens krav: data i databas, ML-modellering i Python, Streamlit-frontend, Git/GitHub,
tydlig README, teknisk rapport (~3 sidor, `docs/rapport.md`).

## Kodens nivå – viktigast av allt

Studenterna ska kunna öppna `src/train.py` och förstå hela ML-flödet, och förklara varje del
för utbildaren. Därför:

- Enkel och tydlig kod före smart kod. Små funktioner, tydliga namn, kommentarer där de hjälper.
- Inga onödiga abstraktionslager, design patterns eller "enterprise"-kod.
- Kursbegreppen ska synas i koden: train/validation/test, stratifiering, k-fold CV,
  GridSearchCV, Pipeline, confusion matrix, precision/recall/F1/ROC-AUC, threshold,
  feature importance, PCA, K-Means, joblib.
- Lägg inte till saker för att det ser avancerat ut. Det som inte krävs av uppgiften
  markeras som "extra" och implementeras inte utan att fråga.
- Vid konflikt mellan uppgift, kod och tester: fråga, gissa inte.

## Scope

I scope:
- SQLite-databas (`data/churn.db`) med kunddata, loggade modellkörningar och prediktioner.
- Binär klassificering med tre modeller (logistisk regression, beslutsträd, random forest)
  i sklearn-`Pipeline`, små grids i `GridSearchCV` med 5-fold CV.
- 60/20/20 stratifierad split. Testmängden används bara för slutlig utvärdering.
- Streamlit med sex sidor: Översikt, Data (EDA), Modeller, Segmentering, Prediktera,
  Ledning (extra).
- K-Means + PCA för kundsegmentering.
- CI på GitHub Actions: lint, tester, end-to-end-träning.

Utanför scope (fråga först):
- Deep learning, SVM eller fler modeller.
- Användarinloggning, API-lager, Docker, annan databas än SQLite.
- Ommärkning eller schemaändringar av datasettet.

## Filstruktur

```
data/raw/telco_churn.csv   rådata, versionshanterad
data/churn.db              SQLite, byggs av python -m src.db, gitignorerad
models/                    churn_model.joblib, results.json, test_predictions.csv – versionshanterade
src/data.py                inläsning, schemavalidering, rensning, feature engineering
src/db.py                  SQLite-funktioner
src/features.py            ColumnTransformer
src/models.py              modellkatalog + grids
src/train.py               hela träningsflödet (CLI: python -m src.train)
src/evaluate.py            metrics, threshold, confusion matrix, ROC, feature importance
src/segment.py             K-Means, PCA, segmentprofiler
src/eda.py                 aggregeringar för EDA-sidan
src/dashboard.py           filtrering, prognos och nyckeltal för Ledning-sidan (extra)
app.py, pages/, app_helpers.py   Streamlit – bara presentation
pages/5_Ledning.py         Ledning-sidan (extra)
tests/                     pytest
docs/rapport.md            teknisk rapport
```

## Arbetssätt

- **Code-first.** All ML-logik i `src/`. `app.py`, `pages/` och `app_helpers.py` visar bara
  resultat. Aggregeringar för grafer får ligga i `src/eda.py`.
- **Alternativ A för modellen.** `python -m src.train` skapar `models/*` som checkas in.
  Appen laddar den sparade modellen och kör aldrig GridSearchCV. Träna om och committa när
  något i `src/` som påverkar modellen ändras.
- **Basic branching.** Allt arbete på `feature/<kort-namn>`, in via PR, CI grön före merge.
  Ändra aldrig `main` direkt. Små, begripliga commits.
- **Tester får inte passera tyst.** `filterwarnings = error`, `--strict-markers`,
  `--strict-config`, `xfail_strict = true`, `fail_under = 90`. Inga `|| true` eller
  `continue-on-error` i CI. Inga `pytest.skip` utan skäl i koden.
- **Mer än happy path.** Varje ny funktion i `src/` ska ha minst ett test för felfallet.
- **Förklara kort** vid viktiga ändringar: vad, varför, hur det passar ML-flödet, vilket
  kursbegrepp det motsvarar.

## Kommandon

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows (Git Bash)
pip install -r requirements-dev.txt

python -m src.db             # bygg data/churn.db från CSV:n
python -m src.train          # träna, jämför, spara models/* och logga i databasen
streamlit run app.py         # starta appen (laddar sparad modell)
pytest --cov=src             # tester med täckningskrav
ruff check . && ruff format --check .
```

## Kända begränsningar och beslut

- `class_weight="balanced"` på alla modeller: churn är minoritetsklass och en missad
  churnare kostar mer än en onödig kontakt. Ger högre recall, lägre precision.
- GridSearchCV använder `scoring="roc_auc"` eftersom det är oberoende av threshold –
  threshold väljs separat i appen.
- Modellval sker på validation med ROC-AUC. Modellerna ligger nära varandra.
- 11 rader har tom `TotalCharges` (alla med `tenure = 0`). De blir NaN, liksom
  `avg_monthly_charge`, och medianimputeras i pipelinen.
- Lokalt körs Python 3.14, CI kör 3.11/3.12. Paketen är exakt pinnade i `requirements.txt`.
- Bokens PDF (kap 1–6) är bildbaserad utan textlager – läs den som bilder om den behövs.
