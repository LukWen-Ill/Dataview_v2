# CLAUDE.md

Projektspecifika regler. Läses tillsammans med de globala reglerna i `~/.claude/CLAUDE.md`.

## Vad projektet är

Fullstack-app som predikterar kundchurn. Telco Customer Churn-datasettet (7 043 kunder,
21 kolumner, 26,5 % churn) → scikit-learn-pipeline → Streamlit-gränssnitt.

Inlämning KK2, kursen "Tillämpad maskininlärning med Python".

## Scope

I scope:
- Binär klassificering: churnar kunden eller inte.
- Flera modeller i sklearn-`Pipeline`, tränade på `data/raw/telco_churn.csv` och jämförda på samma
  split. Hyperparametertuning med `GridSearchCV` som del av träningen.
- Streamlit-app med datautforskning, modellmetrics och prediktion (en kund + CSV-batch).
- CI på GitHub Actions: lint, tester, end-to-end-träning.
- CD: Streamlit Community Cloud deployar automatiskt från `main`.

Utanför scope (lägg inte till utan att fråga):
- Deep learning.
- Databas, användarinloggning, API-lager, Docker.
- Ommärkning eller schemaändringar av datasettet.

## Filstruktur

```
data/raw/telco_churn.csv   rådata, versionshanterad (977 KB)
src/data.py                inläsning, schemavalidering, rensning
src/features.py            ColumnTransformer: skalning + one-hot
src/model.py               pipeline, träning, metrics, spara/ladda
src/train.py               CLI: python -m src.train
.notebooks/                notebooks som redovisar träningen
app.py                     Streamlit-entrypoint
tests/                     pytest
.github/workflows/ci.yml   CI
```

## Arbetssätt

- **Code-first.** All logik bor i `src/`. `app.py` är bara presentation - ingen ML-logik där.
  Notebooks ligger i `.notebooks/` och redovisar träningen genom att importera från `src/` -
  de är aldrig den enda platsen där en modell tränas.
- **Basic branching.** `main` är skyddad i praktiken: allt arbete sker på feature-branch
  (`feature/<kort-namn>`), går in via PR, och CI måste vara grön innan merge.
- **Tester får inte passera tyst.** Konkret betyder det:
  - `filterwarnings = error` i `pyproject.toml` - en varning failar bygget.
  - `--strict-markers --strict-config`, `xfail_strict = true`.
  - Täckningsgräns `fail_under = 90`.
  - Inga `|| true`, inga `continue-on-error` i CI.
  - Inga `pytest.skip` utan att skälet står i koden.
- **Mer än happy path.** Varje ny funktion i `src/` ska ha minst ett test för felfallet:
  saknad kolumn, tom dataram, ogiltigt targetvärde, okänd kategori vid prediktion,
  saknade värden, fil som inte finns.

## Kommandon

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements-dev.txt

python -m src.train          # träna, skriv metrics, spara modell
streamlit run app.py         # starta appen
pytest --cov=src             # testa med täckningsgräns
ruff check . && ruff format --check .
```

## Kända begränsningar

- Modellen tränas om vid appstart (cachas med `st.cache_resource`) i stället för att läsa en
  incheckad `.joblib`. Det undviker versionskrockar mellan sklearn i repo och i molnet.
  Träningen tar under en sekund på 7 000 rader.
- `class_weight="balanced"` prioriterar recall framför precision - modellen hellre flaggar en
  kund i onödan än missar en som faktiskt churnar.
- 11 rader har tom `TotalCharges` (alla med `tenure = 0`). De blir `NaN` och medianimputeras.
