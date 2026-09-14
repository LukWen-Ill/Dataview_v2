# Churn-prediktion

Fullstack-app som räknar ut hur sannolikt det är att en telekomkund säger upp sitt abonnemang.
Data → scikit-learn-modell → Streamlit-gränssnitt, med tester och CI/CD.

![CI](https://github.com/LukWen-Ill/Dataview_v2/actions/workflows/ci.yml/badge.svg)

## Vad appen gör

"Churn" betyder att en kund lämnar. Appen tittar på vad vi vet om en kund - hur länge hen
varit kund, vilket avtal hen har, vad hen betalar per månad, vilka tjänster hen använder -
och svarar med en sannolikhet mellan 0 och 100 %.

Tre flikar:
- **Data** - hur många kunder, hur stor andel som lämnar, churn per avtalstyp.
- **Modell** - hur bra modellen är, mätt på data den aldrig sett under träningen.
- **Prediktera** - fyll i en kund för hand, eller ladda upp en CSV med många kunder och få
  tillbaka en lista sorterad på risk.

## Data

`data/raw/telco_churn.csv` - IBM:s "Telco Customer Churn", 7 043 kunder och 21 kolumner.
26,5 % av kunderna har lämnat. Filen ligger i repot (977 KB).

Kolumnen `Churn` (`Yes`/`No`) är det vi försöker förutsäga. `customerID` används inte som
feature - ett kundnummer säger inget om beteende.

## Modell

En `Pipeline` i scikit-learn med två steg:

1. **Förbehandling.** Numeriska kolumner medianimputeras och skalas. Kategoriska kolumner
   one-hot-kodas med `handle_unknown="ignore"`, så att en avtalstyp modellen aldrig sett
   förut ger en prediktion i stället för en krasch.
2. **Logistisk regression** med `class_weight="balanced"`.

Resultat på testmängden (20 % av datan, som modellen aldrig tränats på):

| Mått | Värde |
|---|---|
| ROC-AUC | 0,84 |
| Recall | 0,78 |
| Precision | 0,50 |
| Träffsäkerhet | 0,74 |

Recall är medvetet prioriterad: det kostar mer att missa en kund som är på väg att lämna än
att kontakta en kund som ändå hade stannat.

## Kom igång

```bash
git clone https://github.com/LukWen-Ill/Dataview_v2.git
cd Dataview_v2

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

pip install -r requirements-dev.txt
```

Kör appen:

```bash
streamlit run app.py
```

Träna modellen från kommandoraden (skriver metrics som JSON och sparar `models/churn_model.joblib`):

```bash
python -m src.train
```

## Tester

```bash
pytest --cov=src
```

59 tester. Uppsättningen är byggd för att aldrig passera tyst:

- `filterwarnings = error` - en varning från vår egen kod failar bygget.
- `--strict-markers --strict-config` och `xfail_strict = true`.
- Täckningsgräns på 90 % (`fail_under` i `pyproject.toml`).
- Inga `continue-on-error` eller `|| true` i CI.

Felfall som testas, inte bara happy path:

| Vad som går fel | Förväntat beteende |
|---|---|
| Datafilen saknas | `FileNotFoundError` med sökvägen i meddelandet |
| Kolumn saknas | `SchemaError` som listar exakt vilka |
| Tom dataram | `SchemaError` |
| `Churn` innehåller `"Kanske"` | `SchemaError` som namnger värdet |
| `TotalCharges` är `" "` eller skräptext | blir `NaN`, imputeras - kraschar inte |
| Okänd kategori vid prediktion (`PaymentMethod = "Swish"`) | kodas som nollor, ger ändå en sannolikhet |
| Targeten har bara en klass | `ValueError` |
| `X` och `y` har olika längd | `ValueError` |
| Modellfil saknas | `FileNotFoundError` som säger hur man tränar |
| Streamlit-appen startar utan data | rött felmeddelande, inte en stacktrace |

Appen röktestas med Streamlits `AppTest`, som kör hela `app.py` och failar på varje
oväntat undantag.

## CI/CD

`.github/workflows/ci.yml` kör på varje push till `main` och varje pull request:

1. **lint** - `ruff check` och `ruff format --check`.
2. **test** - `pytest --cov=src` på Python 3.11 och 3.12.
3. **train** - kör `python -m src.train` end-to-end och laddar upp modellen som artifact.
   Körs bara om testerna är gröna.

**CD:** Streamlit Community Cloud är kopplad till `main` och deployar om automatiskt vid
varje merge. Koppling görs en gång på [share.streamlit.io](https://share.streamlit.io):
peka på repot `LukWen-Ill/Dataview_v2`, branch `main`, main file `app.py`, Python 3.11.

## Arbetsflöde

Basic branching:

```
main                 alltid deploybar
 └─ feature/xyz      allt arbete, in via PR, CI måste vara grön
```

## Projektstruktur

```
data/raw/telco_churn.csv   rådata
src/data.py                inläsning, schemavalidering, rensning
src/features.py            förbehandling
src/model.py               pipeline, träning, metrics, spara/ladda
src/train.py               CLI
app.py                     Streamlit-entrypoint
tests/                     pytest
.github/workflows/ci.yml   CI
```
