# Churn-prediktion på Telco Customer Churn – teknisk rapport

**Kurs:** AI – teori och tillämpning, del 1 · **Uppgift:** Projektarbete Del 2
**Grupp:** [FYLL I: namn, namn, namn]
**Repo:** https://github.com/LukWen-Ill/Dataview_v2 · **App:** [FYLL I: länk till Streamlit Cloud om deployad]

---

## 1. Bakgrund

Ett telekombolag förlorar pengar varje gång en kund säger upp sitt abonnemang ("churn").
Det är dyrare att vinna en ny kund än att behålla en befintlig, så bolaget vill veta *vilka*
kunder som riskerar att lämna för att kunna agera i tid – till exempel med ett erbjudande om
längre avtal.

Vi valde datasettet **Telco Customer Churn** (IBM, via Kaggle): 7 043 kunder och 21 kolumner
med avtalsform, tjänster, betalsätt, kundtid och kostnader. Target är `Churn` (Yes/No).
26,5 % av kunderna har lämnat, alltså är klasserna obalanserade. Det gör problemet till en
**binär klassificering** där accuracy inte räcker som mått – en modell som alltid svarar
"stannar" får 73 % rätt utan att hitta en enda churnare.

Målet med projektet var att bygga ett komplett fullstack-flöde enligt kursens ML-checklista
(bokens kapitel 2): från rådata i en databas, via EDA, förbehandling, modelljämförelse och
utvärdering, till en Streamlit-app där en användare kan prediktera enskilda kunder och välja
threshold utifrån affärsbehov. Som komplement gjorde vi en kundsegmentering med K-Means.

## 2. Huvudresultat

### 2.1 Modelljämförelse

Datan delades stratifierat i 60 % träning (4 225 kunder), 20 % validering (1 409) och
20 % test (1 409). För varje modell användes `GridSearchCV` med 5-delad korsvalidering på
träningsmängden (scoring ROC-AUC). De tre bästa pipelinerna jämfördes på valideringsmängden:

| Modell | Bästa hyperparametrar | CV ROC-AUC | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistisk regression | C = 10 | 0,849 | 0,517 | 0,786 | 0,624 | 0,838 |
| Beslutsträd | max_depth 5, min_samples_split 50 | 0,819 | 0,506 | 0,786 | 0,616 | 0,817 |
| **Random forest** | 200 träd, max_depth 10, max_features sqrt, min_samples_split 20 | 0,847 | 0,524 | 0,775 | 0,626 | **0,838** |

Random forest valdes på grund av marginellt högst ROC-AUC på validering. Skillnaden mot
logistisk regression är dock försumbar (0,8381 mot 0,8376), och den logistiska modellen är
enklare att tolka. Beslutsträdet är tydligt sämre – det är den enklaste modellen och
överanpassar lätt utan begränsning på djupet.

### 2.2 Slutlig utvärdering på testmängden

Den valda modellen tränades om på träning + validering och utvärderades **en enda gång** på
testmängden, som inte använts tidigare:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0,759 | 0,532 | 0,783 | 0,634 | 0,841 |

Testresultatet ligger nära valideringsresultatet, vilket tyder på att modellen generaliserar
och att vi inte överanpassat den mot valideringsmängden. Confusion matrix vid threshold 0,5:
av 374 churnare i testmängden hittar modellen 293 (recall 78 %) och missar 81; av 1 035 som
stannar flaggas 258 felaktigt.

### 2.3 Threshold – affärsavvägningen

Modellen ger en sannolikhet, och gränsen för när vi kallar en kund "churnare" är ett
affärsbeslut. Vi lät användaren välja threshold i appen och visar effekten på testmängden:

| Threshold | Precision | Recall | Flaggade kunder | Missade churnare |
|---|---|---|---|---|
| 0,30 | 0,43 | 0,91 | 781 | 35 |
| 0,40 | 0,48 | 0,85 | 655 | 57 |
| 0,50 | 0,53 | 0,78 | 551 | 81 |
| 0,60 | 0,57 | 0,66 | 432 | 127 |
| 0,70 | 0,65 | 0,53 | 305 | 177 |

Lägre threshold fångar fler churnare men ger fler falska positiva. Om en åtgärd är billig
(ett mejl) kan 0,3 vara rimligt; om den är dyr (en personlig rabatt) bör threshold höjas.
Vi använde `class_weight="balanced"` i alla modeller eftersom en missad churnare i det här
problemet kostar mer än en onödig kontakt – det flyttar standardbeteendet mot högre recall.

### 2.4 Vad driver churn?

Feature importance från random forest: `Contract_Month-to-month` (0,145), `tenure` (0,100),
`TotalCharges` (0,073), `avg_monthly_charge` (0,058), `OnlineSecurity_No` (0,058),
`Contract_Two year` (0,058), `MonthlyCharges` (0,054), `TechSupport_No` (0,049),
`InternetService_Fiber optic` (0,048). Det stämmer med EDA:n: kunder med månadsavtal, kort
kundtid, fiber och utan support-/säkerhetstjänster churnar mest. Två av våra tre härledda
kolumner (`avg_monthly_charge`, `tenure_group_0-12`) hamnar bland de tio viktigaste.

### 2.5 Kundsegmentering

K-Means kördes på samma förbehandlade features (utan Churn). Silhouette score var högst för
k = 2 (0,25) och elbow-kurvan planade ut runt k = 3–4. Vi valde k = 4 för att få segment
som är användbara för verksamheten – ett omdömesbeslut, som boken beskriver. Segmenten
(sorterade på churn-andel) blev:

| Segment | Kunder | Snitt kundtid | Snitt kostnad/mån | Tilläggstjänster | Vanligaste avtal | Churn |
|---|---|---|---|---|---|---|
| 1 | 1 903 | 16 mån | 84 $ | 1,6 | Month-to-month | **56 %** |
| 2 | 1 620 | 21 mån | 50 $ | 1,8 | Month-to-month | 26 % |
| 3 | 1 994 | 59 mån | 91 $ | 4,2 | Two year | 14 % |
| 0 | 1 526 | 31 mån | 21 $ | 0,0 | Two year | 7 % |

Segment 1 – nya kunder med månadsavtal, hög kostnad och få tjänster – är där åtgärder ger
mest. Segmenteringen är icke-vägledd och svarar på "vilka kundtyper har vi?", medan
churn-modellen svarar på "vilka enskilda kunder riskerar att lämna?".

## 3. Teknisk specifikation

**Arkitektur.** `CSV → SQLite → Python/scikit-learn → joblib → Streamlit`. All logik ligger i
`src/`; Streamlit-sidorna visar bara resultat.

**Databas.** SQLite (`data/churn.db`) via Pythons `sqlite3`. Tre tabeller: `customers`
(rådatan), `model_runs` (en rad per utvärderad modell per träning, params och metrics som
JSON) och `predictions` (varje prediktion från appen med indata, sannolikhet, threshold och
beslut). Byggs med `python -m src.db`.

**Förbehandling** (`src/features.py`). `ColumnTransformer`: numeriska kolumner
medianimputeras och standardiseras, kategoriska imputeras med vanligaste värde och
one-hot-kodas med `handle_unknown="ignore"`. Ligger inne i `Pipeline` så att den anpassas
på nytt i varje korsvalideringsfold – ingen läcka från validering till träning.

**Feature engineering** (`src/data.py`). `num_addon_services` (antal tilläggstjänster),
`avg_monthly_charge` (TotalCharges/tenure, NaN vid tenure 0) och `tenure_group`.

**Modellering** (`src/models.py`, `src/train.py`). Logistisk regression, beslutsträd och
random forest med små grids. `GridSearchCV(cv=StratifiedKFold(5), scoring="roc_auc")`.
Modellval på validering, refit på träning + validering, en utvärdering på test.
Slutmodellen sparas med joblib och är incheckad så att appen startar utan att träna.

**Utvärdering** (`src/evaluate.py`). Accuracy, precision, recall, F1, ROC-AUC,
confusion matrix, ROC-kurva, `classification_report`, threshold-tabell och feature
importance/koefficienter med korrekta namn efter one-hot.

**Segmentering** (`src/segment.py`). K-Means (n_init=10) för k = 2–8 med inertia och
silhouette score, PCA till två komponenter (49 % förklarad varians) för visualisering,
segmentprofiler.

**Frontend.** Streamlit med fem sidor: Översikt, Data (EDA med Altair), Modeller
(jämförelse, test, threshold-slider, feature importance), Segmentering, Prediktera (formulär
och CSV-batch, loggning till databasen).

**Kvalitet.** 124 pytest-tester med 99 % täckning (krav 90 %), `filterwarnings = error`,
felfallstester för saknad fil/kolumn/modell/databas, ogiltig target, okänd kategori,
threshold utanför 0–1. Ruff för lint och formatering. GitHub Actions kör lint, tester på
Python 3.11 och 3.12 samt en end-to-end-träning på varje pull request.

**Versioner.** Python 3.11+, pandas 3.0, scikit-learn 1.9, Streamlit 1.63, matplotlib 3.10.

## 4. Utvärdering av gruppens arbete

[FYLL I – skriv detta själva, cirka en halv sida. Förslag på punkter:]

- **Vad har varit bra?** [T.ex. att vi hade ett fungerande flöde tidigt och kunde bygga ut
  det steg för steg; att testerna fångade fel innan de nådde appen.]
- **Vad har vi lärt oss?** [T.ex. varför testmängden måste hållas undan tills sist; att
  accuracy är missvisande vid obalans; att threshold är ett affärsbeslut och inte en
  modellparameter; att förbehandlingen måste ligga i pipelinen för att korsvalideringen ska
  bli rätt.]
- **Hur har Git och GitHub fungerat?** [T.ex. feature-branches, pull requests, CI som måste
  vara grön, konflikter vi stötte på och hur vi löste dem.]
- **Vad hade vi gjort annorlunda?** [T.ex. bestämt databas från start i stället för att lägga
  till den senare; testat på en ren dator tidigare.]
- **Framtida arbete.** [T.ex. logistisk regression som slutmodell för bättre tolkbarhet;
  kalibrera sannolikheterna; koppla segment till churn-modellen i appen; deploy på
  Streamlit Community Cloud.]
