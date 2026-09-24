# Churn-prediktion på Telco Customer Churn – teknisk rapport

**Kurs:** AI – teori och tillämpning, del 1 · **Uppgift:** Projektarbete Del 2
**Grupp:** Jonathan, Lukas och Havash
**Repo:** https://github.com/LukWen-Ill/Dataview_v2

---

## 1. Bakgrund

Ett telekombolag förlorar pengar varje gång en kund säger upp sitt abonnemang ("churn").
Det är dyrare att vinna en ny kund än att behålla en befintlig, så bolaget vill veta _vilka_
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

| Modell               | Bästa hyperparametrar                                           | CV ROC-AUC | Precision | Recall | F1    | ROC-AUC   |
| -------------------- | --------------------------------------------------------------- | ---------- | --------- | ------ | ----- | --------- |
| Logistisk regression | C = 10                                                          | 0,849      | 0,517     | 0,786  | 0,624 | 0,838     |
| Beslutsträd          | max_depth 5, min_samples_split 50                               | 0,819      | 0,506     | 0,786  | 0,616 | 0,817     |
| **Random forest**    | 200 träd, max_depth 10, max_features sqrt, min_samples_split 20 | 0,847      | 0,524     | 0,775  | 0,626 | **0,838** |

Random forest valdes på grund av marginellt högst ROC-AUC på validering. Skillnaden mot
logistisk regression är dock försumbar (0,8381 mot 0,8376), och den logistiska modellen är
enklare att tolka. Beslutsträdet är tydligt sämre – det är den enklaste modellen och
överanpassar lätt utan begränsning på djupet.

### 2.2 Slutlig utvärdering på testmängden

Den valda modellen tränades om på träning + validering och utvärderades **en enda gång** på
testmängden, som inte använts tidigare:

| Accuracy | Precision | Recall | F1    | ROC-AUC |
| -------- | --------- | ------ | ----- | ------- |
| 0,759    | 0,532     | 0,783  | 0,634 | 0,841   |

Testresultatet ligger nära valideringsresultatet, vilket tyder på att modellen generaliserar
och att vi inte överanpassat den mot valideringsmängden. Confusion matrix vid threshold 0,5:
av 374 churnare i testmängden hittar modellen 293 (recall 78 %) och missar 81; av 1 035 som
stannar flaggas 258 felaktigt.

### 2.3 Threshold – affärsavvägningen

Modellen ger en sannolikhet, och gränsen för när vi kallar en kund "churnare" är ett
affärsbeslut. Vi lät användaren välja threshold i appen och visar effekten på testmängden:

| Threshold | Precision | Recall | Flaggade kunder | Missade churnare |
| --------- | --------- | ------ | --------------- | ---------------- |
| 0,30      | 0,43      | 0,91   | 781             | 35               |
| 0,40      | 0,48      | 0,85   | 655             | 57               |
| 0,50      | 0,53      | 0,78   | 551             | 81               |
| 0,60      | 0,57      | 0,66   | 432             | 127              |
| 0,70      | 0,65      | 0,53   | 305             | 177              |

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

| Segment | Kunder | Snitt kundtid | Snitt kostnad/mån | Tilläggstjänster | Vanligaste avtal | Churn    |
| ------- | ------ | ------------- | ----------------- | ---------------- | ---------------- | -------- |
| 1       | 1 903  | 16 mån        | 84 $              | 1,6              | Month-to-month   | **56 %** |
| 2       | 1 620  | 21 mån        | 50 $              | 1,8              | Month-to-month   | 26 %     |
| 3       | 1 994  | 59 mån        | 91 $              | 4,2              | Two year         | 14 %     |
| 0       | 1 526  | 31 mån        | 21 $              | 0,0              | Two year         | 7 %      |

Segment 1 – nya kunder med månadsavtal, hög kostnad och få tjänster – är där åtgärder ger
mest. Segmenteringen är icke-vägledd och svarar på "vilka kundtyper har vi?", medan
churn-modellen svarar på "vilka enskilda kunder riskerar att lämna?".

### 2.6 Dashboard (extra, utanför kursens krav)

Startsidan Dashboard blickar framåt: varje aktiv kund (Churn = No) rullas fram upp till 12 månader
(valbar horisont) med sannolikheten (1 − p)^m att vara kvar, där p är modellens churn-sannolikhet, och summerar
kvarvarande månadsintäkt och förväntad förlust. Blå linje är dagens aktiva kunders
kvarvarande intäkt plus en mockad nykundsförsäljning; röd linje är dagens kunders
kvarvarande intäkt utan nykunder, så gapet mellan linjerna är nykundernas bidrag. Nykundsmocken
är (ca 600 nya kunder månad 1, +5 % per månad med ±20 % seedad
slump, 50 $ per kund, samma churn-risk som de befintliga). Hela prognosen antar konstant
churn-risk per kund och månad samt frysta priser.
Den bygger på den befintliga modellen utan omträning; beräkningarna ligger i
`src/dashboard.py`.

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

**Frontend.** Streamlit med sex sidor: Dashboard (extra: intäktsprognos, startsida), Översikt, Data (EDA
med Altair), Modeller
(jämförelse, test, threshold-slider, feature importance), Segmentering, Prediktera (formulär
och CSV-batch, loggning till databasen).

**Kvalitet.** 162 pytest-tester med 99 % täckning (krav 90 %), `filterwarnings = error`,
felfallstester för saknad fil/kolumn/modell/databas, ogiltig target, okänd kategori,
threshold utanför 0–1. Ruff för lint och formatering. GitHub Actions kör lint, tester på
Python 3.11 och 3.12 samt en end-to-end-träning på varje pull request.

**Versioner.** Python 3.11+, pandas 3.0, scikit-learn 1.9, Streamlit 1.63, matplotlib 3.10

## 4. Utvärdering av gruppens arbete

### 4.1 Vad har varit bra

Den största styrkan i gruppen har varit kunskapsnivån. Alla tre hade läst kapitel 1-6 och gjort
övningsuppgifterna innan projektet startade, och det märktes direkt i diskussionerna. Vi behövde
inte lägga tid på att förklara grundbegrepp för varandra utan kunde gå rakt på frågor som spelar
roll: vilken utvärderingsmetrik som är rimlig vid obalanserade klasser, varför förbehandlingen
måste ligga i pipelinen, och vad som skiljer en signal i datan från ett läckage. När vi väl hade
datasettet på plats tog det mindre än en dag att enas om tre kandidatmodeller, logistisk
regression, beslutsträd och random forest, och motivera varje val utifrån problemet snarare än
utifrån vad som råkade finnas i scikit-learn.

En annan styrka var att vi delade upp ansvaret efter lager i stället för efter person. Data och
databas, modellering och utvärdering samt frontend hölls isär, och gränssnitten emellan var
tydliga. All ML-logik ligger i src/ och appen sköter bara presentationen. Det gjorde att vi kunde arbeta parallellt utan att
trampa på varandra i koden, och att en person kunde sätta sig in i en annans del utan att behöva
förstå allt. Hur mycket var och en hann bidra varierade över projektet, men uppdelningen gjorde
att ingen del blev beroende av att en viss person var tillgänglig.

EDA:n gav också mer än vi väntat oss. Den bekräftade inte bara det uppenbara, att månadskunder
churnar mer, utan gav ett fynd vi inte hade förutsett: churn följer inte priset utan
tjänstepaketet. Kunder med få tjänster som binder dem churnar mest, oavsett om paketet är billigt
eller dyrt. Att en systematisk genomgång av datan kan ändra hur man tänker om problemet var en
konkret lärdom.

### 4.2 Vad har vi lärt oss

Den viktigaste lärdomen kom innan vi skrev en enda rad modellkod: valet av dataset. Vi bytte
dataset flera gånger. Vi började med studentdata, som hade en tydlig målvariabel men gav för lite
att analysera. Sedan ett Spotify-dataset, som var roligare och lärde oss att först formulera vad
som ska predikteras, men som visade sig vara syntetiskt och därmed svårt att dra slutsatser från.
Därefter ett dataset om skärmtid och psykisk hälsa, där vi upptäckte att en av kolumnerna i
praktiken var målvariabeln i förklädnad, alltså ett läckage, och att området dessutom är känsligt
att bygga ett "diagnosverktyg" kring.

Först då landade vi i Telco Customer Churn. I efterhand ser vi att vi inte var obeslutsamma utan
att vi successivt lärde oss vad som gör ett dataset lämpligt, en målvariabel som faktiskt finns i
datan, tillräckligt många och olika typer av förklarande variabler, en verksamhetsfråga som gör
modellen meningsfull, och frånvaro av läckage. Det vände på vår arbetsordning. Man väljer inte
data först och letar sedan efter något att göra med den, man formulerar en fråga och väljer data
som kan besvara den.

Tekniskt lärde vi oss framför allt tre saker. Att train/validation/test måste hållas strikt isär
och att testmängden rörs exakt en gång, något vi byggde in i träningsskriptet snarare än
förlitade oss på disciplin. Att accuracy är ett vilseledande mått när en klass dominerar, och att
threshold är ett beslut skilt från modellen. Och att feature engineering inte automatiskt tillför
något - en av våra tre härledda kolumner visade sig ha korrelation 1,00 med en kolumn vi redan
hade.

### 4.3 Hur har arbetet med Git och GitHub fungerat

Vi bestämde tidigt ett enkelt flöde: feature-grenar, pull requests mot main, och CI med lint,
formatkontroll och tester som måste vara gröna innan merge. Det fungerade i stort sett bra. CI
fångade fel vi själva missat, till exempel en formateringsavvikelse i en notebook som passerade
den lokala lint-kontrollen men inte formatkontrollen.

Det gick inte helt utan friktion. Vid ett tillfälle arbetade två personer ovetande på samma
feature-gren, och den ena versionen visade sig dessutom utgå från en äldre main än den andra.
Grenarna delade namn men nästan ingen kod. Vi löste det genom att döpa om den lokala grenen och
pusha den separat i stället för att försöka rebasa, och tog sedan beslutet i gruppen om vilken
version som skulle leva vidare. Lärdomen är att en gren behöver en ägare, och att man
kontrollerar vad som finns på GitHub innan man börjar arbeta på ett grennamn som redan
existerar.

Notebooks och Git är också en dålig kombination. Outputs och metadata ger stora, oläsliga
diffar, och konflikter i en notebook är nästan omöjliga att lösa manuellt. Vi hanterade det
genom att bara en person arbetade i respektive notebook och genom att all logik låg i vanliga
Python-moduler som notebooken importerade.

Totalt blev det 13 pull requests, varav 8 mergades och 5 stängdes till förmån för andra
lösningar. Granskningen skedde i gruppen och muntligt, inte som formella reviews på GitHub; i
praktiken var det CI som var grinden före merge. Det fungerade i ett projekt av den här
storleken, men i ett större hade vi velat ha en läsande människa på varje pull request.

### 4.4 Vad hade vi gjort annorlunda

Vi hade valt dataset snabbare. Tiden vi lade på tre dataset som förkastades hade räckt till att
träna om modellen utan den redundanta kolumnen och till att skriva rapporten parallellt med koden
i stället för efter.

Vi hade också satt gränser för omfånget tidigare. Projektet fick delar som uppgiften inte kräver,
som kundsegmentering, loggning av prediktioner i databasen och ett täckningskrav på 90 procent i
testerna. Delarna är bra i sig, men de kostade tid som hade gett mer i rapporten och i att alla
tre kan förklara varje del av koden.

Vi använde AI-verktyg som stöd under hela projektet, för kodgranskning, för
att diskutera analysen och för att skriva kod i moduler vi själva specificerat. Det fungerade
bäst när vi satte tydliga regler för vad verktyget fick göra och alltid förstod resultatet innan
det gick in i repot. Det fungerade sämst när koden blev mer avancerad än vår egen förståelse hann
bli, och vi fick backa och läsa in oss i efterhand. Nästa gång skulle vi använda verktygen mer
för att förklara och mindre för att generera.

Slutligen hade vi tagit beslutet om de härledda kolumnerna innan slutmodellen tränades. Nu står
det i EDA:n att en av dem är redundant, samtidigt som den sitter i den modell som körs i appen.
Det är försvarbart, trädmodeller skadas inte av det, men det är inte det beslut vi hade tagit med
facit i hand.
