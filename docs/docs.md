


Huvudresultat

Datan delades in i train, test, val. Vi jämförde 3 modeller och tränade dem med hyperparametrar, gridsearch och kors validering för att få fram den bästa modellen. 
Vi använde ROC-AUC som mått och RandomForest var våran vinnare, skillnaden mot logistik regression med 0,0006 så väldigt lite, och kanske LR hade varit bättre för den är mycket snabbare att träna, tolka och underhålla.
Den valda modellen tränas om train, val och utvärderas på test datan och resultaten visar att den inte är överpassad mot train / val.
Accuracy	Precision	  Recall	  F1	      ROC-AUC
0,759	    0,532	      0,783	    0,634	    0,841

Modellen hittar 78% av kunder som faktiskt lämnar, class_weight="balanced" prioriterar recall för att missa en churnare kostar mer än att kontakta någon en extra gång.

Accuracy för modellen ligger på 0,759 vilket är strax över våran baseline som är 73.5% så en modell som bara gissar "stannar" hade haft 73.5% rätt.
Det visar att accuracy är fel mått för vårat dataset. 

ROC - AUC
Modellen ger en sannolikhet per kund,  och en tröskel avgör var gränser går mellan "stannar" och "churnar". Med tröskel 0.5 räknas allt över 0.5 som churn. Accuracy, precision och recall beror alla på vilken tröskel man väljer.

ROC-kurvan visar istället alla trösklar samtidigt. X-axeln anger hur stor andel av kunderna som stannar som modellen larmar om i onödan, y-axeln hur stor andel av dem som faktiskt churnar som dne hittar. För vår modell ger 0,2 på x-axeln och ungefär 0,7 på y-axeln, om vi accepterar att larma om 20% av dem som stannar så hittar vi 70% av dem som churnar.

AUC är ytan under kurvan. Vår modell får 0.841. En modell som gissar slumpmässigt följer diagonalen och får 0,5 och en perfekt modell går rakt upp i vänstra hörnet och sedan rakt åt höger, vilket hade gett 1,0 i AUC.
ROC - AUC mäter alltså hur väl modellen rangordnar kunder efter risk, oberoende av tröskel. 
Det var därför vi använder det som mått vid modellvalet.

Vi har lagt till så att man kan fylla i för att prediktera churn, där finns en tröskel som kan tas höjas eller sänkas i våran app. När den är 0.3 så stiger recall till 91% och 781 kunder flaggas. Vid 0.7 i tröskel så flaggas 305 kunder men 177 churnare missas, så alltså kan våran app visa en prediktion hur många som lämnar beroende på vilken tröskel som väljs.

Vi har implementerat olika grafer och det är så att man kan se månadsavtal, total kostnad av tjänster, och EDA som visar att churn är kraftigt koncentrerad till första året och till kunder utan bindningstid.

En annan av våra features är kundsegmenterings biten, K-Means med fyra kluster visar bland annat ett segment med månadsavtal, snitt på månadskostnaden, snitt i tillägstjänster, i segment 1 kan man se att det är 56% som churnar och deras vanligaste avtal är månad till månad till skillnad från segmentet med 2 år sonm har 7%.
Det går att ändra så att segmentet och klustringen endast är på tid som kunden har varit kund, månadskostnad och antal tillägstjänster dem har och det är lättare att beskriva.

Vi tränade en linjär regression modell på enbart kolumnerna som har med tjänster som kunden väljer och för kunder med internet, det gav 99.7% R2 score, vilket säger att priserna är en summa av valda tjänster, det är inget som skiftas beroende på kund. Det visar att det höga R2 värdet inte gör att modellen är bra, det som visas är att det inte finns någon mening med att modellera.


Kort teknisk specifikation.

Språk och miljö. Python 3.11, alla beroenden är versionslåsta i requirements.txt för att undvika skillnader mellan lokal miljö och CI, det är viktigt eftersom modellen sparas med joblib och måste laddas med samma scikit-learn version.

Databas så kör vi SQLite via Pythonm vi har 3 tabeller som är customer med rådatan, model_runs som loggar varje utvärderad modell och predictions som loggar varje prediktion från appen.

För modellering så har vi pandas för datahantering och scikit-learn för resten. Allting går igenom en pipeline med columntransformer som gör one hot encoding på våra kolumner med strängar och gör standardisering av värdena så att dem inte är för stora i skillnader.
Eftersom våran förbehandling ligger i pipelinen förhindrar det data leakage för att den anpassas på nytt för varje korsvalidering.

Frontend är byggd i streamlit, sex sidor, med grafer och med rutor som gör att du kan prediktera själv. Appen laddar den sparade modellen och tränar aldrig själv. 
All ML logiken ligger i src/. Sidorna i appen visas bara resultat.

Tester så har vi över 125 tester med alla som går igenom. Vi har RUFF för lint och formatering. GitHub actions kör lint tester och en fullständig träning på varje pull request och det kan inte mergas förrän allting är grönt.

