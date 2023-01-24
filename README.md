# Portfolio Applied Data Science

Bonno Nieuwenhout 19122381 groep 1

## 1. Inleiding

### 1.1 Inhoud
In dit portfolio is te lezen wat ik tijdens mijn minor applied data science heb geleerd en hoe ik mijn vaardigheden heb ontwikkeld. 
Elk punt is opgedeeld in het project voor FoodBoost en het project voor Cofano

### 1.2 FoodBoost
Voor het project van FoodBoost moest er gekeken worden naar het helpen van mensen in het ontwikkelen van een gezonder diet met behulp van machine learning. 
Het doel van het machine learning was dat een gebruiker minder moeite hoeft te doen om een gerecht te vinden dat gezond is en binnen zijn of haar smaak valt. 
Zo kan de gebruiker gezonder eten en toch genieten van het voedsel.

### 1.3 Cofano
Voor het project van Cofano was de probleemstelling het optimaliseren van terminalprocessen. 
Schepen kunnen lange tijd wachten om in- of uitgeladen te worden en dat kost geld. 
De strekking was dus het maken van een reinforcement learning model dat bepaalde processen hiervan kon versnellen.

## 2. Kennis en Literatuur

Ik zal in dit hoofdstuk vertellen over de kennis die ik heb opgedaan en wat ik geleerd heb over data science.

Om te beginnen heb ik kennis opgedaan door de datacamp opdrachten te maken.
Daarnaast heb ik de lectures bijgewoond en deze stof tot mij genomen.
Vervolgens heb ik met deze kennis gewerkt aan de projecten waar ik gaande weg nog meer heb geleerd over data science.
Tijdens het programmeren leer je namelijk over steeds meer data science onderwerpen doordat je opzoek gaat naar oplossingen voor je probleem.

### 2.1 FoodBoost
Met het project van FoodBoost hebben we een supervised machine learning model ontwikkeld.
Dit model zou op basis van aangegeven gerechten voor 1 categorie kunnen aangeven welke mogelijke recepten je nog meer lekker zou vinden.

Bij het supervised machine learning model hebben we gekozen om een classifier toe te passen.
Deze classifier zou de gerechten beoordelen op wel of niet lekker.
Hierbij heb ik geleerd over wat een aantal classifiers inhouden. Om ons model te testen hebben we ook gekeken naar hoe verschillende classifiers presteren.
Er is namelijk niet één classifier die je altijd kan gebruiken. We hebben moeten onderzoeken welke classifier het meest geschikt was om ons probleem op te lossen.
Hierdoor heb ik kennis gedaan over KNN, Logistic Regression, Multinomial Naive Bias, Decision Tree en Random Forest classifiers.

Verder hebben we alle data gesimuleerd. Ik heb geleerd hoe dit gedaan moest worden en vervolgens toegepast

### 2.2 Cofano
Met het project van Cofano hebben we ons bezig gehouden met Reinforcement Learning.
We zijn 4 weken later begonnen aan dit project, dus we hebben daarvoor al via de andere groepen geleerd over het project.

Meden door de gesprekken met Jeroen Vuurens hebben we besloten om ook RL te gebruiken voor ons probleemdomein.
Hiervoor hebben we ons moeten verdiepen in wat reinforcement learning inhoud, hoe je het kan toepassen en hoe je het kan opbouwen.
We hebben gekeken naar een aantal voorbeelden. Een paar om te begrijpen hoe het is gemaakt en een ander aantal dat al meer overkapping zou hebben met ons probleemdomein.

Om ons te helpen hebben gebruik gemaakt van een bekend framework voor RL. Dit framework heeft ons veel tijd en moeite bespaard, omdat we veel dingen hierdoor zelf niet hoefde te bouwen.
Om een RL toe te passen kan je gebruik maken van gym[link]. Met gym kan je RL environments bouwen.
Deze environments kunnen gebruikt worden voor SB3. De RL modellen die SB3 aanbied werken namelijk op basis van deze environments.
Door een RL model van SB3 onze environment mee te geven en het model vervolgens te trainen, krijgen we een RL model dat ons probleemdomein kan oplossen.

Bij het gebruik maken van SB3 heb ik veel geleerd over het PPO en A2C RL model. Ik heb hierbij naar de documentatie gekeken van de modellen en geleerd over de principes over hoe de modellen te werk gaan.
De link naar de documentatie is hier [link]

Om kennis op te doen hoe het maken van een environment werkt hebben we gekeken naar voorbeelden online en video's op youtube.
Ook heb ik gekeken naar de voorbeelden die op SB3 staan. Deze voorbeelden zijn kort van code en goed gedocumenteerd waardoor ze goed te lezen en begrijpen zijn.
In deze link staan een aantal voorbeelden [link]

## 3. Jupyter notebooks

In dit hoofdstuk laat ik zien wat ik heb geprogrammeerd voor beide projecten.
Hoe ik data heb gesimuleerd en hoe ik de modellen heb ontwikkeld. 
Er zijn veel iteraties over de code heengegaan en heb veel dingen continue lopen aanpassen. Hierdoor zijn er veel stukken code die dezelfde functie hebben.
Ik heb daarom alleen de laatste versies benoemd van wat ik heb ontwikkeld, aangezien het is geïtereerd op voorgaande code.

### 3.1 FoodBoost

#### 3.1.1 Simulatie

We hebben de data die we nodig hadden allemaal gesimuleerd. Hiervoor heb ik classes aangemaakt om deze data in op te slaan.
Om dit voor het testen overzichtelijk te houden heb ik een aantal dingen opgesplitst.
Het maken van de users en het maken van de benodigde matrix heb ik in afzonderlijke notebooks gedaan.

- [link] user class
- [link] algemene utils
- [link] Create users
- [link] Create matrix

#### 3.1.2 Classifiers

Ik heb een utils class en een dto class geschreven zodat ik het testen van de classifiers overzichterlijker kon doen.

- [link] classifier utils
- [link] dto

#### 3.1.3 Testen

Om uiteindelijk de gesimuleerde data te gebruiken en de classifiers te testen op de matrixen heb ik een aparte notebook gemaakt.
Hierin wordt de matrix aan de classifiers gegeven en vervolgens wordt er per classifier een score bepaald.
De data voor elke classifier wordt in een tabel gezet en deze vervolgens weergegeven.
In deze tabel staan de scores van elk model zodat kan worden vergeleken welk model het beste is.

Ik data op een aantal manieren getest. Deze zijn hieronder op een rij gezet

- [link] 1 tegen 1
- [link] 1 tegen n
- [link] n tegen n

### 3.2 Cofano



4. ## Presentaties

5. ## Paper

6. ## Datacamp

