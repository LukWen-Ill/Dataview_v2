"""Modellkatalogen: de klassificeringsmodeller vi jämför och deras hyperparametergrids.

Varje modell är en sklearn-Pipeline med samma förbehandling (src/features.py) följt av
klassificeraren. Att förbehandlingen ligger i pipelinen gör att skalning och one-hot
anpassas på nytt i varje korsvalideringsfold - annars läcker information från
valideringsfolden in i träningen.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.features import build_preprocessor

RANDOM_STATE = 42

# Kort motivering till class_weight="balanced" på alla tre modeller:
# Churn är minoritetsklassen (26,5 %). Utan viktning lär sig modellen att "No" nästan
# alltid är rätt och missar många churnare. Med "balanced" kostar en missad churnare
# mer under träningen, vilket ger högre recall. Det är vad vi vill i det här problemet:
# vi vill hellre kontakta en kund i onödan än missa en som faktiskt lämnar.


def build_pipeline(classifier) -> Pipeline:
    """Förbehandling + klassificerare i en enda pipeline."""
    return Pipeline([("preprocess", build_preprocessor()), ("classifier", classifier)])


def build_candidates(random_state: int = RANDOM_STATE) -> dict[str, tuple[Pipeline, dict]]:
    """Modellnamn -> (pipeline, hyperparametergrid).

    Nycklarna i gridet har prefixet "classifier__" eftersom klassificeraren är steget
    som heter "classifier" i pipelinen. Gridarna är medvetet små så att träningen
    går på under en minut.
    """
    logreg = LogisticRegression(
        max_iter=2000,  # standardvärdet 100 räcker inte alltid -> ConvergenceWarning
        class_weight="balanced",
        random_state=random_state,
    )
    tree = DecisionTreeClassifier(class_weight="balanced", random_state=random_state)
    forest = RandomForestClassifier(class_weight="balanced", random_state=random_state)

    return {
        "Logistisk regression": (
            build_pipeline(logreg),
            # C är regulariseringsstyrkan: lägre C = starkare regularisering.
            {"classifier__C": [0.01, 0.1, 1, 10]},
        ),
        "Beslutsträd": (
            build_pipeline(tree),
            # Utan begränsning växer trädet tills det memorerat träningsdatan.
            {
                "classifier__max_depth": [3, 5, 8],
                "classifier__min_samples_split": [2, 20, 50],
            },
        ),
        "Random forest": (
            build_pipeline(forest),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [5, 10],
                "classifier__max_features": ["sqrt", 0.5],
                "classifier__min_samples_split": [2, 20],
            },
        ),
    }
