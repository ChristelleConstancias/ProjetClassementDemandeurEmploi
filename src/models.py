"""Modele de maintenance predictive et son pipeline.

Pipeline sklearn : encodage de la classe machine + gradient boosting.
On expose des probabilites calibrees (important pour l'estimation de
performance sans etiquettes en aval).
"""

from __future__ import annotations

import joblib
import pandas as pd
from datetime import datetime, timezone
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from . import config as C


def sauver(
    pipe: Pipeline,
    nom: str,
    metadata: dict[str, object] | None = None,
) -> str:
    """Sauvegarde un pipeline et les informations utiles à son suivi/reproduction."""
    C.DIR_MODELS.mkdir(parents=True, exist_ok=True)
    chemin = C.DIR_MODELS / f"{nom}.joblib"

    estimateur = pipe.steps[-1][1]
    metadonnees = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": C.CIBLE,
        "pipeline_steps": [step_name for step_name, _ in pipe.steps],
        "estimator_class": (
            f"{type(estimateur).__module__}.{type(estimateur).__qualname__}"
        ),
        "estimator_params": estimateur.get_params(deep=False),
    }
    if metadata:
        metadonnees.update(metadata)

    joblib.dump(
        {
            "artifact_version": 1,
            "model": pipe,
            "metadata": metadonnees,
        },
        chemin,
    )
    return str(chemin)


def charger_artefact(nom: str) -> dict[str, object]:
    """Charge le pipeline et ses métadonnées, y compris les anciens fichiers."""
    contenu = joblib.load(C.DIR_MODELS / f"{nom}.joblib")
    if isinstance(contenu, dict) and "model" in contenu and "metadata" in contenu:
        return contenu
    return {
        "artifact_version": 0,
        "model": contenu,
        "metadata": {},
    }


def charger(nom: str) -> Pipeline:
    """Charge uniquement le pipeline pour conserver le contrat utilisé par l'API."""
    return charger_artefact(nom)["model"]


# Fonction pour évaluer plusieurs modèles sur un jeu de test
def evaluer_modeles(
    modeles: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Compare des pipelines déjà entraînés sur le même jeu de test."""
    resultats = []

    for nom, pipeline in modeles.items():
        y_pred = pipeline.predict(X_test)
        resultats.append(
            {
                "Modèle": nom,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Balanced Accuracy": round(
                    balanced_accuracy_score(y_test, y_pred),
                    4,
                ),
                "Precision macro": round(
                    precision_score(
                        y_test,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),
                    4,
                ),
                "Recall macro": round(
                    recall_score(
                        y_test,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),
                    4,
                ),
                "Recall classe 2": round(
                    recall_score(
                        y_test,
                        y_pred,
                        labels=[2],
                        average=None,
                        zero_division=0,
                    )[0],
                    4,
                ),
                "F1 macro": round(
                    f1_score(
                        y_test,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),
                    4,
                ),
            }
        )

    return pd.DataFrame(resultats).sort_values(
        by="Balanced Accuracy",
        ascending=False,
    )
