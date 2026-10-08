"""API interne de prediction du retour a l'emploi."""

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import uuid

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

from src import config as C, data as D, models as M


class DemandeurEmploi(BaseModel):
    # Définir ici les champs correspondant aux caractéristiques du demandeur d'emploi
    # est ce qu'on a besoin de usage_id??
    # niveau diplome : liste deroulante , ou saisie libre
    usager_id: str
    age: float = Field(ge=16, le=100)
    niveau_diplome: str
    anciennete_poste_ans: float = Field(ge=0)
    code_rome_vise: str
    code_insee_commune: str
    est_allocataire: float
    nationalite_hors_ue: int
    synthese_entretien: str


class Feedback(BaseModel):
    session_id: str
    age: float
    niveau_diplome: str
    anciennete_poste_ans: float
    code_rome_vise: str
    code_insee_commune: str
    est_allocataire: float
    nationalite_hors_ue: int
    synthese_entretien: str
    prediction_modele: int
    classe_reelle: int


# ==========================
# FastAPI
# ==========================

app = FastAPI(title="Orientation Demandeur Emploi API", version="1.0")

MODEL_ARTIFACT_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(C.DIR_MODELS / "pipeline_meilleur_recall_f1.joblib"),
    )
)
LEGACY_MODEL_NAME = "pipeline"


def charger_modele_actif() -> tuple[dict[str, object], str]:
    """Charge le pipeline optimisé, avec l'ancien pipeline comme fallback."""
    if MODEL_ARTIFACT_PATH.exists():
        artifact = joblib.load(MODEL_ARTIFACT_PATH)
        if isinstance(artifact, dict) and "pipeline" in artifact:
            return artifact, "pipeline_meilleur_recall_f1"
        raise ValueError(f"Artefact invalide : {MODEL_ARTIFACT_PATH}")

    legacy_artifact = M.charger_artefact(LEGACY_MODEL_NAME)
    return {
        "pipeline": legacy_artifact["model"],
        "retraining_config": legacy_artifact.get("metadata", {}),
    }, LEGACY_MODEL_NAME


active_artifact, active_model_name = charger_modele_actif()
pipeline = active_artifact["pipeline"]


def journaliser_prediction(
    session_id: str, usager: dict, prediction: int, probabilite: float
):
    log = {
        "session_id": session_id,
        "date_inference": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "age": usager["age"],
        "niveau_diplome": usager["niveau_diplome"],
        "anciennete_poste_ans": usager["anciennete_poste_ans"],
        "code_rome_vise": usager["code_rome_vise"],
        "code_insee_commune": usager["code_insee_commune"],
        "est_allocataire": usager["est_allocataire"],
        "nationalite_hors_ue": usager["nationalite_hors_ue"],
        "synthese_entretien": usager["synthese_entretien"],
        "prediction": prediction,
    }

    df = pd.DataFrame([log])
    C.DIR_LOGS.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        C.DIR_LOGS / "predictions.csv",
        mode="a",
        header=not (C.DIR_LOGS / "predictions.csv").exists(),
        index=False,
    )


# ==========================
# Health Check
# ==========================


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "model": active_model_name,
    }


# ==========================
# Prediction
# ==========================


@app.post("/predict")
def predict(demandeur_emploi: DemandeurEmploi):

    print("Debut prediction pour demandeur_emploi:", demandeur_emploi)

    session_id = str(uuid.uuid4())
    print("Session ID:", session_id)

    try:
        # Préparer les données pour la prédiction
        usager_test = demandeur_emploi.model_dump()
        print("Data prepare pour  prediction:", usager_test)

        # Préparer les données d'entrée pour le modèle
        retraining_config = active_artifact.get("retraining_config", {})
        features = retraining_config.get("features")
        X_test = D.prepare_data(usager_test, features=features)

        # Faire la prédiction avec le pipeline actif
        prediction = pipeline.predict(X_test)[0]

        probabilities = pipeline.predict_proba(X_test)[0]
        class_index = list(pipeline.classes_).index(prediction)
        probability = float(probabilities[class_index])
        # Journaliser la prédiction et la probabilité associée
        journaliser_prediction(session_id, usager_test, int(prediction), probability)

        print("Fin prediction pour demandeur_emploi:", demandeur_emploi)

        return {"prediction": int(prediction), "probability": round(probability, 4)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Une route /feedback permet au conseiller d'enregistrer la classe réellement
# observée après le suivi du demandeur d'emploi. Ces retours utilisateurs sont stockés dans un fichier dédié.
@app.post("/feedback")
def feedback(usager: Feedback):

    df = pd.DataFrame([usager.model_dump()])

    # Création du répertoire s'il n'existe pas
    Path(C.DIR_FEEDBACK).mkdir(parents=True, exist_ok=True)

    fichier_feedback = Path(C.DIR_FEEDBACK) / "feedback.csv"

    # Enregistrer le feedback dans le fichier CSV
    df.to_csv(
        fichier_feedback, mode="a", header=not fichier_feedback.exists(), index=False
    )

    return {"message": "Feedback enregistré"}


def entrainer_artefact(
    artifact: dict[str, object],
) -> tuple[dict[str, object], dict[str, float], int, int]:
    """Réentraîne une copie du meilleur pipeline et renvoie l'artefact prêt à sauver."""
    config = artifact.get("retraining_config")
    if not isinstance(config, dict) or not config.get("best_params"):
        raise ValueError("La configuration de réentraînement est absente de l'artefact")

    target = config.get("target", C.CIBLE)
    features = config.get("features", C.FEATURES_SCENARIO_2C)
    historique = D.load_data(verbose=False)
    jeux = [historique]
    feedback_count = 0

    if C.FEEDBACK_FILE.exists():
        feedback = D.load_feedback(verbose=False)
        if "classe_reelle" in feedback.columns:
            feedback = feedback.rename(columns={"classe_reelle": target})
        if target not in feedback.columns:
            raise ValueError(f"Colonne cible {target!r} absente des feedbacks")
        feedback = feedback.loc[feedback[target].notna()].copy()
        feedback_count = len(feedback)
        if feedback_count:
            jeux.append(feedback)

    donnees = D.prepare_training_data(pd.concat(jeux, ignore_index=True, sort=False))
    missing_features = [
        feature for feature in features if feature not in donnees.columns
    ]
    if missing_features:
        raise ValueError(f"Variables requises absentes : {missing_features}")

    X = donnees[features].copy()
    y = pd.to_numeric(donnees[target], errors="raise").astype(int)
    seed = int(config.get("random_state", C.SEED))
    test_size = float(config.get("test_size", 0.2))
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    pipeline_reentraine = clone(artifact["pipeline"])
    pipeline_reentraine.set_params(**config["best_params"])
    pipeline_reentraine.fit(X_train, y_train)

    predictions = pipeline_reentraine.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "recall_classe_2": float(
            recall_score(
                y_test,
                predictions,
                labels=[2],
                average=None,
                zero_division=0,
            )[0]
        ),
    }

    pipeline_final = clone(pipeline_reentraine).fit(X, y)
    updated_config = {
        **config,
        "last_retrained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(X),
        "feedback_rows": feedback_count,
        "last_evaluation": metrics,
    }
    return (
        {**artifact, "pipeline": pipeline_final, "retraining_config": updated_config},
        metrics,
        len(X),
        feedback_count,
    )


@app.post("/retrain")
def retrain():
    """Réentraîne le pipeline sélectionné à la demande avec l'historique et les feedbacks."""
    global active_artifact, active_model_name, pipeline

    if not MODEL_ARTIFACT_PATH.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                "L'artefact pipeline_meilleur_recall_f1.joblib est introuvable. "
                "Relancez d'abord la cellule de sauvegarde du notebook."
            ),
        )

    try:
        artifact = joblib.load(MODEL_ARTIFACT_PATH)
        updated_artifact, metrics, training_rows, feedback_rows = entrainer_artefact(
            artifact
        )

        retraining_config = updated_artifact["retraining_config"]
        mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(C.MLFLOW_EXPERIMENT_RETRAINING)
        with mlflow.start_run(run_name="api_retraining"):
            mlflow.log_params(
                {
                    "model": str(retraining_config.get("model_name", "unknown")),
                    "scenario": str(retraining_config.get("scenario_name", "unknown")),
                    "training_rows": training_rows,
                    "feedback_rows": feedback_rows,
                    **{
                        f"best_{key}": str(value)
                        for key, value in retraining_config["best_params"].items()
                    },
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=updated_artifact["pipeline"],
                name="pipeline",
            )

        C.DIR_ARCHIVE.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(
            MODEL_ARTIFACT_PATH,
            C.DIR_ARCHIVE / f"pipeline_meilleur_recall_f1_{timestamp}.joblib",
        )

        temporary_path = MODEL_ARTIFACT_PATH.with_name(
            f"{MODEL_ARTIFACT_PATH.stem}_{uuid.uuid4().hex}.tmp"
        )
        joblib.dump(updated_artifact, temporary_path)
        temporary_path.replace(MODEL_ARTIFACT_PATH)

        active_artifact = updated_artifact
        active_model_name = "pipeline_meilleur_recall_f1"
        pipeline = updated_artifact["pipeline"]

        return {
            "message": "Pipeline réentraîné et activé",
            "model": active_model_name,
            "training_rows": training_rows,
            "feedback_rows": feedback_rows,
            "accuracy": round(metrics["accuracy"], 4),
            "f1_macro": round(metrics["f1_macro"], 4),
            "recall_classe_2": round(metrics["recall_classe_2"], 4),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Échec du réentraînement : {exc}"
        ) from exc
