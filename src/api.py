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
import shap
from fastapi import FastAPI, HTTPException
from lightgbm import LGBMClassifier
from pydantic import BaseModel, Field
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src import config as C, data as D, models as M
from src.privacy import append_csv_record, pseudonymiser_usager_id


class DemandeurEmploi(BaseModel):
    # Définir ici les champs correspondant aux caractéristiques du demandeur d'emploi
    # niveau diplome : liste deroulante , ou saisie libre
    usager_id: str
    conseiller_id: str = Field(min_length=1)
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
        str(C.DIR_MODELS / "pipeline_prediction_retour_emploi.joblib"),
    )
)
LEGACY_MODEL_NAME = "pipeline"
SHAP_MODELS = (RandomForestClassifier, LGBMClassifier, XGBClassifier)


def noms_variables_transformees(preprocessor) -> list[str]:
    noms = []
    for nom, transformateur, colonnes in preprocessor.transformers_:
        if nom == "texte":
            vectorizer = transformateur.named_steps["tfidf"]
            noms.extend(
                f"synthese_entretien: {terme}"
                for terme in vectorizer.get_feature_names_out()
            )
        elif nom == "categorielles":
            encoder = transformateur.named_steps["onehot"]
            noms.extend(encoder.get_feature_names_out(colonnes).tolist())
        elif transformateur != "drop":
            noms.extend(colonnes)
    return noms


def charger_modele_actif() -> tuple[dict[str, object], str]:
    """Charge le pipeline optimisé, avec l'ancien pipeline comme fallback."""
    if MODEL_ARTIFACT_PATH.exists():
        artifact = joblib.load(MODEL_ARTIFACT_PATH)
        if isinstance(artifact, dict) and "pipeline" in artifact:
            return artifact, "pipeline_prediction_retour_emploi"
        raise ValueError(f"Artefact invalide : {MODEL_ARTIFACT_PATH}")

    legacy_artifact = M.charger_artefact(LEGACY_MODEL_NAME)
    return {
        "pipeline": legacy_artifact["model"],
        "retraining_config": legacy_artifact.get("metadata", {}),
    }, LEGACY_MODEL_NAME


active_artifact, active_model_name = charger_modele_actif()
pipeline = active_artifact["pipeline"]


def journaliser_prediction(
    session_id: str,
    conseiller_id: str,
    usager: dict,
    usager_id_pseudonymise: str,
    prediction: int,
    probabilite: float,
):
    log = {
        "session_id": session_id,
        "conseiller_id": conseiller_id,
        "usager_id_pseudonymise": usager_id_pseudonymise,
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
        "probability": probabilite,
    }

    append_csv_record(
        C.DIR_LOGS / "predictions.csv",
        log,
        pseudonymiser_ancien_id=True,
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
    session_id = str(uuid.uuid4())

    try:
        # Préparer les données pour la prédiction
        usager_test = demandeur_emploi.model_dump()
        conseiller_id = usager_test["conseiller_id"].strip()
        usager_test["conseiller_id"] = conseiller_id
        usager_id_pseudonymise = pseudonymiser_usager_id(usager_test["usager_id"])
  
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
        journaliser_prediction(
            session_id,
            conseiller_id,
            usager_test,
            usager_id_pseudonymise,
            int(prediction),
            probability,
        )

        return {
            "session_id": session_id,
            "usager_id_pseudonymise": usager_id_pseudonymise,
            "prediction": int(prediction),
            "probability": round(probability, 4),
        }

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================
# Explanation
# ==========================
@app.post("/explain")
def explain(demandeur_emploi: DemandeurEmploi):
    modele = pipeline.named_steps["model"]
    if not isinstance(modele, SHAP_MODELS):
        raise HTTPException(
            status_code=422,
            detail=(
                f"SHAP TreeExplainer ne prend pas en charge {type(modele).__name__}. "
                "Modèles pris en charge : RandomForest, LightGBM et XGBoost."
            ),
        )

    retraining_config = active_artifact.get("retraining_config", {})
    features = retraining_config.get("features") or C.FEATURES_SCENARIO_2
    X = D.prepare_data(demandeur_emploi.model_dump(), features=features)
    X_transforme = pipeline.named_steps["preprocessor"].transform(X)
    if hasattr(X_transforme, "toarray"):
        X_transforme = X_transforme.toarray()

    explainer = shap.TreeExplainer(modele)
    shap_values = explainer.shap_values(X_transforme)
    if isinstance(shap_values, list):
        shap_values = [values.tolist() for values in shap_values]
    else:
        shap_values = shap_values.tolist()

    expected_value = explainer.expected_value
    if hasattr(expected_value, "tolist"):
        expected_value = expected_value.tolist()

    return {
        "classes": [int(classe) for classe in pipeline.classes_],
        "feature_names": noms_variables_transformees(
            pipeline.named_steps["preprocessor"]
        ),
        "shap_values": shap_values,
        "expected_value": expected_value,
    }



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




def calculer_cout_metier(y_reel, predictions) -> int:
    matrice = confusion_matrix(y_reel, predictions, labels=[0, 1, 2])
    return int(
        5 * matrice[2, 0]
        + 2 * matrice[2, 1]
        + matrice[0, 2]
        + matrice[1, 2]
    )


def candidat_est_meilleur(cout_actuel: int, cout_candidat: int) -> bool:
    return cout_candidat < cout_actuel


def entrainer_artefact(
    artifact: dict[str, object],
) -> tuple[dict[str, object], dict[str, float], int, int, int]:
    """Réentraîne une copie du meilleur pipeline et renvoie l'artefact prêt à sauver."""
    config = artifact.get("retraining_config")
    if not isinstance(config, dict) or not config.get("best_params"):
        raise ValueError("La configuration de réentraînement est absente de l'artefact")

    target = config.get("target", C.CIBLE)
    features = config.get("features", C.FEATURES_SCENARIO_2)
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
    cout_candidat = calculer_cout_metier(y_test, predictions)
    predictions_actuelles = artifact["pipeline"].predict(X_test)
    cout_actuel = calculer_cout_metier(y_test, predictions_actuelles)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "cout_metier": float(cout_candidat),
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
        cout_actuel,
    )


@app.post("/retrain")
def retrain():
    """Réentraîne le pipeline sélectionné à la demande avec l'historique et les feedbacks."""
    global active_artifact, active_model_name, pipeline

    if not MODEL_ARTIFACT_PATH.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                "L'artefact pipeline_prediction_retour_emploi.joblib est introuvable. "
                "Relancez d'abord la cellule de sauvegarde du notebook."
            ),
        )

    try:
        artifact = joblib.load(MODEL_ARTIFACT_PATH)
        (
            updated_artifact,
            metrics,
            training_rows,
            feedback_rows,
            cout_actuel,
        ) = entrainer_artefact(artifact)
        cout_candidat = int(metrics["cout_metier"])
        promouvoir = candidat_est_meilleur(cout_actuel, cout_candidat)

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
                    "promoted": str(promouvoir).lower(),
                    **{
                        f"best_{key}": str(value)
                        for key, value in retraining_config["best_params"].items()
                    },
                }
            )
            mlflow.log_metrics(
                {
                    **metrics,
                    "cout_metier_pipeline_actuel": float(cout_actuel),
                }
            )
            mlflow.set_tag("promotion_decision", "promoted" if promouvoir else "rejected")
            mlflow.sklearn.log_model(
                sk_model=updated_artifact["pipeline"],
                name="pipeline",
                skops_trusted_types=[
                    "collections.OrderedDict",
                    "lightgbm.basic.Booster",
                    "lightgbm.sklearn.LGBMClassifier",
                    "numpy.dtype",
                    "numpy.ravel",
                ],
            )

        if not promouvoir:
            return {
                "message": (
                    "Pipeline conservé : le candidat n'améliore pas le coût métier."
                ),
                "promoted": False,
                "model": active_model_name,
                "training_rows": training_rows,
                "feedback_rows": feedback_rows,
                "cout_metier_actuel": cout_actuel,
                "cout_metier_candidat": cout_candidat,
                "accuracy": round(metrics["accuracy"], 4),
                "f1_macro": round(metrics["f1_macro"], 4),
                "recall_classe_2": round(metrics["recall_classe_2"], 4),
            }

        C.DIR_ARCHIVE.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(
            MODEL_ARTIFACT_PATH,
            C.DIR_ARCHIVE / f"pipeline_prediction_retour_emploi_{timestamp}.joblib",
        )

        temporary_path = MODEL_ARTIFACT_PATH.with_name(
            f"{MODEL_ARTIFACT_PATH.stem}_{uuid.uuid4().hex}.tmp"
        )
        joblib.dump(updated_artifact, temporary_path)
        temporary_path.replace(MODEL_ARTIFACT_PATH)

        active_artifact = updated_artifact
        active_model_name = "pipeline_prediction_retour_emploi"
        pipeline = updated_artifact["pipeline"]

        return {
            "message": "Pipeline réentraîné et activé",
            "promoted": True,
            "model": active_model_name,
            "training_rows": training_rows,
            "feedback_rows": feedback_rows,
            "cout_metier_actuel": cout_actuel,
            "cout_metier_candidat": cout_candidat,
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
