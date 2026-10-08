"""Reentraine un candidat a partir d'un artefact approuve et du feedback confirme."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from . import config as C
from .data import load_data
from .serving import charger_modele, preparer_entree


# ==========================
# Préparation des données de référence
# code postaux regroupé en départements
# niveau de diplôme transformé en ordinal
# synthese entretien nettoyée
# anciennete_poste_ans ajustée par rapport à l'âge
# ==========================
def preparer_reference():
    donnees = load_data(verbose=False)
    # La preparation correspond aux cellules deterministes du notebook.
    donnees["departement"] = (
        donnees["code_insee_commune"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
        .str[:2]
    )
    donnees["niveau_diplome_ordinal"] = donnees["niveau_diplome"].map(
        C.ordre_niveau_diplome
    )
    donnees["synthese_entretien"] = (
        donnees["synthese_entretien"]
        .astype("string")
        .fillna("Texte manquant")
        .str.strip()
        .replace("", "Texte manquant")
    )
    masque = donnees["anciennete_poste_ans"] > (donnees["age"] - 16)
    donnees.loc[masque, "anciennete_poste_ans"] = donnees.loc[masque, "age"] - 16
    return donnees


def preparer_feedback(feedback: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Valide le CSV de feedback et convertit ses entrees au format du pipeline."""
    colonnes_entree = {
        "age",
        "niveau_diplome",
        "anciennete_poste_ans",
        "code_rome_vise",
        "code_insee_commune",
        "est_allocataire",
        "nationalite_hors_ue",
        "synthese_entretien",
        "classe_reelle",
    }
    colonnes_manquantes = colonnes_entree.difference(feedback.columns)
    if colonnes_manquantes:
        raise ValueError(
            "Colonnes manquantes dans le CSV de feedback : "
            f"{sorted(colonnes_manquantes)}"
        )

    # verifie la classe reelle
    feedback = feedback.copy()
    feedback["classe_reelle"] = pd.to_numeric(
        feedback["classe_reelle"], errors="coerce"
    )
    feedback = feedback.dropna(subset=["classe_reelle"])
    if feedback.empty:
        raise ValueError("Aucun feedback avec une classe réelle valide")

    labels = feedback["classe_reelle"]
    if not labels.isin([0, 1, 2]).all():
        raise ValueError("Les classes réelles doivent être 0, 1 ou 2")

    entrees = pd.concat(
        [
            preparer_entree(enregistrement)
            for enregistrement in feedback.to_dict(orient="records")
        ],
        ignore_index=True,
    )
    return entrees, labels.astype(int).reset_index(drop=True)


# Préparer les données de référence pour l'entraînement et l'évaluation
def main():
    approved = Path(os.getenv("MODEL_PATH", str(C.DIR_MODELS / "pipeline.joblib")))
    feedback_file = Path(os.getenv("FEEDBACK_FILE", str(C.FEEDBACK_FILE)))
    if not feedback_file.is_file():
        raise FileNotFoundError(f"Fichier de feedback absent : {feedback_file}")

    feedback = pd.read_csv(
        feedback_file,
        dtype={"code_insee_commune": "string"},
    )
    retours, labels = preparer_feedback(feedback)
    if len(labels) < 20:
        raise ValueError(
            "Au moins 20 feedbacks confirmés sont nécessaires; "
            f"{len(labels)} disponible(s)"
        )

    reference = preparer_reference()
    indices_train, indices_test = train_test_split(
        reference.index,
        test_size=0.2,
        random_state=C.SEED,
        stratify=reference[C.CIBLE],
    )
    entrainement = pd.concat(
        [
            reference.loc[indices_train, C.FEATURES_SCENARIO_1].reset_index(drop=True),
            retours,
        ],
        ignore_index=True,
    )
    cibles = pd.concat(
        [reference.loc[indices_train, C.CIBLE].reset_index(drop=True), labels],
        ignore_index=True,
    )
    pipeline = clone(charger_modele(approved))
    pipeline.fit(entrainement, cibles)
    y_test = reference.loc[indices_test, C.CIBLE]
    pred = pipeline.predict(reference.loc[indices_test, C.FEATURES_SCENARIO_1])
    mesures = {
        "f1_macro_reference": float(f1_score(y_test, pred, average="macro")),
        "balanced_accuracy_reference": float(balanced_accuracy_score(y_test, pred)),
        "feedback_count": len(labels),
    }
    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT_RETRAINING)
    with mlflow.start_run() as run:
        mlflow.log_metrics(mesures)
        mlflow.log_params(
            {
                key: str(value)
                for key, value in pipeline.named_steps["modele"].get_params().items()
                if key in {"n_estimators", "max_depth", "learning_rate", "random_state"}
            }
        )
        mlflow.sklearn.log_model(pipeline, name="pipeline")
        version = run.info.run_id
    destination = C.ROOT / "artifacts" / "candidates"
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination / f"{version}.joblib")
    (destination / f"{version}.json").write_text(
        json.dumps({"date_utc": datetime.now(timezone.utc).isoformat(), **mesures}),
        encoding="utf-8",
    )
    print(f"Candidat {version} enregistre; validation et promotion manuelles requises")


if __name__ == "__main__":
    main()
