"""Configuration centralisee

Toutes les valeurs pilotant le flux, le modele, la surveillance et la decision
sont ici. Ne pas coder de valeur en dur ailleurs dans le projet.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Reproductibilite
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Chemins
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEMANDEUR_FILE = DATA_DIR / "dataset_trajectoire_emploi_Sujet Examen CISIA.csv"  # placer ici le CSV si disponible
CODE_POSTAL_FILE = DATA_DIR / "code_postal.csv"  # placer ici le CSV si disponible
MLFLOW_DB_FILE = ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB_FILE.as_posix()}",
)
MLFLOW_EXPERIMENT_INVESTIGATION = "investigation"
MLFLOW_EXPERIMENT_RETRAINING = "retraining"

OUTPUTS = ROOT / "outputs"
DIR_FEEDBACK = OUTPUTS / "feedback"
FEEDBACK_FILE = OUTPUTS / "feedback" / "feedback.csv"  # placer ici le CSV si disponible
DIR_ARCHIVE = OUTPUTS / "archive"
DIR_LOGS = OUTPUTS / "logs"
PREDICTIONS_LOG_FILE = DIR_LOGS / "predictions.csv"


DIR_MODELS = ROOT /"artifacts" 

# --------------------------------------------------------------------------
ordre_niveau_diplome = {
    "Sans diplôme": 1,
    "Bac": 2,
    "Bac + 2": 3,
    "Bac + 3": 4,
    "Bac + 5": 5,
    "Bac + 8": 6
}

# --------------------------------------------------------------------------
# Variables :
#    "usager_id", "age", "niveau_diplome", "anciennete_poste_ans",
#    "code_rome_vise", "code_insee_commune", "est_allocataire",
#    "nationalite_hors_ue", "synthese_entretien", "classe_retour_emploi"
# --------------------------------------------------------------------------
FEATURES_NUM = [
    "age", 
    "anciennete_poste_ans"
]

FEATURES_CAT_ORDINAL= ["niveau_diplome_ordinal"]

FEATURES_CAT = [
    "code_rome_vise", 
    "code_insee_commune", 
    "est_allocataire", 
    "nationalite_hors_ue",
]

FEATURE_TEXT = ["synthese_entretien"]

FEATURES = FEATURES_NUM + FEATURES_CAT_ORDINAL + FEATURES_CAT + FEATURE_TEXT

CIBLE = "classe_retour_emploi"

# --------------------------------------------------------------------------
# Scenario 1 : Approche multimodale complète : Utilisation de l'intégralité 
# des variables (tabulaires + texte vectorisé). 
# --------------------------------------------------------------------------
FEATURES_CAT_SCENARIO_1 = [
    "est_allocataire", 
    "nationalite_hors_ue",
    "famille_metier",
    #"code_rome_vise", 
    "departement"
]
FEATURES_SCENARIO_1 = FEATURES_NUM + FEATURES_CAT_ORDINAL + FEATURES_CAT_SCENARIO_1 + FEATURE_TEXT

# --------------------------------------------------------------------------
# 3 variables sensibles : "nationalite_hors_ue", "age", "est_allocataire"
#
# Scenario 2 : Sans variables sensibles (Approche Éthique) : on retire alors les 
# variables sensibles comme "nationalite_hors_ue", "age", "est_allocataire"
#
# scenario 2B : Approche tabulaire en supprimant uniquement nationalite_hors_ue 
# et en conservant les autres variables sensibles age, est_allocataire
#
# scenario 2C : Approche tabulaire en supprimant 2 variables sensibles 
#   (nationalite_hors_ue, est_allocataire) on garde a
# --------------------------------------------------------------------------
FEATURES_NUM_SCENARIO_2 = [
    "anciennete_poste_ans" 
]
FEATURES_CAT_SCENARIO_2 = [
    "departement",
    "famille_metier",
]
FEATURES_CAT_SCENARIO_2_B = [
    "departement",
    "famille_metier",
    "est_allocataire", 
]
FEATURES_SCENARIO_2 = FEATURES_NUM_SCENARIO_2 + FEATURES_CAT_ORDINAL + FEATURES_CAT_SCENARIO_2 + FEATURE_TEXT
FEATURES_SCENARIO_2B = FEATURES_NUM + FEATURES_CAT_ORDINAL + FEATURES_CAT_SCENARIO_2_B + FEATURE_TEXT
FEATURES_SCENARIO_2C = FEATURES_NUM + FEATURES_CAT_ORDINAL + FEATURES_CAT_SCENARIO_2 + FEATURE_TEXT

# --------------------------------------------------------------------------
# Scenario 3 •	Scénario 3 — Diagnostic par le texte seul (NLP pure) : 
# Prédiction basée exclusivement sur la synthèse écrite de l'entretien de 
# cadrage pour tester la robustesse du modèle de langage face à des verbatims 
# contenant du bruit stochastique. 
# --------------------------------------------------------------------------
FEATURES_SCENARIO_3 = FEATURE_TEXT

# --------------------------------------------------------------------------
# Scenario 4 : Données contextuelles pures (Tabulaire seul) : 
# Prédicteur basé uniquement sur l'âge, les diplômes, l'ancienneté et 
# la géographie, sans l'apport du contexte sémantique textuel. 
# --------------------------------------------------------------------------
FEATURES_CAT_SCENARIO_4 = [
    "departement",
]
FEATURES_SCENARIO_4 = FEATURES_NUM + FEATURES_CAT_ORDINAL + FEATURES_CAT_SCENARIO_4
