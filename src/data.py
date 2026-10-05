"""Chargement des donnees generees.

Les colonnes internes prefixees par "_"  sont la verite terrain : reservees a
l'evaluation et au corrige, jamais donnees en entree au modele.
"""
from __future__ import annotations

import pandas as pd

from . import config as C

# Fonctions de chargement
def load_data(verbose: bool = True) -> pd.DataFrame:
    """Charge le jeu de donnees depuis data/dataset_trajectoire_emploi_Sujet Examen CISIA.csv."""
    if not C.DEMANDEUR_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {C.DEMANDEUR_FILE}\n"
            "Telechargez dataset_trajectoire_emploi_Sujet Examen CISIA et placez-le dans data/dataset_trajectoire_emploi_Sujet Examen CISIA.csv "
            "(voir data/README.md)."
        )
    if verbose:
        print(f"Chargement du CSV : {C.DEMANDEUR_FILE}")
    return pd.read_csv(C.DEMANDEUR_FILE)

# charge le fichier de feedback
def load_feedback(verbose: bool = True) -> pd.DataFrame:
    """Charge le jeu de donnees depuis outputs/feedback/feedback.csv."""
    if not C.FEEDBACK_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {C.FEEDBACK_FILE}\n"
            "Telechargez feedback.csv et placez-le dans outputs/feedback/feedback.csv "
            "(voir data/README.md)."
        )
    if verbose:
        print(f"Chargement du CSV : {C.FEEDBACK_FILE}")
    return pd.read_csv(C.FEEDBACK_FILE)

# Preparation des donnees d'entree pour le modele :
# calcul des variables derivees partage avec l'entrainement.
def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applique les transformations deterministes au jeu d'entrainement."""
    prepared = df.copy()
    prepared["departement"] = (
        prepared["code_insee_commune"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
        .str[:2]
    )
    prepared["famille_metier"] = (
        prepared["code_rome_vise"]
        .astype("string")
        .str.strip()
        .str[:3]
    )
    prepared["niveau_diplome_ordinal"] = prepared["niveau_diplome"].map(
        C.ordre_niveau_diplome
    )
    prepared["synthese_entretien"] = prepared["synthese_entretien"].fillna("").astype(str)

    mask = prepared["anciennete_poste_ans"] > (prepared["age"] - 16)
    prepared.loc[mask, "anciennete_poste_ans"] = prepared.loc[mask, "age"] - 16
    return prepared


def prepare_data(
    usager: dict,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Prepare une entree d'inference avec les memes transformations que l'entrainement."""
    prepared = prepare_training_data(pd.DataFrame([usager]))
    columns = features or C.FEATURES_SCENARIO_1
    return prepared.reindex(columns=columns)
