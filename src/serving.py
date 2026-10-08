"""Contrat d'entree pour un pipeline scenario 1 exporte du notebook."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from . import config as C


def aplatir_texte(valeurs):
    """Adaptateur 2D -> 1D serialisable pour le vectoriseur TF-IDF."""
    return valeurs.ravel()


def preparer_entree(usager: dict) -> pd.DataFrame:
    """Reproduit les variables derivees deterministes du notebook."""
    donnees = dict(usager)
    commune = str(donnees["code_insee_commune"]).strip()
    donnees["departement"] = commune.zfill(5)[:2]
    diplome = donnees.get("niveau_diplome")
    donnees["niveau_diplome_ordinal"] = C.ordre_niveau_diplome.get(diplome)
    age = donnees.get("age")
    if age is not None and donnees["anciennete_poste_ans"] > age - 16:
        donnees["anciennete_poste_ans"] = age - 16
    donnees["synthese_entretien"] = (
        str(donnees.get("synthese_entretien") or "").strip() or "Texte manquant"
    )
    return pd.DataFrame([donnees], columns=C.FEATURES_SCENARIO_1)


def charger_modele(chemin: Path):
    if not chemin.is_file():
        raise FileNotFoundError(f"Modele approuve absent : {chemin}")
    return joblib.load(chemin)
