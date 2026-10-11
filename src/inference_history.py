"""
Gestion de l'historique des inférences
"""

from datetime import date

import pandas as pd

# from . import config as C
# from src import config as C, data as D, models as M
# import config as C
# import data as D
# import models as M
if __package__:
    from .config import PREDICTIONS_LOG_FILE
else:
    from config import PREDICTIONS_LOG_FILE

# Fonctions pour gérer l'historique des inférences pour la page 2_Historique
def get_historique():
    """
    Retourne l'historique complet.
    """
    if not PREDICTIONS_LOG_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(
        PREDICTIONS_LOG_FILE,
        dtype={
            "session_id": "string",
            "conseiller_id": "string",
            "usager_id_pseudonymise": "string",
            "code_rome_vise": "string",
        },
    )

# Fonction pour filtrer les prédictions selon différents critères
def filter_predictions(
    historique: pd.DataFrame,
    *,
    conseiller_id: str = "",
    mes_predictions: bool = True,
    usager_id_pseudonymise: str = "",
    date_debut: date | None = None,
    date_fin: date | None = None,
) -> pd.DataFrame:
    filtre = historique.copy()

    if mes_predictions:
        if not conseiller_id.strip() or "conseiller_id" not in filtre.columns:
            return filtre.iloc[0:0]
        filtre = filtre.loc[
            filtre["conseiller_id"].fillna("").astype(str).str.strip()
            == conseiller_id.strip()
        ]

    if usager_id_pseudonymise.strip():
        if "usager_id_pseudonymise" not in filtre.columns:
            return filtre.iloc[0:0]
        filtre = filtre.loc[
            filtre["usager_id_pseudonymise"].fillna("").astype(str)
            == usager_id_pseudonymise.strip()
        ]

    if "date_inference" in filtre.columns:
        dates = pd.to_datetime(filtre["date_inference"], errors="coerce")
        if date_debut is not None:
            filtre = filtre.loc[dates.dt.date >= date_debut]
            dates = dates.loc[filtre.index]
        if date_fin is not None:
            filtre = filtre.loc[dates.dt.date <= date_fin]

    if "departement" not in filtre.columns and "code_insee_commune" in filtre.columns:
        filtre["departement"] = (
            filtre["code_insee_commune"]
            .astype("string")
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(5)
            .str[:2]
        )

    return filtre


# Fonction pour obtenir les N dernières prédictions
def get_last_predictions(n=20):
    """
    Retourne les N dernières prédictions.
    """
    historique = get_historique()

    if historique.empty:
        return historique

    return historique.tail(n).iloc[::-1]


def get_stats():
    """
    Retourne quelques statistiques.
    """
    historique = get_historique()

    if historique.empty:
        return {}

    return {
        "nombre_inferences": len(historique),
        "classe_0": (historique["prediction"] == 0).sum(),
        "classe_1": (historique["prediction"] == 1).sum(),
        "classe_2": (historique["prediction"] == 2).sum(),
    }
