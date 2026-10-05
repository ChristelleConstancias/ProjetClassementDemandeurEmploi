"""
Gestion de l'historique des inférences
"""

from pathlib import Path
import pandas as pd
#from . import config as C
#from src import config as C, data as D, models as M
#import config as C
#import data as D
#import models as M
from config import (
    PREDICTIONS_LOG_FILE    
)


def get_historique():
    """
    Retourne l'historique complet.
    """
    if not PREDICTIONS_LOG_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(PREDICTIONS_LOG_FILE)


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
