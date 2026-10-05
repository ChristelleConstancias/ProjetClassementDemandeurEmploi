"""Surveillance en production : ecart au domaine d'apprentissage.

Implementation volontairement legere (numpy/scipy) pour tourner partout, y
compris hors ligne. En production, Evidently fournit les memes mesures de
derive de donnees avec plus d'outillage ; les ressources apprenants pointent
vers cette bibliotheque.

Ecart de distribution : PSI (taille d'effet, porte la decision) complete par
un test de Kolmogorov-Smirnov (indicatif).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp

from . import config as C



# --------------------------------------------------------------------------
# Ecart de distribution
# --------------------------------------------------------------------------
def psi(reference: np.ndarray, courant: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index entre deux echantillons continus."""
    # TODO (etape brief) : implementer cette fonction.
    # Indice : histogramme sur des quantiles de la reference, puis somme (p_cur - p_ref) * ln(p_cur / p_ref).
    raise NotImplementedError("A completer : psi")


def ks_pvalue(reference: np.ndarray, courant: np.ndarray) -> float:
    ref = reference[~np.isnan(reference)]
    cur = courant[~np.isnan(courant)]
    if len(ref) < 10 or len(cur) < 10:
        return float("nan")
    return float(stats.ks_2samp(ref, cur).pvalue)


def ecart_variable(reference: np.ndarray, courant: np.ndarray) -> dict:
    """Ecart pour une variable.

    Le PSI (taille d'effet) porte la decision : il est robuste au volume, la ou
    le KS declenche pour un ecart infime des que l'echantillon est grand. Le KS
    est conserve a titre indicatif uniquement.
    """

    # suppression des NaN
    reference = reference[~np.isnan(reference)]
    courant = courant[~np.isnan(courant)]

    # bins à partir de la distribution de référence
    bornes = np.quantile(
        reference,
        np.linspace(0, 1, 11)
    )

    bornes = np.unique(bornes)

    if len(bornes) < 2:
        return {
            "psi": 0.0,
            "ks_stat": 0.0,
            "ks_pvalue": 1.0,
            "derive": False,
        }

    ref_hist, _ = np.histogram(reference, bins=bornes)
    cur_hist, _ = np.histogram(courant, bins=bornes)

    ref_pct = ref_hist / ref_hist.sum()
    cur_pct = cur_hist / cur_hist.sum()

    epsilon = 1e-6

    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    psi = np.sum(
        (ref_pct - cur_pct)
        * np.log(ref_pct / cur_pct)
    )

    ks_stat, ks_pvalue = ks_2samp(
        reference,
        courant
    )

    return {
        "psi": float(psi),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),

        # décision métier basée sur la taille d'effet
        "derive": bool(psi >= 0.25)
    }

# Fonctions de surveillance et de reporting des écarts.
def rapport_ecart(reference: pd.DataFrame, courant: pd.DataFrame,
                  variables: list[str] | None = None) -> pd.DataFrame:
    """Ecart variable par variable entre une reference et une fenetre courante."""
    variables = variables or C.VARS_SURVEILLANCE
    lignes = []
    for v in variables:
        e = ecart_variable(reference[v].to_numpy(), courant[v].to_numpy())
        e["variable"] = v
        lignes.append(e)
    return pd.DataFrame(lignes)[["variable", "psi", "ks_stat", "ks_pvalue", "derive"]]

# Fonctions de surveillance et de reporting des écarts.
def surveiller(reference: pd.DataFrame, stream: pd.DataFrame,
               segmenter: bool = True) -> pd.DataFrame:
    rapports = []

    # on boucle sur les semaines : colonne semaine du Dataframe
    for semaine in sorted(stream["semaine"].unique()):

        courant = stream[stream["semaine"] == semaine]

        # analyse globale
        r = rapport_ecart(reference, courant)

        r["semaine"] = semaine
        r["perimetre"] = "__global__"

        rapports.append(r)

        # analyse par parc
        if segmenter:

            for parc in courant["parc"].unique():

                courant_parc = courant[
                    courant["parc"] == parc
                ]

                # analyse par parc
                r = rapport_ecart(
                    reference,
                    courant_parc
                )

                r["semaine"] = semaine
                r["perimetre"] = parc

                rapports.append(r)

    return pd.concat(
        rapports,
        ignore_index=True
    )

# Fonctions de synthèse des dérives.
def resume_derive(rapport: pd.DataFrame) -> pd.DataFrame:
    """Nb de variables en derive par semaine, au niveau global."""
    g = rapport[rapport["perimetre"] == "__global__"]
    return (g.groupby("semaine")["derive"].sum()
            .rename("n_variables_en_derive").reset_index())
