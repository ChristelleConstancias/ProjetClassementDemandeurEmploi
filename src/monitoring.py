"""Surveillance en production : ecart au domaine d'apprentissage.

Implementation volontairement legere (numpy/scipy) pour tourner partout, y
compris hors ligne. En production, Evidently fournit les memes mesures de
derive de donnees avec plus d'outillage ; les ressources apprenants pointent
vers cette bibliotheque.

Ecart de distribution : PSI (taille d'effet, porte la decision) complete par
un test de Kolmogorov-Smirnov (indicatif).
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from . import config as C, data as D

PSI_SEUIL_DERIVE = 0.25
EPSILON_PSI = 1e-6


# --------------------------------------------------------------------------
# Ecart de distribution
# --------------------------------------------------------------------------
def psi(reference: np.ndarray, courant: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index entre deux echantillons continus."""
    reference = pd.to_numeric(pd.Series(reference), errors="coerce").to_numpy(
        dtype=float
    )
    courant = pd.to_numeric(pd.Series(courant), errors="coerce").to_numpy(
        dtype=float
    )
    reference = reference[np.isfinite(reference)]
    courant = courant[np.isfinite(courant)]

    if n_bins < 1:
        raise ValueError("n_bins doit être supérieur ou égal à 1")
    if reference.size == 0 or courant.size == 0:
        return float("nan")

    quantiles = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    if np.ptp(reference) == 0:
        marge = max(abs(float(reference[0])) * EPSILON_PSI, EPSILON_PSI)
        bornes = np.array(
            [-np.inf, reference[0] - marge, reference[0] + marge, np.inf]
        )
    else:
        bornes = np.unique(
            np.concatenate(([-np.inf], quantiles[1:-1], [np.inf]))
        )

    reference_counts, _ = np.histogram(reference, bins=bornes)
    courant_counts, _ = np.histogram(courant, bins=bornes)
    return _psi_depuis_effectifs(reference_counts, courant_counts)


def _psi_depuis_effectifs(reference: np.ndarray, courant: np.ndarray) -> float:
    if reference.size == 0 or courant.size == 0:
        return float("nan")

    ref = (reference + EPSILON_PSI) / (
        reference.sum() + EPSILON_PSI * reference.size
    )
    cur = (courant + EPSILON_PSI) / (
        courant.sum() + EPSILON_PSI * courant.size
    )
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def ks_pvalue(reference: np.ndarray, courant: np.ndarray) -> float:
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").to_numpy(dtype=float)
    cur = pd.to_numeric(pd.Series(courant), errors="coerce").to_numpy(dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 10 or len(cur) < 10:
        return float("nan")
    return float(ks_2samp(ref, cur).pvalue)


def ecart_variable(reference: np.ndarray, courant: np.ndarray) -> dict:
    """Ecart pour une variable.

    Le PSI (taille d'effet) porte la decision : il est robuste au volume, la ou
    le KS declenche pour un ecart infime des que l'echantillon est grand. Le KS
    est conserve a titre indicatif uniquement.
    """

    reference = pd.to_numeric(pd.Series(reference), errors="coerce").to_numpy(
        dtype=float
    )
    courant = pd.to_numeric(pd.Series(courant), errors="coerce").to_numpy(
        dtype=float
    )
    reference = reference[np.isfinite(reference)]
    courant = courant[np.isfinite(courant)]
    psi_value = psi(reference, courant)

    if len(reference) < 10 or len(courant) < 10:
        ks_statistic = float("nan")
        p_value = float("nan")
    else:
        ks_result = ks_2samp(reference, courant)
        ks_statistic = float(ks_result.statistic)
        p_value = float(ks_result.pvalue)

    return {
        "psi": psi_value,
        "ks_stat": ks_statistic,
        "ks_pvalue": p_value,
        "derive": bool(np.isfinite(psi_value) and psi_value >= PSI_SEUIL_DERIVE),
    }


def ecart_categoriel(reference: pd.Series, courant: pd.Series) -> dict:
    """PSI entre distributions catégorielles, KS n'étant pas adapté ici."""
    ref = reference.astype("string").fillna("<manquant>")
    cur = courant.astype("string").fillna("<manquant>")
    if ref.empty or cur.empty:
        return {
            "psi": float("nan"),
            "ks_stat": float("nan"),
            "ks_pvalue": float("nan"),
            "derive": False,
        }

    categories = pd.Index(ref.unique()).union(pd.Index(cur.unique()))
    ref_counts = ref.value_counts().reindex(categories, fill_value=0).to_numpy()
    cur_counts = cur.value_counts().reindex(categories, fill_value=0).to_numpy()
    psi_value = _psi_depuis_effectifs(ref_counts, cur_counts)
    return {
        "psi": psi_value,
        "ks_stat": float("nan"),
        "ks_pvalue": float("nan"),
        "derive": bool(np.isfinite(psi_value) and psi_value >= PSI_SEUIL_DERIVE),
    }


# Fonctions de surveillance et de reporting des écarts.
def rapport_ecart(
    reference: pd.DataFrame, courant: pd.DataFrame, variables: list[str] | None = None
) -> pd.DataFrame:
    """Ecart variable par variable entre une reference et une fenetre courante."""
    variables = C.FEATURES_SCENARIO_2 if variables is None else variables
    lignes = []
    for v in variables:
        if v not in reference.columns or v not in courant.columns:
            raise ValueError(f"Variable de surveillance absente : {v}")

        if v in C.FEATURE_TEXT:
            ref_text = reference[v].fillna("").astype(str).str.strip()
            cur_text = courant[v].fillna("").astype(str).str.strip()
            mesures_texte = [
                (
                    f"{v}_nombre_mots",
                    ecart_variable(
                        ref_text.str.split().str.len().to_numpy(),
                        cur_text.str.split().str.len().to_numpy(),
                    ),
                ),
                (
                    f"{v}_texte_vide",
                    ecart_categoriel(ref_text.eq(""), cur_text.eq("")),
                ),
            ]
            for nom, resultat in mesures_texte:
                lignes.append(
                    {
                        "variable": nom,
                        "type_mesure": "texte",
                        **resultat,
                        "n_reference": len(reference),
                        "n_recent": len(courant),
                    }
                )
        elif pd.api.types.is_numeric_dtype(reference[v]):
            resultat = ecart_variable(
                reference[v].to_numpy(), courant[v].to_numpy()
            )
            lignes.append(
                {
                    "variable": v,
                    "type_mesure": "numerique",
                    **resultat,
                    "n_reference": len(reference),
                    "n_recent": len(courant),
                }
            )
        else:
            resultat = ecart_categoriel(reference[v], courant[v])
            lignes.append(
                {
                    "variable": v,
                    "type_mesure": "categorielle",
                    **resultat,
                    "n_reference": len(reference),
                    "n_recent": len(courant),
                }
            )

    return pd.DataFrame(lignes)


def rapport_periodique(
    fenetre_jours: int = 7,
    maintenant: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare les logs récents aux données de référence du modèle actif."""
    if fenetre_jours < 1:
        raise ValueError("La fenêtre doit contenir au moins un jour")

    chemin_modele = Path(
        os.getenv(
            "MODEL_PATH",
            str(C.DIR_MODELS / "pipeline_prediction_retour_emploi.joblib"),
        )
    )
    if not chemin_modele.is_file():
        raise FileNotFoundError(f"Artefact du modèle actif absent : {chemin_modele}")

    artefact = joblib.load(chemin_modele)
    configuration = artefact.get("retraining_config", {})
    variables = configuration.get("features") or C.FEATURES_SCENARIO_2

    chemin_logs = C.PREDICTIONS_LOG_FILE
    if not chemin_logs.is_file():
        raise FileNotFoundError(f"Fichier de logs absent : {chemin_logs}")
    try:
        logs = pd.read_csv(
            chemin_logs,
            dtype={
                "code_insee_commune": "string",
                "code_rome_vise": "string",
            },
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Le fichier de logs ne contient aucun enregistrement") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            "Le CSV de prédictions contient des lignes mal formées; "
            "le rapport est interrompu pour éviter un résultat incomplet."
        ) from exc

    if "date_inference" not in logs.columns:
        raise ValueError("La colonne date_inference est absente des logs")

    dates = pd.to_datetime(logs["date_inference"], errors="coerce")
    fin = pd.Timestamp.now() if maintenant is None else pd.Timestamp(maintenant)
    if fin.tzinfo is not None:
        fin = fin.tz_localize(None)
    debut = fin - pd.Timedelta(days=fenetre_jours)
    masque_recent = dates.ge(debut) & dates.le(fin)
    logs_recents = logs.loc[masque_recent].copy()
    if logs_recents.empty:
        raise ValueError(
            f"Aucune prédiction enregistrée dans les {fenetre_jours} derniers jours"
        )

    donnees_reference = D.prepare_training_data(D.load_data(verbose=False))
    donnees_recentes = D.prepare_training_data(logs_recents)
    rapport = rapport_ecart(donnees_reference, donnees_recentes, variables)
    contexte = {
        "fenetre_jours": fenetre_jours,
        "debut": debut.strftime("%Y-%m-%d %H:%M"),
        "fin": fin.strftime("%Y-%m-%d %H:%M"),
        "n_reference": len(donnees_reference),
        "n_recent": len(donnees_recentes),
        "scenario": configuration.get("scenario_name", "inconnu"),
    }
    return rapport, contexte


# Fonctions de surveillance et de reporting des écarts.
def surveiller(
    reference: pd.DataFrame, stream: pd.DataFrame, segmenter: bool = True
) -> pd.DataFrame:
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

                courant_parc = courant[courant["parc"] == parc]

                # analyse par parc
                r = rapport_ecart(reference, courant_parc)

                r["semaine"] = semaine
                r["perimetre"] = parc

                rapports.append(r)

    return pd.concat(rapports, ignore_index=True)


# Fonctions de synthèse des dérives.
def resume_derive(rapport: pd.DataFrame) -> pd.DataFrame:
    """Nb de variables en derive par semaine, au niveau global."""
    g = rapport[rapport["perimetre"] == "__global__"]
    return (
        g.groupby("semaine")["derive"]
        .sum()
        .rename("n_variables_en_derive")
        .reset_index()
    )
