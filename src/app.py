"""
Interface Streamlit pour l'API de prédiction retour à l'emploi
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C

st.set_page_config(page_title="Prediction", layout="wide")

API_URL = "http://localhost:8000"

st.session_state.setdefault("conseiller_id", "cconstancias")
st.session_state.setdefault("prediction_payload", None)
st.session_state.setdefault("prediction_result", None)
st.session_state.setdefault("prediction_explanation", None)


def memoriser_conseiller_id():
    st.session_state["conseiller_id"] = st.session_state["prediction_conseiller_id"]


@st.cache_data
def charger_codes_rome():
    """
    Charge le référentiel ROME.
    Le fichier doit contenir :
    - code_rome
    - label_rome
    """

    df_rome = pd.read_csv(
        C.ROME_FILE,
        sep=",",
        encoding="utf-8-sig",
    )

    df_rome = (
        df_rome[["code_rome", "label_rome"]].drop_duplicates().sort_values("code_rome")
    )

    return df_rome


df_rome = charger_codes_rome()
labels_rome = df_rome.set_index("code_rome")["label_rome"].to_dict()


@st.cache_data
def charger_codes_postaux():
    """
    Charge les codes postaux existants depuis le fichier CSV.
    Retourne une liste unique de codes postaux à 5 caractères.
    """

    fichier = C.CODE_POSTAL_FILE

    if not fichier.exists():
        raise FileNotFoundError(f"Fichier introuvable : {fichier.resolve()}")

    df_codes_postaux = pd.read_csv(fichier, sep=",", dtype=str)

    df_codes_postaux = (
        df_codes_postaux[["code_commune", "nom_de_la_commune"]]
        .drop_duplicates()
        .sort_values("code_commune")
    )

    return df_codes_postaux


df_codes_postaux = charger_codes_postaux()
labels_communes = df_codes_postaux.set_index("code_commune")[
    "nom_de_la_commune"
].to_dict()

st.title("Nouvelle prédiction")

with st.sidebar:
    st.text_input(
        "Identifiant conseiller",
        value=st.session_state["conseiller_id"],
        key="prediction_conseiller_id",
        on_change=memoriser_conseiller_id,
    )
    conseiller_id = st.session_state["conseiller_id"]

# ==========================
# Formulaire
# ==========================

with st.form("prediction_form"):

    st.subheader("Informations de l'usager")

    usager_id = st.text_input("Identifiant usager", placeholder="ID_0000")

    age = st.number_input("Âge", min_value=16, max_value=67, value=30)

    niveau_diplome = st.selectbox(
        "Niveau de diplôme", ["Sans diplôme", "Bac", "Bac+2", "Bac+3", "Bac+5", "Bac+8"]
    )

    anciennete_poste_ans = st.number_input(
        "Ancienneté dans le dernier emploi (années)", min_value=0.0, value=2.0
    )

    code_rome_vise = st.selectbox(
        "Code ROME visé",
        options=df_rome["code_rome"].tolist(),
        format_func=lambda code: f"{code} - {labels_rome[code]}",
    )

    code_insee_commune = st.selectbox(
        "Code INSEE commune",
        options=df_codes_postaux["code_commune"].tolist(),
        format_func=lambda code: f"{code} - {labels_communes[code]}",
    )

    est_allocataire = st.selectbox("Allocataire", [0, 1])

    nationalite_hors_ue = st.selectbox("Nationalité hors UE", [0, 1])

    synthese_entretien = st.text_area("Synthèse entretien", height=150)

    submit = st.form_submit_button("Prédire")

# ==========================
# Appel API
# ==========================

if submit:
    st.session_state["prediction_payload"] = None
    st.session_state["prediction_result"] = None
    st.session_state["prediction_explanation"] = None

    if not conseiller_id.strip():
        st.error("Renseigne l'identifiant du conseiller avant de prédire.")
    elif not usager_id.strip():
        st.error("Renseigne l'identifiant de l'usager avant de prédire.")
    else:
        payload = {
            "usager_id": usager_id,
            "conseiller_id": conseiller_id.strip(),
            "age": age,
            "niveau_diplome": niveau_diplome,
            "anciennete_poste_ans": anciennete_poste_ans,
            "code_rome_vise": code_rome_vise,
            "code_insee_commune": code_insee_commune,
            "est_allocataire": est_allocataire,
            "nationalite_hors_ue": nationalite_hors_ue,
            "synthese_entretien": synthese_entretien,
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            st.session_state["prediction_payload"] = payload
            st.session_state["prediction_result"] = response.json()
        except requests.RequestException as exc:
            st.error(f"Erreur lors de l'appel à l'API : {exc}")
        except ValueError as exc:
            st.error(f"Réponse API invalide : {exc}")

resultat = st.session_state["prediction_result"]
if resultat:
    prediction = resultat["prediction"]
    libelles = {
        0: "Retour rapide (< 6 mois)",
        1: "Retour moyen (6 à 12 mois)",
        2: "Risque longue durée (> 12 mois)",
    }

    st.success("Prédiction réalisée")
    st.metric("Classe prédite", libelles.get(prediction, prediction))
    st.metric("Score de confiance", f"{resultat['probability']:.2%}")

    if st.button("Expliquer la prédiction", key="explain_prediction"):
        try:
            with st.spinner("Calcul de l'explication..."):
                response = requests.post(
                    f"{API_URL}/explain",
                    json=st.session_state["prediction_payload"],
                    timeout=30,
                )
                response.raise_for_status()
                st.session_state["prediction_explanation"] = response.json()
        except requests.HTTPError as exc:
            try:
                detail = response.json().get("detail", str(exc))
            except ValueError:
                detail = str(exc)
            st.error(f"Explication indisponible : {detail}")
        except requests.RequestException as exc:
            st.error(f"Erreur lors de l'appel à l'API : {exc}")
        except ValueError as exc:
            st.error(f"Réponse API invalide : {exc}")

    explication = st.session_state["prediction_explanation"]
    if explication:
        try:
            classes = explication["classes"]
            index_classe = classes.index(int(prediction))
            valeurs_shap = np.asarray(explication["shap_values"])
            nombre_variables = len(explication["feature_names"])
            nombre_classes = len(classes)

            if valeurs_shap.ndim == 3 and valeurs_shap.shape == (
                1,
                nombre_variables,
                nombre_classes,
            ):
                contributions = valeurs_shap[0, :, index_classe]
            elif valeurs_shap.ndim == 3 and valeurs_shap.shape == (
                nombre_classes,
                1,
                nombre_variables,
            ):
                contributions = valeurs_shap[index_classe, 0, :]
            elif valeurs_shap.ndim == 2 and valeurs_shap.shape == (
                nombre_variables,
                nombre_classes,
            ):
                contributions = valeurs_shap[:, index_classe]
            elif valeurs_shap.ndim == 2 and valeurs_shap.shape == (
                1,
                nombre_variables,
            ):
                contributions = valeurs_shap[0]
            else:
                raise ValueError(
                    f"Format SHAP non reconnu : {valeurs_shap.shape}"
                )

            details = pd.DataFrame(
                {
                    "Variable": explication["feature_names"],
                    "Contribution": contributions,
                }
            )
            details["Importance absolue"] = details["Contribution"].abs()
            principales = details.nlargest(15, "Importance absolue").sort_values(
                "Contribution"
            )

            st.subheader("Explication de la prédiction")
            st.caption(
                "Contributions des variables à la classe prédite : une valeur "
                "positive pousse le score vers cette classe, une valeur négative "
                "l'en éloigne."
            )
            st.bar_chart(principales.set_index("Variable")["Contribution"])
            st.dataframe(
                principales.drop(columns="Importance absolue"),
                hide_index=True,
                use_container_width=True,
            )
        except (KeyError, ValueError, IndexError) as exc:
            st.error(f"Impossible d'afficher l'explication : {exc}")
