"""
Interface Streamlit pour l'API de prédiction retour à l'emploi  
"""

import streamlit as st
import pandas as pd
import requests
from inference_history import (
    get_last_predictions,
    get_stats
)

API_URL = "http://localhost:8000"
CODE_POSTAL_FILE = "data/code_postal.csv"  # placer ici le CSV si disponible

@st.cache_data
def charger_codes_rome():
    """
    Charge le référentiel ROME.
    Le fichier doit contenir :
    - code_rome
    - label_rome
    """

    df_rome = pd.read_csv(
        "data/data_rome.csv",
        sep=",",
        encoding="utf-8-sig",
    )

    df_rome = (
        df_rome[
            ["code_rome", "label_rome"]
        ]
        .drop_duplicates()
        .sort_values("code_rome")
    )

    return df_rome

df_rome = charger_codes_rome()

@st.cache_data

def charger_codes_postaux():
    """
    Charge les codes postaux existants depuis le fichier CSV.
    Retourne une liste unique de codes postaux à 5 caractères.
    """

    fichier = CODE_POSTAL_FILE

    import os
    if not os.path.exists(fichier):
        raise FileNotFoundError(
            f"Fichier introuvable : {os.path.abspath(fichier)}"
        )

    df_codes_postaux = pd.read_csv(
        fichier,
        sep=",",
        dtype=str
    )

    df_codes_postaux = (
        df_codes_postaux[
            ["code_commune", "nom_de_la_commune"]
        ]
        .drop_duplicates()
        .sort_values("code_commune")
    )

    return df_codes_postaux

df_codes_postaux = charger_codes_postaux()

st.set_page_config(
    page_title="Orientation Emploi IA",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Orientation des Demandeurs d'Emploi")
st.markdown(
    "Prédiction du délai de retour à l'emploi "
    "à partir des informations renseignées."
)

with st.sidebar:
    st.subheader("Gestion du modèle")
    st.caption("Pipeline sélectionné selon le F1 macro et le rappel de la classe 2.")
    if st.button("Réentraîner le pipeline", use_container_width=True):
        with st.spinner("Réentraînement à partir de l'historique et des feedbacks..."):
            try:
                response = requests.post(
                    f"{API_URL}/retrain",
                    timeout=300,
                )
                response.raise_for_status()
                resultat_retrain = response.json()
                st.success(resultat_retrain["message"])
                st.metric("F1 macro", f"{resultat_retrain['f1_macro']:.3f}")
                st.metric(
                    "Rappel classe 2",
                    f"{resultat_retrain['recall_classe_2']:.3f}",
                )
                st.caption(
                    f"Données utilisées : {resultat_retrain['training_rows']} "
                    f"lignes, dont {resultat_retrain['feedback_rows']} feedbacks."
                )
            except requests.RequestException as exc:
                st.error(f"Échec du réentraînement : {exc}")

# ==========================
# Formulaire
# ==========================

with st.form("prediction_form"):

    st.subheader("Informations de l'usager")

    usager_id = st.text_input(
        "Identifiant usager",
        placeholder="ID_0000"
    )

    age = st.number_input(
        "Âge",
        min_value=16,
        max_value=67,
        value=30
    )

    niveau_diplome = st.selectbox(
        "Niveau de diplôme",
        [
            "Sans diplôme",
            "Bac",
            "Bac+2",
            "Bac+5"
        ]
    )

    anciennete_poste_ans = st.number_input(
        "Ancienneté dans le dernier emploi (années)",
        min_value=0.0,
        value=2.0
    )

    code_rome_vise = st.selectbox(
        "Code ROME visé",
        options=df_rome["code_rome"].tolist(),
        format_func=lambda x: (
            f"{x} - "
            f"{df_rome.loc[df_rome['code_rome'] == x, 'label_rome'].iloc[0]}"
        )
    )

    code_insee_commune = st.selectbox(
        "Code INSEE commune",
        options=df_codes_postaux["code_commune"].tolist(),
        format_func=lambda x: (
            f"{x} - "
            f"{df_codes_postaux.loc[df_codes_postaux['code_commune'] == x, 'nom_de_la_commune'].iloc[0]}"
        )
    )

    est_allocataire = st.selectbox(
        "Allocataire",
        [0, 1]
    )

    nationalite_hors_ue = st.selectbox(
        "Nationalité hors UE",
        [0, 1]
    )

    synthese_entretien = st.text_area(
        "Synthèse entretien",
        height=150
    )

    submit = st.form_submit_button(
        "Prédire"
    )

# ==========================
# Appel API
# ==========================

if submit:

    payload = {
        "usager_id": usager_id,
        "age": age,
        "niveau_diplome": niveau_diplome,
        "anciennete_poste_ans": anciennete_poste_ans,
        "code_rome_vise": code_rome_vise,
        "code_insee_commune": code_insee_commune,
        "est_allocataire": est_allocataire,
        "nationalite_hors_ue": nationalite_hors_ue,
        "synthese_entretien": synthese_entretien
    }

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:

            resultat = response.json()

            prediction = resultat["prediction"]
            probability = resultat["probability"]

            libelles = {
                0: "🟢 Retour rapide (< 6 mois)",
                1: "🟡 Retour moyen (6 à 12 mois)",
                2: "🔴 Risque longue durée (> 12 mois)"
            }

            st.success("Prédiction réalisée")

            st.metric(
                "Classe prédite",
                libelles.get(prediction, prediction)
            )

            st.metric(
                "Score de confiance",
                f"{probability:.2%}"
            )

        else:
            st.error(response.text)

    except Exception as e:
        st.error(f"Erreur : {e}")

# ==========================
# Historique des inférences
# ==========================

st.divider()

st.subheader("Historique des inférences")

historique = get_last_predictions()
print(historique)

if not historique.empty:
    stats = get_stats()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Nombre d'inférences",
        stats["nombre_inferences"]
    )

    col2.metric(
        "Classe 0",
        stats["classe_0"]
    )

    col3.metric(
        "Classe 1",
        stats["classe_1"]
    )

    col4.metric(
        "Classe 2",
        stats["classe_2"]
    )

    st.dataframe(
        historique,
        use_container_width=True
    )

    st.subheader("Envoyer un feedback")

    classes_retour = {
        0: "0 - Retour rapide (< 6 mois)",
        1: "1 - Retour moyen (6 à 12 mois)",
        2: "2 - Retour longue durée (> 12 mois)",
    }

    with st.form("feedback_form"):
        index_prediction = st.selectbox(
            "Prédiction concernée",
            options=historique.index.tolist(),
            format_func=lambda index: (
                f"{historique.loc[index, 'usager_id']} - "
                f"{historique.loc[index, 'date_inference']} - "
                f"classe prédite {historique.loc[index, 'prediction']}"
            ),
        )
        classe_reelle = st.selectbox(
            "Classe réellement constatée",
            options=list(classes_retour),
            format_func=lambda classe: classes_retour[classe],
        )
        envoyer_feedback = st.form_submit_button("Envoyer le feedback")

    if envoyer_feedback:
        prediction = historique.loc[index_prediction]
        feedback_payload = {
            "usager_id": str(prediction["usager_id"]),
            "age": float(prediction["age"]),
            "niveau_diplome": str(prediction["niveau_diplome"]),
            "anciennete_poste_ans": float(prediction["anciennete_poste_ans"]),
            "code_rome_vise": str(prediction["code_rome_vise"]),
            "code_insee_commune": str(prediction["code_insee_commune"]),
            "est_allocataire": float(prediction["est_allocataire"]),
            "nationalite_hors_ue": int(prediction["nationalite_hors_ue"]),
            "synthese_entretien": str(prediction["synthese_entretien"]),
            "prediction_modele": int(prediction["prediction"]),
            "classe_reelle": int(classe_reelle),
        }

        try:
            response = requests.post(
                f"{API_URL}/feedback",
                json=feedback_payload,
                timeout=10,
            )
            response.raise_for_status()
            st.success("Feedback enregistré.")
        except requests.RequestException as exc:
            st.error(f"Échec de l'envoi du feedback : {exc}")

else:
    st.info(
        "Aucune prédiction enregistrée."
    )