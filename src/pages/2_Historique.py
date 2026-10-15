import os
from datetime import date

import pandas as pd
import requests
import streamlit as st

from inference_history import filter_predictions, get_historique
from src.privacy import pseudonymiser_usager_id

API_URL = os.getenv("API_URL", "http://localhost:8000")
# API _ docker
#API_URL = "http://localhost:8001"
#API_URL = "https://projetclassementdemandeuremploi-latest.onrender.com"

st.set_page_config(page_title="Historique des prédictions", layout="wide")
st.title("Historique des prédictions")

st.session_state.setdefault("conseiller_id", "")


def memoriser_conseiller_id():
    st.session_state["conseiller_id"] = st.session_state["historique_conseiller_id"]


conseiller_id = st.sidebar.text_input(
    "Identifiant conseiller",
    value=st.session_state["conseiller_id"],
    key="historique_conseiller_id",
    on_change=memoriser_conseiller_id,
)
historique = get_historique()

if historique.empty:
    st.info("Aucune prédiction enregistrée.")
    st.stop()

dates = pd.to_datetime(historique["date_inference"], errors="coerce").dropna()
date_min = dates.min().date() if not dates.empty else date.today()
date_max = dates.max().date() if not dates.empty else date.today()

with st.sidebar:
    st.subheader("Filtres")
    mes_predictions = st.checkbox("Mes prédictions uniquement", value=True)
    usager_id = st.text_input("Identifiant usager")
    date_debut = st.date_input("Du", value=date_min)
    date_fin = st.date_input("Au", value=date_max)

try:
    usager_id_pseudonymise = (
        pseudonymiser_usager_id(usager_id) if usager_id.strip() else ""
    )
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if date_debut > date_fin:
    st.error("La date de début doit précéder la date de fin.")
    st.stop()

filtre = filter_predictions(
    historique,
    conseiller_id=conseiller_id,
    mes_predictions=mes_predictions,
    usager_id_pseudonymise=usager_id_pseudonymise,
    date_debut=date_debut,
    date_fin=date_fin,
)

if filtre.empty:
    st.info("Aucune prédiction ne correspond aux filtres sélectionnés.")
    st.stop()

tableau = filtre.copy()
tableau["Date et heure"] = pd.to_datetime(
    tableau["date_inference"], errors="coerce"
).dt.strftime("%d/%m/%Y %H:%M")
tableau = tableau.rename(
    columns={
        "code_rome_vise": "Code ROME visé",
        "departement": "Département",
        "prediction": "Prédiction",
    }
)
st.dataframe(
    tableau[["Date et heure", "Code ROME visé", "Département", "Prédiction"]],
    hide_index=True,
    use_container_width=True,
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
        options=filtre.index.tolist(),
        format_func=lambda index: (
            f"{filtre.loc[index, 'date_inference']} - "
            f"{filtre.loc[index, 'code_rome_vise']} - "
            f"classe {filtre.loc[index, 'prediction']}"
        ),
    )
    classe_reelle = st.selectbox(
        "Classe réellement constatée",
        options=list(classes_retour),
        format_func=lambda classe: classes_retour[classe],
    )
    envoyer_feedback = st.form_submit_button("Envoyer le feedback")

if envoyer_feedback:
    prediction = filtre.loc[index_prediction]
    pseudo_id = prediction.get("usager_id_pseudonymise")
    feedback_payload = {
        "session_id": str(prediction["session_id"]),
        "usager_id_pseudonymise": (
            None if pd.isna(pseudo_id) else str(pseudo_id)
        ),
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