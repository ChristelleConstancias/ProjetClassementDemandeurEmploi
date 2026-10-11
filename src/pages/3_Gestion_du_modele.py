import os

import requests
import streamlit as st

from src.monitoring import rapport_periodique

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Gestion du modèle", layout="wide")
st.title("Gestion du modèle")

try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    health_response.raise_for_status()
    active_model = health_response.json()["model"]
    st.metric("Modèle actif", active_model)
except requests.RequestException as exc:
    st.error(f"API indisponible : {exc}")

with st.form("retraining_form"):
    confirmer = st.checkbox("Confirmer le réentraînement du modèle")
    submitted = st.form_submit_button("Réentraîner le modèle")

if submitted:
    if not confirmer:
        st.error("Confirme le réentraînement avant de continuer.")
    else:
        with st.spinner("Réentraînement en cours..."):
            try:
                response = requests.post(f"{API_URL}/retrain", timeout=300)
                response.raise_for_status()
                resultat = response.json()
                if resultat["promoted"]:
                    st.success(resultat["message"])
                else:
                    st.warning(resultat["message"])

                col_actuel, col_candidat = st.columns(2)
                col_actuel.metric(
                    "Coût métier du pipeline actif",
                    resultat["cout_metier_actuel"],
                )
                col_candidat.metric(
                    "Coût métier du candidat",
                    resultat["cout_metier_candidat"],
                    delta=(
                        resultat["cout_metier_candidat"]
                        - resultat["cout_metier_actuel"]
                    ),
                    delta_color="inverse",
                )
                st.metric("F1 macro", f"{resultat['f1_macro']:.3f}")
                st.metric("Rappel de la classe 2", f"{resultat['recall_classe_2']:.3f}")
                st.caption(
                    f"Données utilisées : {resultat['training_rows']} lignes, "
                    f"dont {resultat['feedback_rows']} feedbacks."
                )
            except requests.HTTPError as exc:
                try:
                    detail = response.json().get("detail", str(exc))
                except ValueError:
                    detail = str(exc)
                st.error(f"Échec du réentraînement : {detail}")
            except requests.RequestException as exc:
                st.error(f"Échec du réentraînement : {exc}")

st.divider()
st.subheader("Surveillance de la dérive des données")
st.caption(
    "Compare les variables du modèle actif aux prédictions récentes. "
    "Une dérive signale un changement des entrées, pas une baisse de qualité."
)
fenetre_jours = st.number_input(
    "Fenêtre des prédictions récentes (jours)",
    min_value=1,
    max_value=90,
    value=7,
    step=1,
    key="drift_window_days",
)

if st.button("Calculer le rapport de dérive", key="calculate_drift_report"):
    st.session_state.pop("drift_report", None)
    st.session_state.pop("drift_report_context", None)
    try:
        with st.spinner("Comparaison avec les données de référence..."):
            rapport, contexte = rapport_periodique(int(fenetre_jours))
        st.session_state["drift_report"] = rapport
        st.session_state["drift_report_context"] = contexte
    except (FileNotFoundError, ValueError, KeyError) as exc:
        st.error(f"Rapport de dérive indisponible : {exc}")

rapport = st.session_state.get("drift_report")
contexte = st.session_state.get("drift_report_context")
if rapport is not None and contexte is not None:
    col_reference, col_recent, col_dates = st.columns(3)
    col_reference.metric("Données de référence", contexte["n_reference"])
    col_recent.metric("Prédictions récentes", contexte["n_recent"])
    col_dates.metric("Scénario surveillé", contexte["scenario"])
    st.caption(
        f"Fenêtre récente : du {contexte['debut']} au {contexte['fin']} "
        f"({contexte['fenetre_jours']} jours)."
    )

    variables_en_derive = rapport.loc[rapport["derive"]]
    if variables_en_derive.empty:
        st.success("Aucun signal de dérive au seuil PSI configuré.")
    else:
        st.warning(
            f"{len(variables_en_derive)} indicateur(s) dépassent le seuil PSI. "
            "À examiner; ce résultat ne déclenche pas de réentraînement."
        )

    st.dataframe(
        rapport.sort_values("psi", ascending=False, na_position="last"),
        hide_index=True,
        use_container_width=True,
    )