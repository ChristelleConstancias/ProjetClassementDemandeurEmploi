import streamlit as st
from inference_history import (
    get_last_predictions,
    get_stats
)

st.set_page_config(
    page_title="Historique",
    page_icon="📜",
    layout="wide"
)

st.title("📜 Historique des inférences")

historique = get_last_predictions()

if historique.empty:
    st.info(
        "Aucune prédiction enregistrée."
    )

else:

    stats = get_stats()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Nombre total",
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