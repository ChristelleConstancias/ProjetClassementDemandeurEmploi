from datetime import date

import pandas as pd

from src.inference_history import filter_predictions
from src.privacy import pseudonymiser_usager_id

# Tests pour la fonction filter_predictions du module inference_history
def test_filter_predictions_by_adviser_user_and_dates():
    history = pd.DataFrame(
        [
            {
                "conseiller_id": "C-01",
                "usager_id_pseudonymise": "U-A",
                "date_inference": "2026-10-01 09:30:00",
                "code_rome_vise": "M1602",
                "code_insee_commune": "75056",
                "prediction": 1,
            },
            {
                "conseiller_id": "C-01",
                "usager_id_pseudonymise": "U-A",
                "date_inference": "2026-10-03 10:15:00",
                "code_rome_vise": "J1506",
                "code_insee_commune": "13055",
                "prediction": 2,
            },
            {
                "conseiller_id": "C-02",
                "usager_id_pseudonymise": "U-A",
                "date_inference": "2026-10-02 11:00:00",
                "code_rome_vise": "A1203",
                "code_insee_commune": "69002",
                "prediction": 0,
            },
        ]
    )

    filtered = filter_predictions(
        history,
        conseiller_id="C-01",
        mes_predictions=True,
        usager_id_pseudonymise="U-A",
        date_debut=date(2026, 10, 2),
        date_fin=date(2026, 10, 3),
    )

    assert len(filtered) == 1
    assert filtered.iloc[0]["code_rome_vise"] == "J1506"
    assert filtered.iloc[0]["departement"] == "13"

# Test pour vérifier que les lignes héritées non attribuées peuvent être incluses dans les résultats
def test_filter_predictions_can_include_unattributed_legacy_rows():
    history = pd.DataFrame(
        [
            {
                "date_inference": "2026-10-01 09:30:00",
                "code_rome_vise": "M1602",
                "code_insee_commune": "75056",
                "prediction": 1,
            }
        ]
    )

    assert filter_predictions(history, conseiller_id="C-01").empty
    all_predictions = filter_predictions(history, mes_predictions=False)
    assert len(all_predictions) == 1
    assert all_predictions.iloc[0]["departement"] == "75"

# Test pour vérifier que le filtrage fonctionne correctement avec les identifiants pseudonymisés des usagers
def test_filter_predictions_pseudonymises_raw_usager_id(monkeypatch):
    monkeypatch.setenv("USAGER_ID_HMAC_KEY", "test-secret-key")
    pseudo_id = pseudonymiser_usager_id("ID_3070")
    history = pd.DataFrame(
        [
            {
                "usager_id_pseudonymise": pseudo_id,
                "date_inference": "2026-10-10 09:30:00",
                "code_rome_vise": "M1602",
                "code_insee_commune": "75056",
                "prediction": 1,
            }
        ]
    )

    filtered = filter_predictions(
        history,
        mes_predictions=False,
        usager_id_pseudonymise=pseudonymiser_usager_id(" ID_3070 "),
    )

    assert len(filtered) == 1
    assert filtered.iloc[0]["usager_id_pseudonymise"] == pseudo_id