import joblib
import numpy as np
import pandas as pd

from src import config as C, data as D, monitoring


def test_psi_numeric_and_categorical_detect_distribution_changes():
    values = np.arange(100, dtype=float)

    assert monitoring.psi(values, values) < 1e-8
    assert monitoring.psi(values, values + 100) >= monitoring.PSI_SEUIL_DERIVE

    categorical = monitoring.ecart_categoriel(
        pd.Series(["A"] * 20),
        pd.Series(["B"] * 20),
    )
    assert categorical["derive"]


def test_report_periodic_uses_only_recent_logs_and_selected_model_features(
    tmp_path, monkeypatch
):
    variables = C.FEATURES_SCENARIO_2
    model_path = tmp_path / "model.joblib"
    logs_path = tmp_path / "predictions.csv"
    joblib.dump({"retraining_config": {"features": variables}}, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setattr(C, "PREDICTIONS_LOG_FILE", logs_path)

    reference = pd.DataFrame(
        {
            "age": [30] * 20,
            "niveau_diplome": ["Bac", "Bac+2"] * 10,
            "anciennete_poste_ans": np.arange(20, dtype=float),
            "code_rome_vise": ["M1602", "J1506"] * 10,
            "code_insee_commune": ["75056", "13055"] * 10,
            "synthese_entretien": ["projet emploi formation"] * 20,
        }
    )
    monkeypatch.setattr(D, "load_data", lambda verbose=False: reference.copy())

    logs = pd.DataFrame(
        {
            "date_inference": [
                "2026-10-10 10:00:00",
                "2026-10-09 10:00:00",
                "2026-10-01 10:00:00",
            ],
            "usager_id_pseudonymise": ["secret-1", "secret-2", "secret-3"],
            "age": [30, 30, 30],
            "niveau_diplome": ["Bac", "Bac+2", "Bac"],
            "anciennete_poste_ans": [1.0, 2.0, 3.0],
            "code_rome_vise": ["M1602", "J1506", "M1602"],
            "code_insee_commune": ["75056", "13055", "75056"],
            "synthese_entretien": [
                "projet stable",
                "formation emploi",
                "ancien log hors fenêtre",
            ],
        }
    )
    logs.to_csv(logs_path, index=False)

    rapport, contexte = monitoring.rapport_periodique(
        fenetre_jours=7,
        maintenant=pd.Timestamp("2026-10-11 12:00:00"),
    )

    assert contexte["n_reference"] == len(reference)
    assert contexte["n_recent"] == 2
    assert contexte["scenario"] == "inconnu"
    assert "anciennete_poste_ans" in rapport["variable"].tolist()
    assert "synthese_entretien_nombre_mots" in rapport["variable"].tolist()
    assert not any("usager_id" in name for name in rapport["variable"])