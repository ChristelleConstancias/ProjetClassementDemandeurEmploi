from src.api import candidat_est_meilleur, calculer_cout_metier
from src import api


def test_active_model_loader_allows_missing_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        api,
        "MODEL_ARTIFACT_PATH",
        tmp_path / "pipeline_prediction_retour_emploi.joblib",
    )

    def artefact_historique_absent(_nom):
        raise FileNotFoundError

    monkeypatch.setattr(api.M, "charger_artefact", artefact_historique_absent)

    artifact, model_name = api.charger_modele_actif()

    assert artifact == {"pipeline": None, "retraining_config": {}}
    assert model_name == "unavailable"


def test_business_cost_uses_notebook_error_weights():
    y_reel = [2, 2, 0, 1, 0]
    predictions = [0, 1, 2, 2, 1]

    assert calculer_cout_metier(y_reel, predictions) == 9


def test_candidate_is_promoted_only_for_strictly_lower_business_cost():
    assert candidat_est_meilleur(10, 9)
    assert not candidat_est_meilleur(10, 10)
    assert not candidat_est_meilleur(10, 11)