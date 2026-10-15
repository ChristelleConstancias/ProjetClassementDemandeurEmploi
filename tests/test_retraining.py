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
    current = {
        "cout_metier": 10,
        "erreurs_2_vers_0": 2,
        "recall_classe_2": 0.7,
        "f1_macro": 0.6,
    }
    assert candidat_est_meilleur(current, {**current, "cout_metier": 9})
    assert not candidat_est_meilleur(current, {**current, "cout_metier": 11})
    assert not candidat_est_meilleur(current, current)


def test_candidate_tie_break_matches_notebook_selection_order():
    current = {
        "cout_metier": 10,
        "erreurs_2_vers_0": 2,
        "recall_classe_2": 0.7,
        "f1_macro": 0.6,
    }
    fewer_critical_errors = {**current, "erreurs_2_vers_0": 1}
    higher_recall = {**current, "recall_classe_2": 0.8}
    higher_f1 = {**current, "f1_macro": 0.7}

    assert candidat_est_meilleur(current, fewer_critical_errors)
    assert candidat_est_meilleur(current, higher_recall)
    assert candidat_est_meilleur(current, higher_f1)
    assert not candidat_est_meilleur(
        current,
        {**current, "erreurs_2_vers_0": 3, "recall_classe_2": 0.9},
    )