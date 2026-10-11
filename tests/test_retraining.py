from src.api import candidat_est_meilleur, calculer_cout_metier


def test_business_cost_uses_notebook_error_weights():
    y_reel = [2, 2, 0, 1, 0]
    predictions = [0, 1, 2, 2, 1]

    assert calculer_cout_metier(y_reel, predictions) == 9


def test_candidate_is_promoted_only_for_strictly_lower_business_cost():
    assert candidat_est_meilleur(10, 9)
    assert not candidat_est_meilleur(10, 10)
    assert not candidat_est_meilleur(10, 11)