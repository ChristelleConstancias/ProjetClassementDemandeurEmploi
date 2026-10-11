import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import api

# Test pour vérifier que l'explication échoue si le modèle est LogisticRegression avant que SHAP ne soit appelé
def test_explain_rejects_logistic_regression_before_shap_is_called(monkeypatch):
    monkeypatch.setattr(
        api,
        "pipeline",
        Pipeline(
            [
                ("preprocessor", None),
                ("model", LogisticRegression()),
            ]
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "shap",
        SimpleNamespace(
            TreeExplainer=lambda *_args: pytest.fail(
                "TreeExplainer ne doit pas être appelé"
            )
        ),
    )

    demandeur = api.DemandeurEmploi(
        usager_id="ID-TEST",
        conseiller_id="CONSEILLER-TEST",
        age=30,
        niveau_diplome="Bac",
        anciennete_poste_ans=2,
        code_rome_vise="A1203",
        code_insee_commune="75001",
        est_allocataire=1,
        nationalite_hors_ue=0,
        synthese_entretien="projet stable",
    )

    with pytest.raises(HTTPException) as error:
        api.explain(demandeur)


    assert error.value.status_code == 422
    assert "LogisticRegression" in error.value.detail