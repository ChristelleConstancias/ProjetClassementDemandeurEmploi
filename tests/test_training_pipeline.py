import numpy as np
import pandas as pd

from src import config as C
from src.data import prepare_training_data
from src.train_model import (
    build_pipeline,
    build_scenario_split,
    evaluate_model,
)


def make_training_data(row_count: int = 60) -> pd.DataFrame:
    classes = np.arange(row_count) % 3
    return pd.DataFrame(
        {
            "age": 20 + np.arange(row_count) % 35,
            "niveau_diplome": np.array(
                ["Sans diplôme", "Bac", "Bac + 2"]
            )[classes],
            "anciennete_poste_ans": 1 + np.arange(row_count) % 8,
            "code_rome_vise": np.array(["A1203", "M1602", "J1506"])[classes],
            "code_insee_commune": np.array(["75001", "13055", "69002"])[classes],
            "est_allocataire": np.arange(row_count) % 2,
            "nationalite_hors_ue": (np.arange(row_count) // 2) % 2,
            "synthese_entretien": np.array(
                ["projet stable", "besoin formation", "frein transport"]
            )[classes],
            C.CIBLE: classes,
        }
    )


def test_prepare_training_data_derives_features_and_bounds_experience():
    source = make_training_data(3)
    source.loc[0, "anciennete_poste_ans"] = 10

    prepared = prepare_training_data(source)

    assert prepared["departement"].tolist() == ["75", "13", "69"]
    assert prepared["famille_metier"].tolist() == ["A12", "M16", "J15"]
    assert prepared["niveau_diplome_ordinal"].tolist() == [1, 2, 3]
    assert prepared.loc[0, "anciennete_poste_ans"] == 4


def test_random_forest_pipeline_trains_and_evaluates_on_stratified_split():
    prepared = prepare_training_data(make_training_data())
    X_train, X_test, y_train, y_test = build_scenario_split(
        prepared,
        "complet",
    )
    pipeline = build_pipeline("complet", "random_forest")
    pipeline.set_params(modele__n_estimators=8)

    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_train, y_train, X_test, y_test)

    assert len(X_train) == 48
    assert len(X_test) == 12
    assert metrics["test"]["confusion_matrix"].shape == (3, 3)
    assert len(metrics["y_pred"]) == len(y_test)
