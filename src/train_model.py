"""Entraînement reproductible du modèle de classement des demandeurs d'emploi.

Ce module centralise toutes les étapes du pipeline ML :
"""

from __future__ import annotations

import argparse
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import config as C, data as D, models as M

mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
mlflow.set_experiment(C.MLFLOW_EXPERIMENT_RETRAINING)

SCENARIOS: dict[str, list[str]] = {
    "complet": C.FEATURES_SCENARIO_1,
    "sans_variables_sensibles": C.FEATURES_SCENARIO_2,
    "sans_nationalite_hors_ue": C.FEATURES_SCENARIO_2B,
    "sans_age_nationalite_statut": C.FEATURES_SCENARIO_2C,
    "texte_seul": C.FEATURES_SCENARIO_3,
    "tabulaire_seul": C.FEATURES_SCENARIO_4,
}

MODEL_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "modele__C": loguniform(1e-3, 1e2),
        "modele__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "modele__n_estimators": randint(200, 601),
        "modele__max_depth": [5, 8, 12, 16, None],
        "modele__min_samples_leaf": [1, 2, 4, 8],
        "modele__min_samples_split": [2, 5, 10],
        "modele__max_features": ["sqrt", "log2", 0.5],
        "modele__class_weight": ["balanced", "balanced_subsample"],
    },
    "lightgbm": {
        "modele__n_estimators": randint(100, 501),
        "modele__learning_rate": uniform(0.01, 0.19),
        "modele__num_leaves": randint(15, 80),
        "modele__max_depth": [-1, 4, 6, 8, 12],
        "modele__min_child_samples": randint(10, 80),
        "modele__subsample": uniform(0.7, 0.3),
        "modele__colsample_bytree": uniform(0.7, 0.3),
        "modele__reg_alpha": uniform(0, 1),
        "modele__reg_lambda": uniform(0, 2),
        "modele__class_weight": [None, "balanced"],
    },
    "xgboost": {
        "modele__n_estimators": randint(100, 501),
        "modele__learning_rate": uniform(0.01, 0.19),
        "modele__max_depth": randint(3, 11),
        "modele__min_child_weight": randint(1, 10),
        "modele__subsample": uniform(0.7, 0.3),
        "modele__colsample_bytree": uniform(0.7, 0.3),
        "modele__gamma": uniform(0, 5),
        "modele__reg_alpha": uniform(0, 1),
        "modele__reg_lambda": uniform(0, 2),
    },
}


def prepare_dataframe(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Applique les transformations déterministes avant le split."""
    source = df if df is not None else D.load_data()
    return D.prepare_training_data(source)


def build_scenario_split(
    df: pd.DataFrame,
    scenario_name: str,
    test_size: float = 0.2,
    seed: int = C.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Construit le split train/test sur un scénario donné."""
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Scénario inconnu : {scenario_name}")

    features = SCENARIOS[scenario_name]
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(
            f"Scénario {scenario_name} contient des colonnes manquantes : {missing}"
        )

    X = df[features].copy()
    y = df[C.CIBLE].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(scenario_name: str) -> ColumnTransformer:
    """Construit un préprocesseur ColumTransformer par type de variable."""
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Scénario inconnu : {scenario_name}")

    features = SCENARIOS[scenario_name]
    numeric_cols = [col for col in C.FEATURES_NUM if col in features]
    ordinal_cols = [col for col in C.FEATURES_CAT_ORDINAL if col in features]
    categorical_cols = [col for col in C.FEATURES_CAT if col in features]
    text_cols = [col for col in C.FEATURE_TEXT if col in features]

    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        transformers.append(
            (
                "numeriques",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )

    if ordinal_cols:
        transformers.append(
            (
                "ordinal",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ordinal_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "categorielles",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    if text_cols:
        transformers.append(
            (
                "texte",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                        ("flatten", FunctionTransformer(np.ravel, validate=False)),
                        ("tfidf", TfidfVectorizer()),
                    ]
                ),
                text_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_estimator(model_name: str, seed: int = C.SEED):
    """Construit un estimateur de base par nom de modèle."""
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=6,
            random_state=seed,
            verbose=-1,
        )
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Modèle inconnu : {model_name}")


def build_pipeline(scenario_name: str, model_name: str) -> Pipeline:
    """Construit le pipeline de prétraitement + modèle."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(scenario_name)),
            ("modele", build_estimator(model_name)),
        ]
    )


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scenario_name: str,
    n_iter: int = 10,
    seed: int = C.SEED,
) -> RandomizedSearchCV:
    """Recherche d'hyperparamètres sur l'espace défini par modèle."""
    if model_name not in MODEL_SEARCH_SPACES:
        raise ValueError(f"Modèle non supporté pour l'optimisation : {model_name}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=build_pipeline(scenario_name, model_name),
        param_distributions=MODEL_SEARCH_SPACES[model_name],
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Calcule les métriques d'évaluation standardisées."""
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    metrics_train = {
        "accuracy": accuracy_score(y_train, y_train_pred),
        "f1_macro": f1_score(y_train, y_train_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
    }
    metrics_test = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "f1_macro": f1_score(y_test, y_test_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "confusion_matrix": confusion_matrix(y_test, y_test_pred, labels=[0, 1, 2]),
    }

    return {
        "train": metrics_train,
        "test": metrics_test,
        "y_pred": y_test_pred,
    }


def save_model_artifact(
    pipeline: Pipeline,
    scenario_name: str,
    model_name: str,
    metrics: dict[str, Any],
    artifact_name: str | None = None,
) -> str:
    """Sauvegarde le pipeline et les métadonnées associées."""
    basename = artifact_name or f"{scenario_name}_{model_name}_model"
    metadata = {
        "scenario_name": scenario_name,
        "model_name": model_name,
        "metrics": {
            "f1_macro_test": float(metrics["test"]["f1_macro"]),
            "accuracy_test": float(metrics["test"]["accuracy"]),
            "balanced_accuracy_test": float(metrics["test"]["balanced_accuracy"]),
        },
    }
    return M.sauver(pipeline, basename, metadata=metadata)


def run_training(
    scenario_name: str = "complet",
    model_name: str = "random_forest",
    tune: bool = False,
    save_artifact: bool = True,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Pipeline complet d'entraînement pour un scénario/model donné."""
    df = prepare_dataframe()
    X_train, X_test, y_train, y_test = build_scenario_split(df, scenario_name)

    if tune:
        search = tune_model(model_name, X_train, y_train, scenario_name)
        model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
    else:
        model = build_pipeline(scenario_name, model_name)
        model.fit(X_train, y_train)
        best_params = model.named_steps["modele"].get_params(deep=False)
        best_score = None

    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    result = {
        "scenario_name": scenario_name,
        "model_name": model_name,
        "best_params": best_params,
        "best_score_cv": best_score,
        "metrics": metrics,
    }

    mlflow_run_name = run_name or f"{scenario_name}_{model_name}"
    try:
        with mlflow.start_run(run_name=mlflow_run_name):
            if best_params:
                mlflow.log_params({str(k): v for k, v in best_params.items()})
            if best_score is not None:
                mlflow.log_metric("f1_macro_cv", float(best_score))
            mlflow.log_metric("f1_macro_test", float(metrics["test"]["f1_macro"]))
            mlflow.log_metric("accuracy_test", float(metrics["test"]["accuracy"]))
            mlflow.log_metric(
                "balanced_accuracy_test", float(metrics["test"]["balanced_accuracy"])
            )

            figure, axes = __import__("matplotlib").pyplot.subplots()
            from sklearn.metrics import ConfusionMatrixDisplay

            ConfusionMatrixDisplay.from_predictions(
                y_test,
                metrics["y_pred"],
                ax=axes,
                labels=[0, 1, 2],
            )
            mlflow.log_figure(figure, "matrice_confusion.png")
            __import__("matplotlib").pyplot.close(figure)

            mlflow.sklearn.log_model(
                sk_model=model,
                name="pipeline",
                registered_model_name=f"RetourEmploi_{scenario_name}_{model_name}",
            )
    except (
        Exception
    ) as exc:  # pragma: no cover - robustness for local MLflow compatibility issues
        print(f"MLflow logging skipped for {mlflow_run_name}: {exc}")

    if save_artifact:
        save_model_artifact(model, scenario_name, model_name, metrics)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraînement d'un modèle de tri des demandeurs d'emploi."
    )
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="complet")
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "random_forest", "lightgbm", "xgboost"],
        default="random_forest",
    )
    parser.add_argument(
        "--tune", action="store_true", help="Lance une recherche d'hyperparamètres."
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Ne sauvegarde pas le modèle localement."
    )
    args = parser.parse_args()

    result = run_training(
        scenario_name=args.scenario,
        model_name=args.model,
        tune=args.tune,
        save_artifact=not args.no_save,
    )
    print(f"Scenario : {result['scenario_name']}")
    print(f"Model : {result['model_name']}")
    print(f"F1 macro test : {result['metrics']['test']['f1_macro']:.4f}")
    print(f"Accuracy test : {result['metrics']['test']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
