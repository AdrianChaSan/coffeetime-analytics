"""Entrenamiento reproducible de los modelos de CoffeeTime Café.

Entrena y exporta:
- model.joblib: GradientBoostingClassifier para predecir `return_next_week`
- reg_model.joblib: RandomForestRegressor para estimar `avg_spent` (USD)
- model_metadata.json: métricas de validación cruzada, importancia de variables
  y estadísticas de referencia (sin datos personales ni registros crudos).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "coffee_sales_clean.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

FEATURES = [
    "age",
    "visit_frequency",
    "coffee_type_preferred",
    "time_of_visit",
    "satisfaction_score",
    "total_monthly_spent",
    "store_location",
    "membership_tier",
    "promo_used",
    "avg_wait_time_min",
    "days_since_last_visit",
]

CATEGORICAL_COLS = [
    "coffee_type_preferred",
    "time_of_visit",
    "store_location",
    "membership_tier",
]

NUMERICAL_COLS = [
    "age",
    "visit_frequency",
    "satisfaction_score",
    "total_monthly_spent",
    "promo_used",
    "avg_wait_time_min",
    "days_since_last_visit",
]

FEATURE_LABELS = {
    "age": "Edad",
    "visit_frequency": "Frecuencia de visitas",
    "satisfaction_score": "Satisfacción",
    "total_monthly_spent": "Gasto mensual total (USD)",
    "promo_used": "Uso de promoción",
    "avg_wait_time_min": "Tiempo de espera (min)",
    "days_since_last_visit": "Días desde última visita",
    "coffee_type_preferred_Cold Brew": "Café: Cold Brew",
    "coffee_type_preferred_Espresso": "Café: Espresso",
    "coffee_type_preferred_Latte": "Café: Latte",
    "time_of_visit_Noche": "Horario: Noche",
    "time_of_visit_Tarde": "Horario: Tarde",
    "store_location_Centro": "Local: Centro",
    "store_location_Mall": "Local: Mall",
    "store_location_Residencial": "Local: Residencial",
    "membership_tier_Platinum": "Membresía: Platinum",
    "membership_tier_Regular": "Membresía: Regular",
    "membership_tier_Silver": "Membresía: Silver",
}


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERICAL_COLS),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_COLS,
            ),
        ]
    )


def extract_top_importances(pipeline: Pipeline, step_name: str, top_n: int = 8) -> list[dict]:
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
    all_names = NUMERICAL_COLS + cat_names
    importances = pipeline.named_steps[step_name].feature_importances_
    pairs = sorted(zip(all_names, importances), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        {
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "importance": round(float(imp), 4),
        }
        for name, imp in pairs
    ]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {DATA_PATH}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y_clf = df["return_next_week"]
    y_reg = df["avg_spent"]

    # 1. Modelo de clasificación (Gradient Boosting)
    clf_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_f1_folds = cross_val_score(clf_pipeline, X, y_clf, cv=skf, scoring="f1")
    y_clf_oof = cross_val_predict(clf_pipeline, X, y_clf, cv=skf)
    cm = confusion_matrix(y_clf, y_clf_oof)

    clf_pipeline.fit(X, y_clf)
    joblib.dump(clf_pipeline, ARTIFACTS_DIR / "model.joblib", compress=3)

    # 2. Modelo de regresión (Random Forest Regressor)
    reg_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_reg_oof = cross_val_predict(reg_pipeline, X, y_reg, cv=kf)
    reg_rmse = float(root_mean_squared_error(y_reg, y_reg_oof))
    reg_mae = float(mean_absolute_error(y_reg, y_reg_oof))
    reg_r2 = float(r2_score(y_reg, y_reg_oof))

    reg_pipeline.fit(X, y_reg)
    joblib.dump(reg_pipeline, ARTIFACTS_DIR / "reg_model.joblib", compress=3)

    # 3. Metadatos y estadísticas agregadas para la app
    numeric_summary = {}
    for col in NUMERICAL_COLS + ["avg_spent"]:
        series = df[col]
        numeric_summary[col] = {
            "min": round(float(series.min()), 2),
            "median": round(float(series.median()), 2),
            "mean": round(float(series.mean()), 2),
            "max": round(float(series.max()), 2),
        }

    metadata = {
        "dataset": {
            "n_samples": int(len(df)),
            "return_rate": round(float(y_clf.mean()), 4),
            "avg_spent_mean": round(float(y_reg.mean()), 2),
            "avg_spent_median": round(float(y_reg.median()), 2),
            "numeric_summary": numeric_summary,
            "categories": {col: sorted(df[col].unique().tolist()) for col in CATEGORICAL_COLS},
        },
        "classification": {
            "model_name": "GradientBoostingClassifier",
            "target": "return_next_week",
            "cv_folds": 5,
            "f1_cv_mean": round(float(clf_f1_folds.mean()), 4),
            "f1_oof": round(float(f1_score(y_clf, y_clf_oof)), 4),
            "accuracy_oof": round(float(accuracy_score(y_clf, y_clf_oof)), 4),
            "precision_pos_oof": round(float(precision_score(y_clf, y_clf_oof)), 4),
            "recall_pos_oof": round(float(recall_score(y_clf, y_clf_oof)), 4),
            "confusion_matrix": cm.tolist(),
            "top_features": extract_top_importances(clf_pipeline, "classifier", top_n=8),
        },
        "regression": {
            "model_name": "RandomForestRegressor",
            "target": "avg_spent",
            "cv_folds": 5,
            "rmse_oof": round(reg_rmse, 4),
            "mae_oof": round(reg_mae, 4),
            "r2_oof": round(reg_r2, 4),
            "top_features": extract_top_importances(reg_pipeline, "regressor", top_n=8),
        },
        "versions": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }

    with open(ARTIFACTS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Artefactos generados correctamente en artifacts/:")
    print(f"  - artifacts/model.joblib      (F1 CV: {metadata['classification']['f1_cv_mean']:.4f}, Acc: {metadata['classification']['accuracy_oof']:.4f})")
    print(f"  - artifacts/reg_model.joblib  (RMSE OOF: {metadata['regression']['rmse_oof']:.4f} USD, R2: {metadata['regression']['r2_oof']:.4f})")
    print("  - artifacts/model_metadata.json")


if __name__ == "__main__":
    main()
