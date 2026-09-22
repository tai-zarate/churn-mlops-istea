from __future__ import annotations

 

import json

from pathlib import Path

 

import pandas as pd

from joblib import dump

from sklearn.dummy import DummyClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (

    accuracy_score,

    confusion_matrix,

    f1_score,

    precision_score,

    recall_score,

    roc_auc_score,

)

from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.pipeline import Pipeline

 

from src.data.load import cargar_datos, separar_variables

from src.data.split import dividir_datos

from src.features.preprocessor import construir_preprocesador

 

 

ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = ROOT / "reports"

MODELS_DIR = ROOT / "models"

 

SCORING = {

    "precision": "precision",

    "recall": "recall",

    "f1": "f1",

    "roc_auc": "roc_auc",

}

 

 

def crear_pipeline(modelo):

    return Pipeline(

        steps=[

            ("preprocessor", construir_preprocesador()),

            ("model", modelo),

        ]

    )

 

 

def evaluar_modelo(y_true, probabilities):

    predictions = (probabilities >= 0.5).astype(int)

 

    return {

        "accuracy": round(accuracy_score(y_true, predictions), 4),

        "precision": round(precision_score(y_true, predictions, zero_division=0), 4),

        "recall": round(recall_score(y_true, predictions, zero_division=0), 4),

        "f1": round(f1_score(y_true, predictions, zero_division=0), 4),

        "roc_auc": round(roc_auc_score(y_true, probabilities), 4),

        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),

    }

 

 

def main():

    REPORTS_DIR.mkdir(exist_ok=True)

    MODELS_DIR.mkdir(exist_ok=True)

 

    df = cargar_datos()

    X, y = separar_variables(df)

    X_train, X_test, y_train, y_test = dividir_datos(X, y)

 

    modelos = {

        "baseline_dummy": DummyClassifier(

            strategy="most_frequent",

            random_state=42,

        ),

        "logistic_regression": LogisticRegression(

            max_iter=1000,

            random_state=42,

        ),

        "random_forest": RandomForestClassifier(

            n_estimators=300,

            min_samples_leaf=3,

            random_state=42,

            n_jobs=-1,

        ),

    }

 

    cv = StratifiedKFold(

        n_splits=5,

        shuffle=True,

        random_state=42,

    )

 

    resultados = []

 

    for nombre, modelo in modelos.items():

        pipeline = crear_pipeline(modelo)

 

        scores = cross_validate(

            pipeline,

            X_train,

            y_train,

            cv=cv,

            scoring=SCORING,

            n_jobs=-1,

        )

 

        resultados.append(

            {

                "modelo": nombre,

                "cv_precision": round(scores["test_precision"].mean(), 4),

                "cv_recall": round(scores["test_recall"].mean(), 4),

                "cv_f1": round(scores["test_f1"].mean(), 4),

                "cv_roc_auc": round(scores["test_roc_auc"].mean(), 4),

            }

        )

 

    resultados_df = pd.DataFrame(resultados).sort_values(

        by=["cv_recall", "cv_f1"],

        ascending=False,

    )

 

    resultados_df.to_csv(

        REPORTS_DIR / "model_comparison_cv.csv",

        index=False,

    )

 

    print("\nResultados de validación cruzada:")

    print(resultados_df.to_string(index=False))

 

    # Se prioriza recall: detectar churn evita falsos negativos.

    nombre_ganador = resultados_df.iloc[0]["modelo"]

    modelo_ganador = modelos[nombre_ganador]

 

    pipeline_final = crear_pipeline(modelo_ganador)

    pipeline_final.fit(X_train, y_train)

 

    probabilities = pipeline_final.predict_proba(X_test)[:, 1]

    metricas_test = evaluar_modelo(y_test, probabilities)

    metricas_test["modelo_seleccionado"] = nombre_ganador

 

    with open(REPORTS_DIR / "final_test_metrics.json", "w", encoding="utf-8") as file:

        json.dump(metricas_test, file, indent=4)

 

    dump(pipeline_final, MODELS_DIR / "churn_model.joblib")

 

    print("\nModelo seleccionado:", nombre_ganador)

    print("Métricas finales sobre test:")

    for metrica, valor in metricas_test.items():

        print(f"{metrica}: {valor}")

 

 

if __name__ == "__main__":

    main()

 
