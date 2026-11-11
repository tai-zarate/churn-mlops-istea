# Fixtures compartidas por la suite.
#
# Los tests no usan models/churn_model.joblib: ese archivo no esta en el repo y
# en un clon limpio no existe. En su lugar entrenamos un modelo chico al vuelo
# con el mismo Pipeline de produccion, asi la suite corre en cualquier maquina
# y en CI sin necesidad de dvc pull ni de un entrenamiento previo.

import json
import random
from pathlib import Path

import pandas as pd
import pytest
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocessing import construir_preprocesador

RAIZ = Path(__file__).resolve().parents[1]

# Valores posibles de cada campo, tomados del contrato de app/schemas.py.
CATEGORIAS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def _datos_sinteticos(n=200, semilla=42):
    # Dataset minimo que cubre todas las categorias del contrato. No busca
    # parecerse al real: solo tiene que permitir que el Pipeline se ajuste.
    rnd = random.Random(semilla)
    filas = []
    for i in range(n):
        fila = {campo: valores[i % len(valores)] for campo, valores in CATEGORIAS.items()}
        fila["tenure"] = rnd.randint(0, 72)
        fila["MonthlyCharges"] = round(rnd.uniform(18.0, 120.0), 2)
        fila["TotalCharges"] = round(fila["tenure"] * fila["MonthlyCharges"], 2)
        filas.append(fila)
    X = pd.DataFrame(filas)
    # Etiqueta con señal: contrato mensual y poca antiguedad tiran a churn.
    y = ((X.Contract == "Month-to-month") & (X.tenure < 36)).astype(int)
    return X, y


@pytest.fixture(scope="session")
def ruta_modelo(tmp_path_factory):
    # Entrena y serializa un pipeline completo (preprocesamiento + estimador),
    # igual que hace src/training/train.py pero sin validacion cruzada.
    X, y = _datos_sinteticos()
    pipeline = Pipeline([
        ("preprocessor", construir_preprocesador()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X, y)

    ruta = tmp_path_factory.mktemp("modelo") / "churn_model.joblib"
    dump(pipeline, ruta)
    return ruta


@pytest.fixture(autouse=True)
def _apuntar_al_modelo_de_prueba(ruta_modelo, monkeypatch):
    # predictor.py lee MODEL_PATH al importarse, asi que no alcanza con tocar
    # la variable de entorno: hay que reemplazar el atributo del modulo y
    # limpiar la cache del modelo ya cargado.
    from src.inference import predictor

    monkeypatch.setattr(predictor, "MODEL_PATH", ruta_modelo)
    monkeypatch.setattr(predictor, "MODEL_VERSION", "test-1")
    monkeypatch.setattr(predictor, "_modelo", None)
    yield
    predictor._modelo = None


@pytest.fixture
def cliente_api():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture
def cliente_valido():
    # Payload de ejemplo del enunciado (examples/predict_request_valid.json).
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 32,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 101.35,
        "TotalCharges": 2865.88,
    }
