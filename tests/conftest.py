import pandas as pd
import pytest
from fastapi.testclient import TestClient
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.main import app
from src.features.preprocessing import construir_preprocesador
from src.inference import predictor

CLIENTE = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 32, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check", "MonthlyCharges": 101.35, "TotalCharges": 2865.88,
}


@pytest.fixture(scope="session")
def ruta_modelo(tmp_path_factory):
    # Modelo minimo entrenado con variaciones del cliente de ejemplo, para que
    # los tests no dependan de models/churn_model.joblib (no esta en el repo).
    X = pd.DataFrame([dict(CLIENTE, tenure=t, Contract=c)
                      for t in (1, 12, 40, 70)
                      for c in ("Month-to-month", "Two year")])
    y = (X.Contract == "Month-to-month").astype(int)
    modelo = Pipeline([("preprocessor", construir_preprocesador()),
                       ("model", LogisticRegression())]).fit(X, y)
    ruta = tmp_path_factory.mktemp("modelo") / "churn_model.joblib"
    dump(modelo, ruta)
    return ruta


@pytest.fixture(autouse=True)
def modelo_de_prueba(ruta_modelo, monkeypatch):
    monkeypatch.setattr(predictor, "MODEL_PATH", ruta_modelo)
    monkeypatch.setattr(predictor, "_modelo", None)


@pytest.fixture
def api():
    return TestClient(app)


@pytest.fixture
def cliente():
    return dict(CLIENTE)
