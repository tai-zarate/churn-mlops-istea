import os
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "churn_model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "local-dev")
# Umbrales de nivel de riesgo.
# Elegidos sobre la distribución de probabilidades del conjunto de test:
# HIGH concentra 58% de churn real (tasa base 26,4%) y capta 220 de 372 casos
# positivos en el 27% de la cartera, un volumen operable por retención.
UMBRAL_MEDIUM = 0.35
UMBRAL_HIGH = 0.65


def nivel_de_riesgo(probabilidad: float) -> str:
    if probabilidad >= UMBRAL_HIGH:
        return "HIGH"
    if probabilidad >= UMBRAL_MEDIUM:
        return "MEDIUM"
    return "LOW"

_modelo = None


def cargar_modelo():
    global _modelo
    if _modelo is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en {MODEL_PATH}. "
                "Ejecutá: python -m src.training.train"
            )
        _modelo = joblib.load(MODEL_PATH)
    return _modelo
def predecir(datos_cliente: dict) -> dict:
    modelo = cargar_modelo()
    df = pd.DataFrame([datos_cliente])

    probabilidad = float(modelo.predict_proba(df)[0, 1])

    return {
        "prediction": "CHURN" if probabilidad >= 0.5 else "NO_CHURN",
        "probability": round(probabilidad, 4),
        "risk_level": nivel_de_riesgo(probabilidad),
        "model_version": MODEL_VERSION,
    }

if __name__ == "__main__":
    cliente = {
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

    print(predecir(cliente))