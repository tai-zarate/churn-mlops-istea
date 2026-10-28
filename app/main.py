import uuid

from fastapi import FastAPI, HTTPException

from app.schemas import ClienteRequest, HealthResponse, PrediccionResponse
from src.inference.predictor import MODEL_VERSION, cargar_modelo, predecir

app = FastAPI(
    title="Customer Churn API",
    description="Servicio de inferencia para predicción de abandono de clientes.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        cargar_modelo()
        modelo_cargado = True
    except FileNotFoundError:
        modelo_cargado = False

    return HealthResponse(
        status="ok" if modelo_cargado else "degraded",
        model_loaded=modelo_cargado,
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PrediccionResponse)
def predict(cliente: ClienteRequest):
    try:
        resultado = predecir(cliente.model_dump())
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))

    return PrediccionResponse(**resultado, request_id=str(uuid.uuid4()))