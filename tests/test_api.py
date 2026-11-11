# Contrato HTTP: endpoints, forma de la respuesta y manejo de errores.

import pytest


def test_health(cliente_api):
    respuesta = cliente_api.get("/health")
    assert respuesta.status_code == 200

    cuerpo = respuesta.json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["model_loaded"] is True
    assert cuerpo["model_version"] == "test-1"


def test_predict_devuelve_200(cliente_api, cliente_valido):
    assert cliente_api.post("/predict", json=cliente_valido).status_code == 200


def test_estructura_y_tipos_de_la_respuesta(cliente_api, cliente_valido):
    cuerpo = cliente_api.post("/predict", json=cliente_valido).json()

    assert set(cuerpo) == {
        "prediction",
        "probability",
        "risk_level",
        "model_version",
        "request_id",
    }
    assert isinstance(cuerpo["prediction"], str)
    assert isinstance(cuerpo["probability"], float)
    assert isinstance(cuerpo["risk_level"], str)
    assert isinstance(cuerpo["model_version"], str)
    assert isinstance(cuerpo["request_id"], str)


def test_cada_request_trae_su_propio_id(cliente_api, cliente_valido):
    primero = cliente_api.post("/predict", json=cliente_valido).json()["request_id"]
    segundo = cliente_api.post("/predict", json=cliente_valido).json()["request_id"]
    assert primero != segundo


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("tenure", -12),                    # fuera de rango
        ("Contract", "Weekly"),             # categoria inexistente
        ("MonthlyCharges", "not-a-number"), # tipo equivocado
        ("gender", "Otro"),
        ("SeniorCitizen", 2),
    ],
)
def test_entradas_invalidas_dan_422(cliente_api, cliente_valido, campo, valor):
    respuesta = cliente_api.post("/predict", json=dict(cliente_valido, **{campo: valor}))
    assert respuesta.status_code == 422
    assert any(campo in error["loc"] for error in respuesta.json()["detail"])


def test_campo_faltante_da_422(cliente_api, cliente_valido):
    cliente_valido.pop("tenure")
    assert cliente_api.post("/predict", json=cliente_valido).status_code == 422


def test_payload_invalido_del_enunciado(cliente_api, cliente_valido):
    # examples/predict_request_invalid.json trae tres errores a proposito.
    # La API tiene que reportarlos juntos, no cortar en el primero.
    invalido = dict(
        cliente_valido,
        tenure=-12,
        Contract="Weekly",
        MonthlyCharges="not-a-number",
    )
    respuesta = cliente_api.post("/predict", json=invalido)

    assert respuesta.status_code == 422
    campos = {error["loc"][-1] for error in respuesta.json()["detail"]}
    assert campos == {"tenure", "Contract", "MonthlyCharges"}


def test_swagger_disponible(cliente_api):
    assert cliente_api.get("/docs").status_code == 200

    esquema = cliente_api.get("/openapi.json")
    assert esquema.status_code == 200
    assert "/predict" in esquema.json()["paths"]
