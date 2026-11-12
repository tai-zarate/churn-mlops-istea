import pytest


def test_health(api):
    r = api.get("/health")
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["model_loaded"] is True
    assert cuerpo["model_version"]


def test_predict(api, cliente):
    r = api.post("/predict", json=cliente)
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) == {"prediction", "probability", "risk_level",
                           "model_version", "request_id"}
    assert isinstance(cuerpo["probability"], float)


def test_request_id_distinto_en_cada_llamada(api, cliente):
    ids = {api.post("/predict", json=cliente).json()["request_id"] for _ in range(2)}
    assert len(ids) == 2


@pytest.mark.parametrize("campo, valor", [("tenure", -12), ("Contract", "Weekly"),
                                          ("MonthlyCharges", "abc"), ("SeniorCitizen", 2)])
def test_entrada_invalida(api, cliente, campo, valor):
    r = api.post("/predict", json=dict(cliente, **{campo: valor}))
    assert r.status_code == 422
    assert campo in r.json()["detail"][0]["loc"]


def test_campo_faltante(api, cliente):
    del cliente["tenure"]
    assert api.post("/predict", json=cliente).status_code == 422


def test_swagger(api):
    assert api.get("/docs").status_code == 200
    assert "/predict" in api.get("/openapi.json").json()["paths"]
