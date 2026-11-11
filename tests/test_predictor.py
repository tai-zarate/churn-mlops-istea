# Logica critica de inferencia, sin pasar por HTTP.

import pytest

from src.inference import predictor
from src.inference.predictor import cargar_modelo, nivel_de_riesgo, predecir


def test_carga_del_modelo():
    modelo = cargar_modelo()
    assert hasattr(modelo, "predict_proba")
    # El artefacto serializado tiene que traer el preprocesamiento adentro,
    # para que entrenamiento e inferencia apliquen las mismas transformaciones.
    assert "preprocessor" in modelo.named_steps


def test_el_modelo_se_carga_una_sola_vez():
    primero = cargar_modelo()
    segundo = cargar_modelo()
    assert primero is segundo


def test_modelo_ausente_avisa(monkeypatch, tmp_path):
    monkeypatch.setattr(predictor, "MODEL_PATH", tmp_path / "no-existe.joblib")
    monkeypatch.setattr(predictor, "_modelo", None)
    with pytest.raises(FileNotFoundError):
        cargar_modelo()


@pytest.mark.parametrize(
    "probabilidad, esperado",
    [
        (0.00, "LOW"),
        (0.34, "LOW"),
        (0.35, "MEDIUM"),   # el limite inferior entra en MEDIUM
        (0.64, "MEDIUM"),
        (0.65, "HIGH"),     # el limite inferior entra en HIGH
        (1.00, "HIGH"),
    ],
)
def test_niveles_de_riesgo(probabilidad, esperado):
    assert nivel_de_riesgo(probabilidad) == esperado


def test_prediccion_valida(cliente_valido):
    resultado = predecir(cliente_valido)

    assert resultado["prediction"] in ("CHURN", "NO_CHURN")
    assert 0.0 <= resultado["probability"] <= 1.0
    assert resultado["risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_la_prediccion_es_coherente_con_la_probabilidad(cliente_valido):
    resultado = predecir(cliente_valido)
    esperado = "CHURN" if resultado["probability"] >= 0.5 else "NO_CHURN"
    assert resultado["prediction"] == esperado
    assert resultado["risk_level"] == nivel_de_riesgo(resultado["probability"])


def test_total_charges_nulo(cliente_valido):
    # El contrato declara TotalCharges opcional para clientes con tenure = 0,
    # que todavia no facturaron. El pipeline tiene que imputarlo, no romperse.
    cliente = dict(cliente_valido, tenure=0, TotalCharges=None)
    resultado = predecir(cliente)
    assert 0.0 <= resultado["probability"] <= 1.0
