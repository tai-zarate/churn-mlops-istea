import pytest

from src.inference import predictor
from src.inference.predictor import cargar_modelo, nivel_de_riesgo, predecir


def test_carga_del_modelo():
    assert "preprocessor" in cargar_modelo().named_steps


def test_modelo_ausente(monkeypatch, tmp_path):
    monkeypatch.setattr(predictor, "MODEL_PATH", tmp_path / "no-existe.joblib")
    with pytest.raises(FileNotFoundError):
        cargar_modelo()


@pytest.mark.parametrize("prob, nivel", [(0.34, "LOW"), (0.35, "MEDIUM"),
                                         (0.64, "MEDIUM"), (0.65, "HIGH")])
def test_niveles_de_riesgo(prob, nivel):
    assert nivel_de_riesgo(prob) == nivel


def test_prediccion_valida(cliente):
    r = predecir(cliente)
    assert r["prediction"] in ("CHURN", "NO_CHURN")
    assert 0 <= r["probability"] <= 1
    assert r["risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_total_charges_nulo(cliente):
    # clientes con tenure 0 todavia no facturaron: el contrato lo permite
    r = predecir(dict(cliente, tenure=0, TotalCharges=None))
    assert 0 <= r["probability"] <= 1
