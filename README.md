
## Instalación y reproducción

```bash
git clone https://github.com/tai-zarate/churn-mlops-istea.git
cd churn-mlops-istea

python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # Linux / macOS

pip install -r requirements.txt
```

Recuperar los datos desde el remote de DVC:

```bash
dvc pull
```

Ejecutar el entrenamiento completo:

```bash
python -m src.training.train
```

Genera:

- `models/churn_model.joblib` — pipeline completo (preprocesamiento + estimador)
- `reports/model_comparison_cv.csv` — tabla comparativa de los seis modelos
- `reports/final_test_metrics.json` — métricas del modelo seleccionado sobre test

## Reproducibilidad

- **Seed:** 42, definida en un único lugar y reutilizada en split, modelos y CV.
- **Partición:** 80/20, estratificada por la variable objetivo. Verificado:
  26,36% de churn en train y 26,40% en test.
- **Selección:** validación cruzada estratificada de 5 folds sobre train. El
  conjunto de test no interviene en la selección del modelo.
- **Preprocesamiento:** encapsulado en un `Pipeline` de scikit-learn, ajustado
  únicamente con train. El artefacto serializado contiene preprocesamiento y
  estimador, de modo que entrenamiento e inferencia aplican las mismas
  transformaciones.
- **Dependencias:** versiones fijadas en `requirements.txt`.

## Limitaciones conocidas

- La precision del modelo seleccionado es baja (0,48). Es una consecuencia
  deliberada de priorizar recall, no un defecto a corregir sin revisar antes el
  criterio de negocio.
- El threshold de clasificación está fijo en 0,50. Ajustarlo es el mecanismo más
  directo para mover el punto de operación y queda pendiente de análisis.
- Los experimentos todavía no están registrados en MLflow.