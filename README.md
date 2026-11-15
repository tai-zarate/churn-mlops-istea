# Customer Churn — MLOps | ISTEA 2026

Predicción de abandono de clientes en telecomunicaciones, construida como
sistema reproducible y desplegable: del dataset al servicio de inferencia
containerizado.

Proyecto integrador de Laboratorio de Minería de Datos, ISTEA.

---

## Problema de negocio

Identificar clientes con riesgo de abandono para habilitar acciones de retención
antes de la baja.

- **Tipo:** clasificación binaria supervisada.
- **Target:** `Churn` (Yes / No).
- **Salida:** probabilidad de abandono y nivel de riesgo operativo.

## Dataset

7.043 clientes, 21 columnas. Versionado con DVC, remote en DagsHub.
`customerID` se descarta: es identificador, no predictor.

Distribución del target: **73,6% No / 26,4% Sí**.

### Hallazgos del EDA y decisiones de preprocesamiento

**1. Valores faltantes en `TotalCharges`.** 26 nulos, todos en clientes con
`tenure = 0`. Existen además 4 clientes con `tenure = 0` y valor `0.0`
explícito, lo que confirma que el nulo significa "todavía no facturó".
→ Se imputa con constante 0. Usar media o mediana asignaría ~1.400 de historial
a clientes recién ingresados, que son el segmento de mayor churn.

**2. Categorías redundantes.** `"No internet service"` aparece en seis columnas
de servicios y corresponde siempre a los mismos 1.549 clientes con
`InternetService = No`. Con OneHotEncoder generaría columnas idénticas entre sí.
→ Se recodifica como `"No"`. Mismo criterio para `"No phone service"` en
`MultipleLines`.

**3. Desbalance de clases.** Un clasificador trivial que prediga siempre "No"
alcanza 73,6% de accuracy sin aprender nada.
→ Accuracy no se usa como métrica de selección.

## Métrica de decisión y caso de negocio

Métrica principal: **recall sobre la clase positiva**, acompañada de precision,
F1, ROC-AUC y matriz de confusión.

El falso negativo —clasificar como estable a un cliente que estaba por irse—
cuesta el cliente completo más el costo de readquisición. El falso positivo
—ofrecer retención a quien se quedaba— cuesta solo el margen del descuento.
En telecomunicaciones adquirir cuesta sensiblemente más que retener, por lo que
la asimetría justifica priorizar recall aun a costa de precision.

## Comparación de modelos

Validación cruzada estratificada de 5 folds sobre entrenamiento. Seis
configuraciones, cada una respondiendo una pregunta concreta.

| Modelo | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| logistic_regression_balanced | 0,5063 | **0,7677** | 0,6099 | 0,8276 |
| random_forest_balanced_shallow | 0,5109 | 0,7253 | 0,5993 | 0,8177 |
| random_forest_balanced | 0,5377 | 0,6774 | 0,5993 | 0,8133 |
| logistic_regression | 0,6712 | 0,4835 | 0,5619 | 0,8278 |
| random_forest | 0,6652 | 0,4572 | 0,5414 | 0,8138 |
| baseline_dummy | 0,0000 | 0,0000 | 0,0000 | 0,5000 |

### Lectura de los resultados

El baseline confirma el piso: nunca predecir churn da recall 0 y ROC-AUC 0,5.

Compensar el desbalance con `class_weight="balanced"` sube el recall de forma
consistente en ambas familias: de 0,4835 a 0,7677 en el modelo lineal y de
0,4572 a 0,6774 en el de árboles. El efecto no depende del algoritmo.

El ROC-AUC se mantiene prácticamente igual entre variantes balanceadas y no
balanceadas. Eso indica que `class_weight` no mejora la capacidad de
discriminación: desplaza el punto de operación hacia la clase minoritaria. Es
también la razón por la que la accuracy baja mientras el modelo se vuelve más
útil para el caso de negocio.

## Modelo candidato

- **Modelo:** `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)`
- **Registrado como:** `churn-classifier`, versión 1 en MLflow Model Registry
- **Run de origen:** `7279edd0517147c8a200015cde3ed0e8`
- **Criterio de selección:** mayor recall en validación cruzada, con F1 como
  desempate. Coherente con la métrica de decisión declarada arriba.

### Métricas sobre test

| Métrica | Valor |
|---|---|
| Recall | 0,7177 |
| Precision | 0,4785 |
| F1 | 0,5742 |
| ROC-AUC | 0,8117 |
| Accuracy | 0,7189 |

Matriz de confusión:

|  | Predicho: No churn | Predicho: Churn |
|---|---|---|
| **Real: No churn** | 746 | 291 |
| **Real: Churn** | 105 | 267 |

De los 372 clientes del test que efectivamente abandonaron, el modelo detecta
267 y no detecta 105. A cambio señala 291 que no se iban. El intercambio es de
aproximadamente dos acciones de retención innecesarias por cada cliente
adicional detectado, favorable mientras el valor de retener supere el doble del
costo de la acción.

---

## Tracking de experimentos

MLflow con DagsHub como tracking server.

**Proyecto:** https://dagshub.com/tai-zarate/churn-mlops-istea

Cada uno de los seis modelos genera un Run con sus hiperparámetros y métricas de
validación cruzada. El seleccionado genera un Run adicional con las métricas
sobre test, los artefactos de resultados y el registro en Model Registry.

| Elemento | Valor |
|---|---|
| Experimento | `churn-prediction` |
| Modelo registrado | `churn-classifier` v1 |
| Run de origen | `7279edd0517147c8a200015cde3ed0e8` |

La API expone esa versión en `model_version`, de modo que puede determinarse qué
Run, qué código y qué datos originaron cualquier predicción.

---

## API de inferencia

FastAPI con contratos Pydantic. Documentación OpenAPI en `/docs`.

### `GET /health`

```json
{"status": "ok", "model_loaded": true, "model_version": "1"}
```

Distingue dos estados: el servicio responde pero sin modelo cargado
(`degraded`), o responde con el modelo listo (`ok`). Son problemas distintos y
se resuelven distinto.

### `POST /predict`

Entrada: 19 características del cliente. Salida:

```json
{
  "prediction": "CHURN",
  "probability": 0.9145,
  "risk_level": "HIGH",
  "model_version": "1",
  "request_id": "32088a59-55d8-48ae-b582-4e5e9144ed00"
}
```

### Niveles de riesgo

`LOW < 0,35 ≤ MEDIUM < 0,65 ≤ HIGH`

Umbrales definidos sobre la distribución de probabilidades del conjunto de test:

| Banda | Clientes | Churn real | Positivos captados |
|---|---|---|---|
| LOW | 666 (47%) | 8,3% | 55 / 372 |
| MEDIUM | 364 (26%) | 26,6% | 97 / 372 |
| HIGH | 379 (27%) | 58,0% | 220 / 372 |

HIGH concentra más del doble de la tasa base de churn (26,4%) y capta el 59% de
los casos positivos en el 27% de la cartera, un volumen operable para acciones
de retención de alto costo. MEDIUM queda para acciones automatizadas de bajo
costo.

### Validación de entrada

Los campos categóricos usan `Literal` en lugar de `str`. Con `str`, un valor no
visto en entrenamiento pasaría la validación y el `OneHotEncoder` lo ignoraría
silenciosamente por `handle_unknown="ignore"`, devolviendo una predicción basada
en datos incompletos. Con `Literal` el request se rechaza con 422 y el motivo
explícito.

`TotalCharges` es opcional: el nulo representa clientes con `tenure = 0` que aún
no facturaron. Exigirlo impediría scorear al segmento de mayor riesgo.

---

## Tests

Suite de 17 tests con pytest.

```bash
python -m pytest -q
```

Cubre los seis casos requeridos —carga del modelo, predicción válida, estructura
y tipos de la respuesta, `/health`, `/predict` y entradas inválidas— más los
bordes exactos de los umbrales de riesgo, el comportamiento con el modelo
ausente, la unicidad del `request_id` y la disponibilidad de Swagger.

Los tests entrenan un modelo mínimo en memoria en lugar de depender de
`models/churn_model.joblib`, que no está versionado en Git. Eso permite
ejecutarlos en un clon limpio sin entrenar primero.

---

## Docker

### Build y ejecución

```bash
docker compose up -d --build
docker compose ps        # debe mostrar (healthy)
```

La API queda en `http://localhost:8000`, con Swagger en `/docs`.

Para detener:

```bash
docker compose down
```

### Decisiones

**Dependencias separadas.** La imagen instala `requirements-api.txt`, que no
incluye MLflow, DVC, matplotlib ni pytest. Un servicio de inferencia no entrena:
reduce el tamaño de la imagen y acelera el build.

**El modelo entra como artefacto.** Se copia `models/churn_model.joblib`,
correspondiente a la versión indicada por `MODEL_VERSION`. No se entrena dentro
del build: eso haría la imagen no reproducible, ya que cada build generaría un
modelo distinto y se perdería la trazabilidad al Run de origen.

**Sin secretos en la imagen.** `.env` está en `.dockerignore`. Compose lo usa
solo para resolver `MODEL_VERSION`; al contenedor llega únicamente lo declarado
en `environment`.

**Healthcheck** sobre `GET /health`, cada 30 segundos.

### Variables de entorno

| Variable | Descripción | Default |
|---|---|---|
| `MODEL_PATH` | Ruta al pipeline serializado | `models/churn_model.joblib` |
| `MODEL_VERSION` | Versión registrada en Model Registry | `local-dev` |
| `MLFLOW_TRACKING_URI` | Tracking server | `sqlite:///mlflow.db` |
| `MLFLOW_TRACKING_USERNAME` | Usuario de DagsHub | — |
| `MLFLOW_TRACKING_PASSWORD` | Token de DagsHub | — |

Ver `.env.example`.

---

## Estructura del proyecto

```
churn-mlops-istea/
├── app/
│   ├── main.py            endpoints /health y /predict
│   └── schemas.py         contratos Pydantic
├── data/raw/              datos versionados con DVC
├── models/                artefacto serializado (no versionado en Git)
├── notebooks/             EDA y análisis por segmentos
├── reports/               resultados del entrenamiento
├── src/
│   ├── data/              carga y partición
│   ├── features/          pipeline de preprocesamiento
│   ├── training/          entrenamiento, selección y tracking
│   ├── evaluation/        métricas
│   └── inference/         carga del modelo y predicción
├── tests/                 suite de pytest
├── Dockerfile
├── compose.yaml
├── requirements.txt       entorno de desarrollo y entrenamiento
└── requirements-api.txt   dependencias del servicio
```

---

## Instalación y uso

```bash
git clone https://github.com/tai-zarate/churn-mlops-istea.git
cd churn-mlops-istea

python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # Linux / macOS

pip install -r requirements.txt
cp .env.example .env                # completar con token propio
```

Recuperar los datos. DagsHub requiere autenticación incluso en repositorios
públicos, así que antes hay que configurar el acceso (queda en
`.dvc/config.local`, que no se versiona):

```bash
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user tai-zarate
dvc remote modify dagshub --local password ddd828aa291ab93af98186a495ec7ea9386a6975

dvc pull
```

Entrenar:

```bash
python -m src.training.train
```

Genera `models/churn_model.joblib`, `reports/model_comparison_cv.csv` y
`reports/final_test_metrics.json`, y registra los Runs en MLflow.

Levantar la API localmente:

```bash
uvicorn app.main:app --reload
```

O con Docker:

```bash
docker compose up -d --build
```

---

## Reproducibilidad

- **Seed:** 42, definida en un único lugar y reutilizada en split, modelos y CV.
- **Partición:** 80/20 estratificada. Verificado: 26,36% de churn en train y
  26,40% en test.
- **Selección:** validación cruzada de 5 folds sobre train. El conjunto de test
  no interviene en la selección.
- **Preprocesamiento:** encapsulado en un `Pipeline` de scikit-learn, ajustado
  solo con train. El artefacto serializado contiene preprocesamiento y
  estimador, de modo que entrenamiento e inferencia aplican las mismas
  transformaciones.
- **Dependencias:** versiones fijadas, coincidentes con las que serializaron el
  modelo.
- **Trazabilidad:** commit → Run de MLflow → versión del Model Registry →
  `model_version` que expone la API.

---

## Limitaciones conocidas

- La precision del modelo (0,48) es baja. Es consecuencia deliberada de
  priorizar recall, no un defecto a corregir sin revisar antes el criterio de
  negocio.
- El threshold de clasificación está fijo en 0,50. Ajustarlo es el mecanismo más
  directo para mover el punto de operación y queda pendiente de análisis.
- El modelo entra a la imagen como artefacto copiado. Una versión posterior
  podría descargarlo del Model Registry en tiempo de build.
- No hay aún pipeline de CI/CD ni publicación de la imagen en un registro.