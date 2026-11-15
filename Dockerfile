FROM python:3.12-slim

WORKDIR /app

# Las dependencias van primero: Docker cachea esta capa y no vuelve a
# instalarlas mientras requirements-api.txt no cambie.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Código de la aplicación.
COPY src/ ./src/
COPY app/ ./app/

# Artefacto del modelo. Corresponde a la versión registrada en MLflow
# Model Registry indicada por MODEL_VERSION.
COPY models/churn_model.joblib ./models/churn_model.joblib

ENV MODEL_PATH=/app/models/churn_model.joblib
ENV MODEL_VERSION=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]