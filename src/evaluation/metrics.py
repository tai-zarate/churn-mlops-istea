from sklearn.metrics import (
accuracy_score,
f1_score,
precision_score,
recall_score,
roc_auc_score)

def metricas(y, y_probables, umbral: float = 0.5):
  y_pred = (y_probables > umbral)
  return {"accuracy": accuracy_score(y,y_pred),
    "precision": precision_score(y,y_pred),
    "f1 score": f1_score(y,y_pred),
    "recall": recall_score(y,y_pred),
    "roc_auc": roc_auc_score(y,y_pred),
         }
# no se usa accuracy como metrica principal, se prioriza recall porque un falso negtivo implica perder un cliente sin intervenir
