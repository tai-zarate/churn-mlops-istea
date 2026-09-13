import pandas as pd
from pathlib import Path

RUTA_DATOS = Path(__file__).resolve().parents[2] / "data" / "raw" / "customer_churn_historical.csv"

def cargar_datos():
    df = pd.read_csv(RUTA_DATOS)
    return df

def separar_variables(df):
    df = df.drop(columns=["customerID"])
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])
    return X, y

if __name__ == "__main__":
    df = cargar_datos()
    X, y = separar_variables(df)
    print("Filas y columnas de X:", X.shape)
    print("Distribución de y:")
    print(y.value_counts(normalize=True))