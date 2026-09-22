#3 Preprocesamiento de datos

# Se definen explicitamente para evitar inferencias erroneas en la API con un solo registro
COL_NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]

COL_CATEGORICAS = [ "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod" ]



# Columnas donde sin internet equivale a No para evitar redundancias
SERVICIOS_INTERNET = [ "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies" ]

import pandas as pd
def unificar_categorias(df: pd.DataFrame):
    # normaliza respuestas negativas antes del encoder para no duplicar columnas
    df = df.copy()
    df[SERVICIOS_INTERNET] = df[SERVICIOS_INTERNET].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    return df

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

def construir_preprocesador():
    num_transformer = Pipeline([
        # Imputa 0 en TotalCharges porque tenure = 0 indica clientes nuevos sin facturacion
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()) ])

    cat_transformer = OneHotEncoder(
        handle_unknown="ignore",
        drop="if_binary",  # Descarta una categoria en las binarias para no generar 13 pares de columnas espejo
        sparse_output=False)

# Cada rama recibe su propia lista de columnas.
    preprocessor = ColumnTransformer([
        ("num", num_transformer, COL_NUMERICAS),
        ("cat", cat_transformer, COL_CATEGORICAS) ])

    return Pipeline([
        ("unify_categories", FunctionTransformer(unificar_categorias)),
        ("features", preprocessor) ])

################################################################################

# Check 
if __name__ == "__main__":

    from src.data.load import cargar_datos, separar_variables
    from src.data.split import dividir_datos

    # Verificacion del pipeline
    df = cargar_datos()
    X, y = separar_variables(df)
    X_train, X_test, _, _ = dividir_datos(X, y)

    pipeline = construir_preprocesador()
    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test)

    print(f"Train shape: {X_train.shape} -> {X_train_proc.shape}")
    print(f"Test shape:  {X_test.shape} -> {X_test_proc.shape}")
    print(f"NaNs restantes: {pd.isna(X_train_proc).sum()}")
