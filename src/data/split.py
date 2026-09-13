from sklearn.model_selection import train_test_split

from src.data.load import cargar_datos, separar_variables

SEMILLA = 42
TEST_SIZE = 0.2

def dividir_datos(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=SEMILLA,
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = cargar_datos()
    X, y = separar_variables(df)
    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    print("Train:", X_train.shape, "| churn:", round(y_train.mean(), 4))
    print("Test: ", X_test.shape, "| churn:", round(y_test.mean(), 4))