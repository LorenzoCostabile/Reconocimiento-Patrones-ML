import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data(path_values, path_labels, delimiter=";"):
    X_train = pd.read_csv(path_values, delimiter=delimiter)
    Y_train = pd.read_csv(path_labels, delimiter=delimiter)
    return X_train, Y_train

def scale_data(X_train, method="MinMax"):
    if method == "MinMax":
        scaler = MinMaxScaler().set_output(transform='pandas')
    elif method == "Standard":
        scaler = StandardScaler().set_output(transform='pandas')
    scaler.fit(X_train)
    X_scl = scaler.transform(X_train)
    return X_scl

def convertir_etiquetas_a_numeros(Y_train):
    label_map = {label: i for i, label in enumerate(Y_train.iloc[:, 1].unique())}
    Y_color = Y_train.iloc[:, 1].map(label_map)
    return Y_color
