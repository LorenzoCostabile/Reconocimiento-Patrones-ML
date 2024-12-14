from utils import load_data, scale_data
from pca import perform_pca, plot_2d, plot_variance
from filtrado import variance_filter
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

def feature_varianza(X_train):
    # Añadir la varianza de cada fila de X_train como una nueva columna
    values = X_train.var(axis=1)
    return pd.DataFrame({'varianza':values})

class FeatureVarianzaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return feature_varianza(X)

def feature_media(X_train):
    # Añadir la media de cada fila de X_train como una nueva columna
    values = X_train.mean(axis=1)
    return pd.DataFrame({'media':values})

class FeatureMediaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return feature_media(X)

def feature_desviacion_tipica(X_train):
    # Añadir la desviación típica de cada fila de X_train como una nueva columna
    values = X_train.std(axis=1)
    return pd.DataFrame({'desviacion_tipica':values})

class FeatureDesviacionTipicaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return feature_desviacion_tipica(X)

def feature_mediana(X_train):
    # Añadir la mediana de cada fila de X_train como una nueva columna
    values = X_train.median(axis=1)
    return pd.DataFrame({'mediana':values})

class FeatureMedianaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return feature_mediana(X)


def feature_rango(X_train):
    # Añadir el rango (max - min) de cada fila de X_train como una nueva columna
    values = X_train.max(axis=1) - X_train.min(axis=1)
    return pd.DataFrame({'rango':values})

class FeatureRangoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        return feature_rango(X)
    

def add_features(X_train):
    X_train_original = X_train.copy()
    X_train['varianza'] = feature_varianza(X_train_original)
    X_train['media'] = feature_media(X_train_original)
    X_train['desviacion_tipica'] = feature_desviacion_tipica(X_train_original)
    X_train['mediana'] = feature_mediana(X_train_original)
    X_train['rango'] = feature_rango(X_train_original)
    return X_train

def only_new_features(X_train):
    features = pd.DataFrame()
    features['varianza'] = feature_varianza(X_train)
    features['media'] = feature_media(X_train) 
    features['desviacion_tipica'] = feature_desviacion_tipica(X_train)
    features['mediana'] = feature_mediana(X_train)
    features['rango'] = feature_rango(X_train)
    return features

if __name__ == "__main__":

    PATH_DATA_VALUES = "X_train.csv"
    PATH_DATA_LABELS = "Y_train.csv"

    X_train, y_train = load_data(PATH_DATA_VALUES, PATH_DATA_LABELS)
    
    X_train = only_new_features(X_train)


    X_train = scale_data(X_train)
    X_train = variance_filter(X_train, threshold=0.03)

    X_train_pca, pca = perform_pca(X_train, n_components=2)
    plot_2d(X_train_pca, y_train)




