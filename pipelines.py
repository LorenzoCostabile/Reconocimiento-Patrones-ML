from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from newfeatures import FeatureDesviacionTipicaTransformer, FeatureMediaTransformer, FeatureVarianzaTransformer, FeatureRangoTransformer, FeatureMedianaTransformer, feature_varianza, feature_media, feature_desviacion_tipica, feature_rango, feature_mediana
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import LocallyLinearEmbedding
import pandas as pd

class OperacionIdentidad(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

pipe_sin_nada = Pipeline([('operacion_identidad', OperacionIdentidad())])

pipe_new_features = FeatureUnion([('media', FeatureMediaTransformer()),
                                  ('varianza', FeatureVarianzaTransformer()),
                                  ('desviacion_tipica', FeatureDesviacionTipicaTransformer()),
                                  ('rango', FeatureRangoTransformer()),
                                  ('mediana', FeatureMedianaTransformer())])

class MantenerNuevasFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def only_new_features(X):
            print(X.head())
            features = pd.DataFrame()
            features['varianza'] = feature_varianza(X)
            features['media'] = feature_media(X)
            features['desviacion_tipica'] = feature_desviacion_tipica(X)
            features['mediana'] = feature_mediana(X)
            features['rango'] = feature_rango(X)
            return features
        
        return only_new_features(X)

class VisualizarNuevasFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(X.head())
        return X

pipe_only_new_features = Pipeline([('mantener_nuevas_features', MantenerNuevasFeatures()),
                                  ('visualizar_nuevas_features', VisualizarNuevasFeatures()),
                                  ('scaler', MinMaxScaler())])

pipe_last_embedding = [('scaler', MinMaxScaler())]
n_components = 2
n_neighbors = 40
random_state = 1234
max_iter = 200
pipe_last_embedding.append((
    'lle',
        LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components,
                            max_iter=max_iter, random_state=random_state).set_output(transform='pandas') )
)
pipe_last_embedding = Pipeline(pipe_last_embedding)

pipe_last_pca = Pipeline([('scaler', MinMaxScaler()),
                      ('pca', PCA(n_components=3))])

pipe_completa_con_embedding = Pipeline([('pipe1', pipe_new_features),
                          ('pipe2', pipe_last_embedding)])

pipe_completa_con_pca = Pipeline([('pipe1', pipe_new_features),
                          ('pipe2', pipe_last_pca)])

