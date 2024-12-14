import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import load_data, scale_data
from sklearn.base import BaseEstimator, TransformerMixin

def perform_pca(X_scl, n_components):
    col_names = ['pc%d' % (col + 1) for col in range(n_components)]
    pca = PCA(n_components=n_components)
    pca.fit(X_scl)
    X_pca = pd.DataFrame(pca.transform(X_scl), columns=col_names)
    return X_pca, pca

def plot_variance(pca):
    plt.stem(np.cumsum(pca.explained_variance_ratio_), 'r')
    plt.stem(pca.explained_variance_ratio_, 'xb')
    plt.show()

def plot_2d(X_pca, Y_train):
    plt.figure(figsize=(10, 8))
    label_map = {label: i for i, label in enumerate(Y_train.iloc[:, 1].unique())}
    numeric_labels = Y_train.iloc[:, 1].map(label_map)
    plt.scatter(X_pca['pc1'], X_pca['pc2'], c=numeric_labels, cmap='tab20', alpha=0.8)
    plt.xlabel('Primera Componente Principal')
    plt.ylabel('Segunda Componente Principal')
    plt.title('Visualización PCA en 2D')
    plt.colorbar(label='Clase')
    plt.grid(True, alpha=0.3)
    unique_labels = Y_train.iloc[:, 1].unique()
    for i, label in enumerate(unique_labels):
        plt.scatter([], [], c=[plt.cm.tab20(i/len(unique_labels))], label=label)
    plt.legend(title='Clases', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()

def plot_3d(X_pca, Y_train):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    label_map = {label: i for i, label in enumerate(Y_train.iloc[:, 1].unique())}
    numeric_labels = Y_train.iloc[:, 1].map(label_map)
    scatter = ax.scatter(X_pca['pc1'], X_pca['pc2'], X_pca['pc3'], 
                         c=numeric_labels, cmap='tab20', alpha=0.8)
    ax.set_xlabel('Primera Componente Principal')
    ax.set_ylabel('Segunda Componente Principal')
    ax.set_zlabel('Tercera Componente Principal')
    ax.set_title('Visualización PCA en 3D')
    unique_labels = Y_train.iloc[:, 1].unique()
    for i, label in enumerate(unique_labels):
        ax.scatter([], [], [], c=[plt.cm.tab20(i/len(unique_labels))], label=label)
    ax.legend(title='Clases', bbox_to_anchor=(1.15, 1))
    plt.colorbar(scatter, label='Clase')
    plt.tight_layout()
    plt.show()

class PCA_pipeline(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.X_pca, self.pca = perform_pca(X, self.n_components)
        return self

    def transform(self, X):
        return self.X_pca

if __name__ == "__main__":
    # Ejemplo de uso
    path_values = "X_train.csv"
    path_labels = "Y_train.csv"
    X_train, Y_train = load_data(path_values, path_labels)
    X_scaled = scale_data(X_train)
    X_pca, pca = perform_pca(X_scaled, n_components=3)
    plot_variance(pca)
    plot_2d(X_pca, Y_train)
    plot_3d(X_pca, Y_train)