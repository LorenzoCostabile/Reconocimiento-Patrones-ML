import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE
import matplotlib.pyplot as plt
import argparse
from utils import load_data, scale_data, convertir_etiquetas_a_numeros

PATH_VALUES = "X_train.csv"
PATH_LABELS = "Y_train.csv"

def apply_embedding(X_sel, method):
    if method == "LLE":
        n_components = 2
        n_neighbors = 30
        random_state = 1234
        max_iter = 100

        embedding = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, max_iter=max_iter, random_state=random_state)
    
    elif method == "Isomap":
        n_components = 2
        n_neighbors = 10
        metric = 'cosine' #<-- 'cityblock', 'cosine', 'euclidean' , 'haversine' , 'l1' , 'l2' , 'manhattan' , 'nan_euclidean' 
        max_iter = 500

        embedding = Isomap(n_neighbors=n_neighbors, n_components=n_components, metric=metric, max_iter=max_iter)
    
    elif method == "TSNE":
        n_components = 2
        perplexity = 20
        metric = 'l1' #<-- 'cityblock', 'cosine', 'euclidean' , 'haversine' , 'l1' , 'l2' , 'manhattan' , 'nan_euclidean' 
        random_state = 1234
        max_iter = 500
        n_iter_without_progress=150

        embedding = TSNE(perplexity=perplexity, n_components=n_components, metric=metric, random_state=random_state, n_iter_without_progress=n_iter_without_progress)
    
    else:
        
        raise ValueError("Método de embedding no soportado.")
    
    embedding.set_output(transform='pandas')
    X_embedded = embedding.fit_transform(X_sel)
    col_names = [f'{method.lower()}{i+1}' for i in range(n_components)]
    X_embedded.columns=col_names
    
    return X_embedded

def plot_embedding(X_embedded, Y_train, method, Y_color):
    plt.figure(figsize=(5, 5))
    scatter = plt.scatter(X_embedded.iloc[:, 0], X_embedded.iloc[:, 1], c=Y_color, cmap='tab20', s=60, marker='.', alpha=0.6)
    plt.title(f'Visualización {method}')
    plt.colorbar(scatter, label='Clase')
    unique_labels = Y_train.iloc[:, 1].unique()
    for i, label in enumerate(unique_labels):
        plt.scatter([], [], c=[plt.cm.tab20(i/len(unique_labels))], label=label)
    plt.legend(title='Clases', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", "-e", type=str, default="LLE", choices=["LLE", "Isomap", "TSNE"])
    args = parser.parse_args()

    PATH_VALUES = "X_train.csv"
    PATH_LABELS = "Y_train.csv"

    X_train, Y_train = load_data(PATH_VALUES, PATH_LABELS)
    X_scaled = scale_data(X_train, method="Standard")

    cols_idx = range(1, len(X_scaled.columns))#[2,4,6,8] #<-- select columns (or just put the names in 'cols' and comment this line)
    cols  = X_scaled.columns[cols_idx] 
    X_sel = X_scaled[cols]

    Y_color = convertir_etiquetas_a_numeros(Y_train)

    X_embedded = apply_embedding(X_sel, method=args.embedding)
    plot_embedding(X_embedded, Y_train, method=args.embedding, Y_color=Y_color)

