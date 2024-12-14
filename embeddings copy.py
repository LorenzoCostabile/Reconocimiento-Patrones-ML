import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE
import matplotlib.pyplot as plt
import argparse

PATH_VALUES = "X_train.csv"
PATH_LABELS = "Y_train.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", "-e", type=str, default="LLE", choices=["LLE", "Isomap", "TSNE"])
    args = parser.parse_args()

    X_train = pd.read_csv(PATH_VALUES, delimiter=";")
    Y_train = pd.read_csv(PATH_LABELS, delimiter=";")

    #scaler = MinMaxScaler().set_output(transform='pandas')
    scaler = StandardScaler().set_output(transform='pandas')
    scaler.fit(X_train)
    X_scl = scaler.transform(X_train)

    cols_idx = range(1, len(X_scl.columns))#[2,4,6,8] #<-- select columns (or just put the names in 'cols' and comment this line)
    cols  = X_scl.columns[cols_idx] 
    X_sel = X_scl[cols] 

    print(X_sel.shape)
    # Convertir las etiquetas a valores numéricos para el colormap
    label_map = {label: i for i, label in enumerate(Y_train.iloc[:, 1].unique())}
    Y_color = Y_train.iloc[:, 1].map(label_map)

    if args.embedding == "LLE":
        #LLE locally linear embedding
        n_components = 2
        n_neighbors = 30
        random_state = 1234
        max_iter = 100

        lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components,
                                    max_iter=max_iter, random_state=random_state)
        lle.set_output(transform='pandas')
        lle.fit(X_sel)
        X_lle = lle.transform(X_sel)
        col_names = ['LLE%d'%(col+1) for col in range(n_components)]
        X_lle.columns=col_names

        print(X_lle.head())


        # Crear un scatter plot con una paleta de colores más vibrante
        plt.figure(figsize=(5, 5))
        scatter = plt.scatter(X_lle['LLE1'], X_lle['LLE2'], 
                            c=Y_color, 
                            cmap='tab20',  # Usar una paleta más colorida
                            s=60, 
                            marker='.', 
                            alpha=0.6)  # Aumentar la opacidad

        # Añadir leyenda con los nombres de las clases
        unique_labels = Y_train.iloc[:, 1].unique()
        for i, label in enumerate(unique_labels):
            plt.scatter([], [], c=[plt.cm.tab20(i/len(unique_labels))], label=label)
        plt.legend(title='Clases', bbox_to_anchor=(1.15, 1))

        plt.colorbar(scatter, label='Clase')
        plt.tight_layout()
        plt.show()

    if args.embedding == "Isomap":

        #Isomap
        n_components = 2
        n_neighbors = 10
        metric = 'cosine' #<-- 'cityblock', 'cosine', 'euclidean' , 'haversine' , 'l1' , 'l2' , 'manhattan' , 'nan_euclidean' 
        max_iter = 500

        isom = Isomap(n_neighbors=n_neighbors, n_components=n_components, metric=metric,
                    max_iter=max_iter)
        isom.set_output(transform='pandas')
        isom.fit(X_sel)
        X_isom = isom.transform(X_sel)
        col_names = ['isomap%d'%(col+1) for col in range(n_components)]
        X_isom.columns=col_names

        print(X_isom.head())

        #Dibujar Isomap
        plt.figure(figsize=(5, 5))
        scatter = plt.scatter(X_isom['isomap1'], X_isom['isomap2'], 
                            c=Y_color, 
                            cmap='tab20',  # Usar una paleta más colorida
                            s=60, 
                            marker='.', 
                            alpha=0.6)  # Aumentar la opacidad
        plt.show()

    if args.embedding == "TSNE":

        #TSNE
        n_components = 2
        perplexity = 20
        metric = 'l1' #<-- 'cityblock', 'cosine', 'euclidean' , 'haversine' , 'l1' , 'l2' , 'manhattan' , 'nan_euclidean' 
        random_state = 1234
        max_iter = 500
        n_iter_without_progress=150
        tsn = TSNE(perplexity=perplexity, n_components=n_components, metric=metric,
                random_state=random_state, n_iter_without_progress=n_iter_without_progress)
        tsn.set_output(transform='pandas')
        X_tsn = tsn.fit_transform(X_sel)
        col_names = ['tsne%d'%(col+1) for col in range(n_components)]
        X_tsn.columns=col_names

        X_tsn.head()

        #Dibujar TSNE
        plt.figure(figsize=(5, 5))
        scatter = plt.scatter(X_tsn['tsne1'], X_tsn['tsne2'], 
                            c=Y_color, 
                            cmap='tab20',  # Usar una paleta más colorida
                            s=60, 
                            marker='.', 
                            alpha=0.6)  # Aumentar la opacidad
        plt.show()




