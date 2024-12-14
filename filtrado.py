import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from utils import load_data, scale_data
from pca import perform_pca, plot_2d, plot_variance

def variance_filter(X, threshold=0.02):
    selector = VarianceThreshold(threshold).set_output(transform='pandas')
    selector.fit(X)
    X_sel = selector.transform(X)
    removed_features = list(set(X.columns) - set(X_sel.columns))
    print(f'{len(removed_features)} features have been removed: {removed_features}')
    return X_sel


if __name__ == "__main__":
    PATH_VALUES = "X_train.csv"
    PATH_LABELS = "Y_train.csv"
    
    X_train, Y_train = load_data(PATH_VALUES, PATH_LABELS)
    X_scaled = scale_data(X_train, method='unit')
    X_selected = variance_filter(X_scaled, threshold=0.03)
    X_pca, pca = perform_pca(X_selected, n_components=2)
    plot_variance(pca)
    plot_2d(X_pca, Y_train)




