from sklearn.preprocessing import PolynomialFeatures
from utils import load_data, scale_data
from pca import plot_2d, perform_pca, plot_variance

def apply_polynomial_features(X_train, poly_degree):
    polyfeat = PolynomialFeatures(poly_degree).set_output(transform="pandas")
    return polyfeat.fit_transform(X_train)

if __name__ == "__main__":
    PATH_VALUES = "X_train.csv"
    PATH_LABELS = "Y_train.csv"

    X_train, Y_train = load_data(PATH_VALUES, PATH_LABELS)
    X_poly = apply_polynomial_features(X_train, poly_degree=2)
    X_scaled = scale_data(X_poly)

    X_pca, pca = perform_pca(X_scaled, n_components=2)
    plot_2d(X_pca, Y_train)
    plot_variance(pca)

