from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from pipelines import pipe_only_new_features
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
import joblib

PATH_DATA_VALUES = 'datos_separados/X_train.csv'
PATH_DATA_LABELS = 'datos_separados/Y_train.csv'

X_train = pd.read_csv(PATH_DATA_VALUES, delimiter=';')
# Eliminar la columna id, la primera
X_train = X_train.iloc[:,1:]

Y_train = pd.read_csv(PATH_DATA_LABELS,delimiter=';')
Y_train = Y_train.iloc[:,1:]

TAMANIO_INPUT = X_train.shape[1]
NUM_CLASSES = 5


hidden_layer_sizes=[TAMANIO_INPUT,64,32,NUM_CLASSES]
# Es una red que devuelve la etiqueta en onehotencoding
activation='logistic'
learning_rate='adaptive'
learning_rate_init=0.001
max_iter=1000

pipe_Clf = Pipeline([('clf_NN', MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                      activation=activation, learning_rate=learning_rate,
                      learning_rate_init=learning_rate_init, max_iter=max_iter,
                      verbose=True))])


pipeline_entrenamiento = Pipeline([('pipe_only_new_features', pipe_only_new_features),
                                   ('pipe_Clf', pipe_Clf)])

#convertir a onehotencoding
Y_train = pd.get_dummies(Y_train)
print(Y_train.head())

# Configurar K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Realizar validación cruzada
cv_results = cross_validate(pipeline_entrenamiento, X_train, Y_train, cv=kf, return_train_score=True)

# Imprimir resultados de la validación cruzada
print("Resultados de la validación cruzada:")
print("Train scores:", cv_results['train_score'])
print("Test scores:", cv_results['test_score'])

# Guardar el modelo entrenado en el último fold
pipeline_entrenamiento.fit(X_train, Y_train)
joblib.dump(pipeline_entrenamiento, 'modelo_nn.pkl')

# Cargar datos de prueba
PATH_DATA_TEST_VALUES = 'datos_separados/X_test.csv'
PATH_DATA_TEST_LABELS = 'datos_separados/Y_test.csv'
X_test = pd.read_csv(PATH_DATA_TEST_VALUES, delimiter=';')
Y_test = pd.read_csv(PATH_DATA_TEST_LABELS, delimiter=';')
X_test = X_test.iloc[:,1:]  # Eliminar la columna id, la primera
Y_test = Y_test.iloc[:,1:]
Y_test = pd.get_dummies(Y_test)

# Evaluar el modelo en los datos de prueba
test_score = pipeline_entrenamiento.score(X_test, Y_test)
print("Test score en datos de prueba:", test_score)


